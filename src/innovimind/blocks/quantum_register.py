import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuantumRegister(nn.Module):
    def __init__(self):
        super().__init__()
        # 위상 스케일링 파라미터
        self.phase_scale = nn.Parameter(torch.tensor(1.0))
        self.min_scale = 0.85
        self.max_scale = 1.25
        
        # 위상 스무딩 파라미터
        self.phase_smoothing = 0.5
        self.min_smoothing = 0.3
        self.max_smoothing = 0.7
        
        # 가우시안 커널 초기화
        kernel_size = 11
        sigma = 2.5
        x = torch.linspace(-5, 5, kernel_size)
        kernel = torch.exp(-x**2 / (2*sigma**2))
        self.register_buffer('smoothing_kernel', kernel / kernel.sum())
        
        # 노이즈 임계값
        self.noise_threshold = 0.05 * np.pi

    def prepare_normalized(self, x):
        """입력 텐서를 정규화된 양자 상태로 변환
        
        Args:
            x: 입력 텐서 (batch_size, seq_len, num_heads, head_dim) 또는
               (batch_size, num_heads, seq_len, head_dim)
        
        Returns:
            정규화된 양자 상태 (batch_size, num_heads, seq_len, head_dim)
        """
        # 1. 차원 순서 변경 (필요한 경우)
        if x.shape[1] != self.num_heads:
            x = x.transpose(1, 2)
            
        # 2. L2 정규화
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        x = x / (norm + 1e-8)
        
        # 3. 위상 보존 처리
        if x.is_complex():
            phase = torch.angle(x)
            phase = phase * torch.clamp(self.phase_scale, self.min_scale, self.max_scale)
            magnitude = torch.abs(x)
            x = magnitude * torch.exp(1j * phase)
            
            # 위상 스무딩 적용
            smoothing_strength = torch.clamp(self.phase_smoothing, 
                                          self.min_smoothing, 
                                          self.max_smoothing)
            phase_noise = torch.abs(phase - torch.mean(phase, dim=-1, keepdim=True))
            noise_mask = phase_noise > self.noise_threshold
            
            if noise_mask.any():
                # 3단계 스무딩
                x_smooth = x
                for _ in range(3):
                    phase_smooth = torch.angle(x_smooth)
                    phase_smooth = F.conv1d(
                        phase_smooth.reshape(-1, 1, x.size(-1)),
                        self.smoothing_kernel.view(1, 1, -1),
                        padding='same'
                    ).reshape(x.shape)
                    magnitude_smooth = torch.abs(x_smooth)
                    x_smooth = magnitude_smooth * torch.exp(1j * phase_smooth)
                    x = torch.where(noise_mask.unsqueeze(-1), 
                                  x_smooth, 
                                  x)
        
        return x

    def entangle(self, state1, state2):
        """두 양자 상태의 얽힘 생성
        
        Args:
            state1: 첫 번째 양자 상태 (batch_size, num_heads, seq_len, head_dim)
            state2: 두 번째 양자 상태 (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            얽힘 상태 (batch_size, num_heads, seq_len, seq_len)
        """
        # 1. 상태 정규화
        state1 = self.prepare_normalized(state1)
        state2 = self.prepare_normalized(state2)
        
        # 2. 외적 기반 얽힘 연산
        entangled = torch.einsum('bhid,bhjd->bhij', state1, state2.conj())
        
        # 3. 얽힘 강도 조절 (0.8)
        entangled = entangled * 0.8
        
        # 4. 정규화
        norm = torch.norm(entangled, p=2, dim=(-2, -1), keepdim=True)
        entangled = entangled / (norm + 1e-8)
        
        return entangled 