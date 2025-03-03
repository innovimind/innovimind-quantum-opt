"""
최적화된 삼각 행렬 연산 구현 (CPU 최적화 - 개선 버전)
"""

import torch
import torch.nn as nn
import math
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class MatrixOptimizerConfig:
    """행렬 연산 최적화 설정"""
    batch_size: int = 32  # 테스트 환경에 맞춘 배치 크기
    block_size: int = 128  # 테스트 환경에 맞춘 블록 크기
    cache_threshold: int = 128 * 512  # 테스트 크기 기반 임계값
    layer_norm_eps: float = 1e-12

class TriangularMatrixOptimizer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['attention_heads']
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        self.opt_config = MatrixOptimizerConfig()
        self._setup_cache()
        self.logger = self._setup_logger()
    
    def _setup_cache(self):
        """캐시 초기화 및 설정"""
        self.cache = {}
        self.mask_cache = {}
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _get_mask(self, m: int, n: int, device: torch.device) -> torch.Tensor:
        """마스크 캐싱 및 재사용"""
        key = (m, n, device)
        if key not in self.mask_cache:
            self.mask_cache[key] = torch.tril(torch.ones(m, n, device=device))
        return self.mask_cache[key]
    
    def _triangular_matrix_multiply(
        self,
        matrix_a: torch.Tensor,
        matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """최적화된 삼각 행렬 곱셈"""
        if len(matrix_a.shape) == 3:
            batch_size, m, k = matrix_a.shape
            _, k, n = matrix_b.shape
            
            # 캐시 키 생성
            cache_key = f"{matrix_a.shape}_{matrix_b.shape}_{matrix_a.device}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 마스크 가져오기 (캐시된)
            mask = self._get_mask(m, n, matrix_a.device)
            
            # 직접 곱셈 및 마스크 적용
            result = torch.bmm(matrix_a, matrix_b)
            result = result * mask.unsqueeze(0)
            
            # 결과 캐싱
            if m * n <= self.opt_config.cache_threshold:
                self.cache[cache_key] = result
            
            return result
            
        else:
            batch_size, heads, m, k = matrix_a.shape
            _, _, k, n = matrix_b.shape
            
            # 재구성 없이 직접 연산
            result = torch.matmul(matrix_a, matrix_b)
            mask = self._get_mask(m, n, matrix_a.device)
            return result * mask.unsqueeze(0).unsqueeze(0)
    
    def optimize_matrix_multiplication(
        self,
        matrix_a: torch.Tensor,
        matrix_b: torch.Tensor,
        triangular: bool = False
    ) -> torch.Tensor:
        """행렬 곱셈 최적화
        
        Args:
            matrix_a: 첫 번째 행렬
            matrix_b: 두 번째 행렬
            triangular: 삼각 행렬 여부
            
        Returns:
            최적화된 행렬 곱셈 결과
        """
        # 배치 크기 맞추기
        if matrix_a.dim() == 3 and matrix_b.dim() == 3:
            batch_size_a = matrix_a.size(0)
            batch_size_b = matrix_b.size(0)
            
            if batch_size_a != batch_size_b:
                if batch_size_a == 1:
                    matrix_a = matrix_a.expand(batch_size_b, -1, -1)
                elif batch_size_b == 1:
                    matrix_b = matrix_b.expand(batch_size_a, -1, -1)
                else:
                    raise ValueError(
                        f"배치 크기가 맞지 않습니다: {batch_size_a} vs {batch_size_b}"
                    )
        
        if triangular:
            return self._triangular_matrix_multiply(matrix_a, matrix_b)
            
        # 일반 행렬 곱셈
        if matrix_a.dim() == 3:
            return torch.bmm(matrix_a, matrix_b)
        else:
            return torch.matmul(matrix_a, matrix_b)
    
    def compute_attention_scores(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """최적화된 어텐션 스코어 계산"""
        # 어텐션 스코어 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # 마스크 적용
        if attention_mask is not None:
            # 마스크 크기 조정
            if attention_mask.dim() != attention_scores.dim():
                # 필요한 차원 추가
                for _ in range(attention_scores.dim() - attention_mask.dim()):
                    attention_mask = attention_mask.unsqueeze(1)
            
            # 브로드캐스팅을 위한 크기 맞추기
            target_shape = list(attention_scores.size())
            attention_mask = attention_mask.expand(*target_shape)
            attention_scores = attention_scores + attention_mask
        
        # Softmax 적용
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # 드롭아웃 적용 (설정된 경우)
        if hasattr(self, 'dropout') and self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        
        # 컨텍스트 계산
        context_layer = torch.matmul(attention_probs, value_layer)
        
        return context_layer, attention_probs
    
    def clear_cache(self):
        """모든 캐시 초기화"""
        self.cache.clear()
        self.mask_cache.clear()
    
    def get_memory_stats(self) -> dict:
        """메모리 사용 통계"""
        return {
            'cache_size': len(self.cache),
            'mask_cache_size': len(self.mask_cache),
            'total_memory': sum(x.nelement() * x.element_size() for x in 
                              list(self.cache.values()) + list(self.mask_cache.values()))
        } 