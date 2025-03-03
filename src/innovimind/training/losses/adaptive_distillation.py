import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class AdaptiveDistillationLoss(nn.Module):
    """적응형 지식 증류 손실 함수
    
    학생 모델의 효율적인 학습을 위해 교사 모델의 지식을 전달하면서
    동시에 태스크별 손실을 고려하는 적응형 손실 함수
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.temperature = config["training"]["loss"]["temperature"]
        self.alpha = config["training"]["loss"]["alpha"]
        self.label_smoothing = config["training"]["loss"]["label_smoothing"]
        
    def forward(
        self,
        student_outputs: torch.Tensor,
        targets: torch.Tensor,
        teacher_outputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """손실 계산
        
        Args:
            student_outputs: 학생 모델의 출력 (batch_size, seq_len, vocab_size)
            targets: 정답 레이블 (batch_size, seq_len)
            teacher_outputs: 교사 모델의 출력 (batch_size, seq_len, vocab_size)
            mask: 패딩 마스크 (batch_size, seq_len)
            
        Returns:
            각 손실 컴포넌트를 포함하는 딕셔너리
        """
        # 기본 태스크 손실 (레이블 스무딩 적용)
        task_loss = self._compute_task_loss(student_outputs, targets, mask)
        
        # 증류 손실 (교사 모델이 있는 경우)
        distillation_loss = torch.tensor(0.0, device=student_outputs.device)
        if teacher_outputs is not None:
            distillation_loss = self._compute_distillation_loss(
                student_outputs, teacher_outputs, mask
            )
        
        # 최종 손실 계산
        total_loss = (1 - self.alpha) * task_loss + self.alpha * distillation_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss,
            'distillation_loss': distillation_loss
        }
    
    def _compute_task_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """태스크 손실 계산 (레이블 스무딩 적용)"""
        vocab_size = outputs.size(-1)
        
        # 레이블 스무딩 적용
        smooth_targets = torch.zeros_like(outputs)
        smooth_targets.fill_(self.label_smoothing / (vocab_size - 1))
        smooth_targets.scatter_(-1, targets.unsqueeze(-1), 1 - self.label_smoothing)
        
        # 크로스 엔트로피 계산
        loss = -torch.sum(smooth_targets * F.log_softmax(outputs, dim=-1), dim=-1)
        
        # 마스킹 적용
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
            
        return loss
    
    def _compute_distillation_loss(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """지식 증류 손실 계산"""
        # 소프트 타겟 생성
        student_logits = student_outputs / self.temperature
        teacher_logits = teacher_outputs / self.temperature
        
        # KL 발산 계산
        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        
        # 온도에 따른 스케일링
        loss = loss * (self.temperature ** 2)
        
        # 마스킹 적용
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
            
        return loss
    
    def _compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """평가 지표 계산"""
        # 예측값 추출
        predictions = outputs.argmax(dim=-1)
        
        # 정확도 계산
        correct = (predictions == targets)
        if mask is not None:
            correct = correct * mask
            accuracy = correct.sum() / mask.sum()
        else:
            accuracy = correct.float().mean()
            
        return {
            'accuracy': accuracy
        } 