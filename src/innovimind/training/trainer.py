import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
import yaml
from pathlib import Path
import time
from tqdm import tqdm

from .losses.adaptive_distillation import AdaptiveDistillationLoss

class Trainer:
    """모델 학습 관리자"""
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config_path: str,
        teacher_model: Optional[nn.Module] = None
    ):
        self.model = model
        self.teacher_model = teacher_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()
        
        # 옵티마이저 설정
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 손실 함수 설정
        self.criterion = AdaptiveDistillationLoss(self.config)
        
        # 혼합 정밀도 학습 설정
        self.scaler = GradScaler(enabled=self.config["training"]["mixed_precision"]["enabled"])
        
        # 체크포인트 관리
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 메트릭 기록
        self.train_metrics: Dict[str, float] = {}
        self.val_metrics: Dict[str, float] = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """옵티마이저 생성"""
        optimizer_config = self.config["training"]["optimizer"]
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
            betas=(optimizer_config["beta1"], optimizer_config["beta2"]),
            eps=optimizer_config["eps"]
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """학습률 스케줄러 생성"""
        scheduler_config = self.config["training"]["scheduler"]
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=scheduler_config["warmup_steps"],
            T_mult=2,
            eta_min=scheduler_config["min_lr_ratio"] * scheduler_config["learning_rate"]
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        with tqdm(self.train_loader, desc="Training") as pbar:
            for batch_idx, (features, targets) in enumerate(pbar):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # 교사 모델 출력 계산 (있는 경우)
                teacher_outputs = None
                if self.teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(features)
                
                # 혼합 정밀도 학습
                with autocast(enabled=self.config["training"]["mixed_precision"]["enabled"]):
                    # 순전파
                    outputs = self.model(features)
                    
                    # 손실 계산
                    loss_dict = self.criterion(
                        outputs,
                        targets,
                        teacher_outputs=teacher_outputs
                    )
                    loss = loss_dict['total_loss']
                
                # 역전파
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 그래디언트 클리핑
                if self.config["training"]["gradient_clipping"]["enabled"]:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clipping"]["max_norm"]
                    )
                
                # 옵티마이저 스텝
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # 스케줄러 스텝
                self.scheduler.step()
                
                # 메트릭 업데이트
                total_loss += loss.item() * features.size(0)
                total_samples += features.size(0)
                
                # 프로그레스바 업데이트
                pbar.set_postfix({
                    'loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0]
                })
                
                # 디버깅 체크
                if self.config["training"]["debugging"]["enabled"]:
                    self._check_debugging(loss_dict, outputs)
        
        # 에폭 메트릭 계산
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'lr': self.scheduler.get_last_lr()[0]
        }
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증 수행"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with tqdm(self.val_loader, desc="Validation") as pbar:
            for features, targets in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # 교사 모델 출력 계산 (있는 경우)
                teacher_outputs = None
                if self.teacher_model is not None:
                    teacher_outputs = self.teacher_model(features)
                
                # 순전파
                outputs = self.model(features)
                
                # 손실 계산
                loss_dict = self.criterion(
                    outputs,
                    targets,
                    teacher_outputs=teacher_outputs
                )
                
                # 메트릭 업데이트
                total_loss += loss_dict['total_loss'].item() * features.size(0)
                total_samples += features.size(0)
                
                # 프로그레스바 업데이트
                pbar.set_postfix({'loss': loss_dict['total_loss'].item()})
        
        # 검증 메트릭 계산
        val_metrics = {
            'loss': total_loss / total_samples
        }
        
        return val_metrics
    
    def train(self, num_epochs: int) -> None:
        """전체 학습 수행"""
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # 학습
            train_metrics = self.train_epoch()
            self.train_metrics = train_metrics
            
            # 검증
            val_metrics = self.validate()
            self.val_metrics = val_metrics
            
            # 로깅
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # 체크포인트 저장
            if self._should_save_checkpoint(val_metrics['loss']):
                self._save_checkpoint(epoch, val_metrics)
            
            # 조기 종료 체크
            if self._should_stop_training():
                self.logger.info("Early stopping triggered")
                break
    
    def _check_debugging(self, loss_dict: Dict[str, torch.Tensor], outputs: torch.Tensor) -> None:
        """디버깅 체크 수행"""
        if torch.isnan(loss_dict['total_loss']):
            self.logger.error("NaN loss detected")
            raise ValueError("NaN loss detected")
            
        if self.config["training"]["debugging"]["gradient_check"]:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        self.logger.error(f"Invalid gradient norms detected in {name}")
                        raise ValueError(f"Invalid gradient norms detected in {name}")
    
    def _should_save_checkpoint(self, val_loss: float) -> bool:
        """체크포인트 저장 여부 결정"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False
    
    def _should_stop_training(self) -> bool:
        """조기 종료 여부 결정"""
        return (
            self.config["training"]["early_stopping"]["enabled"] and
            self.patience_counter >= self.config["training"]["early_stopping"]["patience"]
        )
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        save_path = Path(self.config["checkpointing"]["save_dir"]) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, save_path)
        self.logger.info(f"Checkpoint saved: {save_path}")
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """메트릭 로깅"""
        self.logger.info(
            f"Epoch {epoch+1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"LR: {train_metrics['lr']:.6f}"
        ) 