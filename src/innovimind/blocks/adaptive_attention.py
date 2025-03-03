import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
import time
import sys
import hashlib
from collections import defaultdict
from .optimized_matrix import TriangularMatrixOptimizer

from ..research.adaptive_attention_analysis import AttentionComplexityAnalyzer

class FastCache:
    """최적화된 캐시 시스템"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.stats = defaultdict(int)
        self.max_size = max_size
        self.access_times = {}
        self.batch_stats = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, x: torch.Tensor) -> str:
        """경량화된 캐시 키 생성"""
        with torch.no_grad():
            # 배치 독립적 특징 추출
            shape_key = '_'.join(map(str, x.shape))
            
            # 효율적인 통계 계산
            mean = x.float().mean().item()
            std = x.float().std().item()
            
            # 빠른 해시 생성
            key = f"{shape_key}_{mean:.4f}_{std:.4f}"
            return hashlib.sha256(key.encode()).hexdigest()[:8]
    
    def get(self, key: str, batch_size: int) -> Optional[Tuple[torch.Tensor, Dict]]:
        """캐시 조회"""
        if key in self.cache:
            self.stats[key] += 1
            self.access_times[key] = time.time()
            self.hits += 1
            
            if batch_size not in self.batch_stats:
                self.batch_stats[batch_size] = {'hits': 0, 'misses': 0}
            self.batch_stats[batch_size]['hits'] += 1
            
            return self.cache[key]
            
        self.misses += 1
        if batch_size in self.batch_stats:
            self.batch_stats[batch_size]['misses'] += 1
        return None
    
    def put(self, key: str, value: Tuple[torch.Tensor, Dict], batch_size: int):
        """캐시 저장"""
        if len(self.cache) >= self.max_size:
            # LFU + LRU 하이브리드 정책
            candidates = sorted(
                self.stats.items(),
                key=lambda x: (x[1], self.access_times.get(x[0], 0))
            )[:len(self.cache)//10]
            
            for old_key, _ in candidates:
                del self.cache[old_key]
                del self.stats[old_key]
                if old_key in self.access_times:
                    del self.access_times[old_key]
        
        self.cache[key] = value
        self.stats[key] = 1
        self.access_times[key] = time.time()
        
        if batch_size not in self.batch_stats:
            self.batch_stats[batch_size] = {'hits': 0, 'misses': 1}
    
    def get_stats(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """캐시 통계"""
        total = self.hits + self.misses
        stats = {
            'size': len(self.cache),
            'total_accesses': total,
            'unique_keys': len(self.stats),
            'hits': self.hits,
            'misses': self.misses
        }
        
        if total > 0:
            stats['hit_rate'] = self.hits / total
        else:
            stats['hit_rate'] = 0.0
        
        if batch_size is not None and batch_size in self.batch_stats:
            batch_total = sum(self.batch_stats[batch_size].values())
            if batch_total > 0:
                stats['batch_hit_rate'] = (
                    self.batch_stats[batch_size]['hits'] / batch_total
                )
            else:
                stats['batch_hit_rate'] = 0.0
        
        return stats

class OptimizedMemoryPool:
    """최적화된 메모리 풀"""
    def __init__(self, max_pool_size: int = 100):
        # 크기별 풀 초기화
        self.size_classes = [
            (4, 32),   # (배치 크기, 풀 크기)
            (8, 16),
            (16, 8),
            (32, 4)
        ]
        
        self.pools = {}
        self.stats = defaultdict(int)
        self.access_times = {}
        self.max_pool_size = max_pool_size
    
    def get_size_class(self, batch_size: int) -> Tuple[int, int]:
        """최적 크기 클래스 선택"""
        for size, pool_size in self.size_classes:
            if batch_size <= size:
                return size, pool_size
        return self.size_classes[-1]
    
    def get_tensor(self, batch_size: int, shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """최적화된 텐서 할당"""
        size_class, pool_size = self.get_size_class(batch_size)
        key = (size_class, shape, dtype)
        
        # 풀 초기화
        if key not in self.pools:
            self.pools[key] = []
            
            # 미리 할당
            for _ in range(pool_size):
                tensor = torch.empty(shape, dtype=dtype)
                self.pools[key].append(tensor)
        
        # 텐서 재사용
        if self.pools[key]:
            tensor = self.pools[key].pop()
            tensor.zero_()
        else:
            tensor = torch.empty(shape, dtype=dtype)
            
        self.stats[key] += 1
        self.access_times[id(tensor)] = time.time()
        
        return tensor
    
    def release_tensor(self, tensor: torch.Tensor, batch_size: int):
        """텐서 반환"""
        size_class, _ = self.get_size_class(batch_size)
        key = (size_class, tensor.shape, tensor.dtype)
        
        if key in self.pools:
            if len(self.pools[key]) < self.get_size_class(batch_size)[1]:
                self.pools[key].append(tensor.detach())
                
        if id(tensor) in self.access_times:
            del self.access_times[id(tensor)]
    
    def cleanup_old_tensors(self, max_age: float = 60.0):
        """오래된 텐서 정리"""
        current_time = time.time()
        for tensor_id, access_time in list(self.access_times.items()):
            if current_time - access_time > max_age:
                del self.access_times[tensor_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """풀 통계"""
        total_tensors = sum(len(pool) for pool in self.pools.values())
        total_allocations = sum(self.stats.values())
        
        return {
            'total_tensors': total_tensors,
            'total_allocations': total_allocations,
            'size_classes': len(self.size_classes),
            'pools': len(self.pools),
            'reuse_rate': total_tensors / max(1, total_allocations)
        }

class AdaptiveAttentionRouter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_types = config.get('num_attention_types', 3)
        
        # 라우팅 네트워크
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.num_attention_types),
            nn.Softmax(dim=-1)
        )
        
        # 어텐션 타입별 임계값
        self.attention_thresholds = nn.Parameter(
            torch.ones(self.num_attention_types) * 0.5
        )
        
        # 캐시
        self.routing_cache = {}
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        use_cache: bool = True
    ) -> torch.Tensor:
        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        # 캐시 키 생성
        cache_key = f"{batch_size}_{seq_length}"
        if use_cache and cache_key in self.routing_cache:
            return self.routing_cache[cache_key]
            
        # 시퀀스 평균으로 입력 특성 추출
        sequence_features = hidden_states.mean(dim=1)
        
        # 라우팅 확률 계산
        routing_logits = self.router(sequence_features)
        
        # 임계값 적용
        routing_weights = routing_logits * (routing_logits > self.attention_thresholds)
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-12)
        
        if use_cache:
            self.routing_cache[cache_key] = routing_weights
            
        return routing_weights
        
    def clear_cache(self):
        """캐시 초기화"""
        self.routing_cache.clear()

class AdaptiveAttentionBlock(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['attention_heads']
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 어텐션 타입별 가중치
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # 출력 변환
        self.output_transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=config['layer_norm_eps'])
        
        # 어텐션 스케일링
        self.attention_scale = math.sqrt(self.attention_head_size)
        
        # 캐시
        self.cache = {}
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
        
    def merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3)
        new_shape = x.size()[:-2] + (self.all_head_size,)
        return x.contiguous().view(*new_shape)
        
    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """어텐션 마스크 전처리
        
        Args:
            attention_mask: 원본 어텐션 마스크
            batch_size: 배치 크기
            seq_length: 시퀀스 길이
            dtype: 텐서 데이터 타입
            
        Returns:
            전처리된 어텐션 마스크
        """
        # 마스크 차원 확장
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
            
        # 마스크 크기 조정
        attention_mask = attention_mask.expand(
            batch_size,
            self.num_attention_heads,
            seq_length,
            seq_length
        )
        
        # 데이터 타입 변환
        attention_mask = attention_mask.to(dtype=dtype)
        
        # 마스킹을 위한 값 변환 (0 -> -10000, 1 -> 0)
        return (1.0 - attention_mask) * -10000.0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.size(0)
        seq_length = hidden_states.size(1)
        
        # 캐시 키 생성
        cache_key = f"{batch_size}_{seq_length}"
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
            
        # 쿼리, 키, 값 변환
        query_layer = self.split_heads(self.query(hidden_states))
        key_layer = self.split_heads(self.key(hidden_states))
        value_layer = self.split_heads(self.value(hidden_states))
        
        # 어텐션 스코어 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / self.attention_scale
        
        # 어텐션 마스크 처리
        if attention_mask is not None:
            # 마스크 전처리
            attention_mask = self._prepare_attention_mask(
                attention_mask,
                batch_size,
                seq_length,
                attention_scores.dtype
            )
            attention_scores = attention_scores + attention_mask
            
        # 어텐션 확률 계산
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 어텐션 가중치 적용
        if attention_weights is not None:
            attention_probs = attention_probs * attention_weights.unsqueeze(1).unsqueeze(2)
            
        attention_probs = self.dropout(attention_probs)
        
        # 컨텍스트 계산
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.merge_heads(context_layer)
        
        # 출력 변환
        attention_output = self.output_transform(context_layer)
        attention_output = self.dropout(attention_output)
        attention_output = self.layer_norm(attention_output + hidden_states)
        
        if use_cache:
            self.cache[cache_key] = (attention_output, attention_probs)
            
        return attention_output, attention_probs

class AdaptiveAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 어텐션 라우터
        self.router = AdaptiveAttentionRouter(config)
        
        # 어텐션 블록
        self.attention_blocks = nn.ModuleList([
            AdaptiveAttentionBlock(config)
            for _ in range(config.get('num_attention_types', 3))
        ])
        
        # 출력 결합
        self.output_combine = nn.Linear(
            config['hidden_size'] * config.get('num_attention_types', 3),
            config['hidden_size']
        )
        
        # 출력 변환
        self.output_transform = nn.Linear(config['hidden_size'], config['hidden_size'])
        
        # 드롭아웃
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        
        # 성능 메트릭스
        self.metrics = {
            'routing_decisions': [],
            'attention_weights': [],
            'cache_hits': 0,
            'total_calls': 0
        }
        
        # 최적화된 행렬 연산 추가
        self.matrix_optimizer = TriangularMatrixOptimizer(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        batch_size, seq_length = hidden_states.size()[:2]
        
        # 라우팅 가중치 계산
        routing_weights = self.router(hidden_states)
        
        # 어텐션 마스크 전처리
        if attention_mask is not None:
            attention_mask = self._prepare_attention_mask(
                attention_mask,
                batch_size,
                seq_length,
                hidden_states.dtype
            )
        
        # 각 어텐션 블록 처리
        block_outputs = []
        block_weights = []
        
        for i, block in enumerate(self.attention_blocks):
            # 블록별 어텐션 계산
            block_output, block_weight = block(
                hidden_states,
                attention_mask=attention_mask,
                attention_weights=routing_weights[:, i]
            )
            block_outputs.append(block_output)
            block_weights.append(block_weight)
        
        # 블록 출력 결합
        combined_output = torch.cat(block_outputs, dim=-1)
        combined_output = self.output_combine(combined_output)
        
        # 최종 출력 변환
        output = self.output_transform(combined_output)
        output = self.dropout(output)
        output = self.layer_norm(output + hidden_states)
        
        # 메트릭스 업데이트
        self.metrics['routing_decisions'].append(routing_weights.detach().cpu())
        self.metrics['attention_weights'].extend([w.detach().cpu() for w in block_weights])
        self.metrics['total_calls'] += 1
        
        return output, {
            'routing_weights': routing_weights,
            'block_weights': block_weights,
            'memory_stats': self.matrix_optimizer.get_memory_stats()
        }
        
    def clear_cache(self):
        """캐시 초기화"""
        self.matrix_optimizer.clear_cache()
        self.router.clear_cache()
        for block in self.attention_blocks:
            block.clear_cache()

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """어텐션 마스크 전처리
        
        Args:
            attention_mask: 원본 어텐션 마스크
            batch_size: 배치 크기
            seq_length: 시퀀스 길이
            dtype: 텐서 데이터 타입
            
        Returns:
            전처리된 어텐션 마스크
        """
        # 마스크 차원 확장
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
            
        # 마스크 크기 조정
        attention_mask = attention_mask.expand(
            batch_size,
            1,
            seq_length,
            seq_length
        )
        
        # 데이터 타입 변환
        attention_mask = attention_mask.to(dtype=dtype)
        
        # 마스킹을 위한 값 변환 (0 -> -10000, 1 -> 0)
        return (1.0 - attention_mask) * -10000.0

class AdaptiveMultiHeadAttention(nn.Module):
    """적응형 멀티헤드 어텐션
    
    입력 데이터의 복잡도에 따라 동적으로 어텐션 헤드 수를 조절하는 어텐션 메커니즘
    """
    def __init__(
        self,
        config: Dict,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        max_cache_size: int = 1000
    ):
        """초기화 (최적화 버전)"""
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.max_num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.max_cache_size = max_cache_size  # 캐시 크기 설정
        
        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
        
        # 결정론적 가중치 초기화를 위해 시드 고정
        torch.manual_seed(42)
        
        # QKV 투영 (결정론적 초기화)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        nn.init.xavier_uniform_(self.qkv.weight)
        if bias:
            nn.init.zeros_(self.qkv.bias)
        
        # 출력 투영 (결정론적 초기화)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if bias:
            nn.init.zeros_(self.out_proj.bias)
        
        # 동적 헤드 선택을 위한 게이트 (결정론적 초기화)
        self.head_gates = nn.Parameter(torch.ones(num_heads))
        self.head_importance = nn.Parameter(torch.ones(num_heads))
        
        # 헤드 선택 임계값 (결정론적 초기화)
        self.head_threshold = nn.Parameter(torch.tensor(0.5))
        
        # 최적화된 메모리 및 캐시 관리자
        self.memory_pool = OptimizedMemoryPool(max_pool_size=100)
        self.cache_manager = FastCache(max_size=max_cache_size)
        
        # 복잡도 분석기 초기화
        self.complexity_analyzer = AttentionComplexityAnalyzer(config)
        
        # 배치 처리 최적화
        self.batch_stats = {}
        self.batch_cache = {}
        
        # 성능 모니터링
        self.performance_stats = {
            'processing_time': [],
            'memory_usage': [],
            'batch_sizes': [],
            'cache_hits': []
        }
        
        # 캐시 관련 속성 추가
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 시드 초기화 복원
        torch.manual_seed(torch.initial_seed())
        
    def _compute_qkv(
        self,
        x: torch.Tensor,
        active_heads: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """QKV 계산 (결정론적 버전)"""
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            
            # 1. 결정론적 QKV 투영
            qkv = self.qkv(x)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.max_num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            
            # 2. 결정론적 활성 헤드 선택
            active_heads = active_heads.view(1, -1, 1, 1)
            qkv = qkv * active_heads
            
            # 3. 결정론적 QKV 분리
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # 4. float32로 변환하여 일관성 보장
            q = q.float()
            k = k.float()
            v = v.float()
        
        return q, k, v
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """어텐션 계산 (결정론적 버전)"""
        with torch.no_grad():
            # 1. 결정론적 스케일링
            scale = math.sqrt(q.size(-1))
            
            # 2. 결정론적 어텐션 스코어 계산
            scores = torch.matmul(
                q.reshape(-1, seq_len, self.head_dim),
                k.reshape(-1, seq_len, self.head_dim).transpose(-2, -1)
            ).reshape(batch_size, self.max_num_heads, seq_len, seq_len)
            
            attention_weights = F.softmax(scores, dim=-1)
            
            # 3. 마스킹 적용
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # 4. 결정론적 소프트맥스 (고정된 온도 사용)
            attn = F.softmax(scores, dim=-1)
            
            # 5. 평가 모드에서는 드롭아웃 비활성화
            if self.training:
                attn = F.dropout(attn, p=self.dropout, training=True)
            
            # 6. 결정론적 출력 계산
            output = torch.matmul(attn, v)
            
            # 7. 결정론적 잔차 연결
            output = output + v.mean(dim=2, keepdim=True)
        
        return output, attn
    
    def _manage_cache(self) -> None:
        """캐시 크기 관리 (최적화 버전)"""
        if len(self.cache) > self.max_cache_size:
            # 1. 캐시 항목을 정렬 가능한 형태로 변환
            cache_items = list(self.cache.items())
            
            # 2. 캐시 키로 정렬하여 결정론적 동작 보장
            cache_items.sort(key=lambda x: x[0])
            
            # 3. 제거할 항목 수 계산
            num_to_remove = len(self.cache) - self.max_cache_size
            
            # 4. 가장 오래된 항목들 제거
            for i in range(num_to_remove):
                del self.cache[cache_items[i][0]]

    def _update_cache_stats(self, cache_hit: bool) -> None:
        """캐시 통계 업데이트 (최적화 버전)
        
        Args:
            cache_hit: 캐시 히트 여부
        """
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_cache_stats(self) -> Dict[str, float]:
        """캐시 성능 통계 반환 (최적화 버전)
        
        Returns:
            캐시 성능 지표를 포함하는 딕셔너리
        """
        total_accesses = max(1, self.cache_hits + self.cache_misses)
        hit_rate = self.cache_hits / total_accesses
        
        return {
            'cache_hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'total_accesses': total_accesses,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'efficiency_score': hit_rate * (1 - len(self.cache) / self.max_cache_size)
        }

    def _generate_cache_key(self, x: torch.Tensor) -> str:
        """결정론적 캐시 키 생성 (최적화 버전)
        
        Args:
            x: 입력 텐서
            
        Returns:
            캐시 키 문자열
        """
        with torch.no_grad():
            # 1. 배치 무관 통계 계산
            x_float = x.float()
            
            # 배치 차원 제외한 평균
            mean = x_float.mean(dim=0).mean().item()
            std = x_float.std(dim=0).mean().item()
            
            # 형상 정보 (배치 제외)
            shape = tuple(x.shape[1:])
            
            # 2. 결정론적 해시 입력 생성
            stats_str = f"{mean:.4f}_{std:.4f}_{shape}"
            
            # 3. SHA-256 해시 계산
            content_hash = hashlib.sha256(
                stats_str.encode()
            ).hexdigest()[:8]
            
            return f"s{shape}_h{content_hash}"

    def _select_active_heads(self, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """활성 헤드 선택 (최적화 버전)
        
        Args:
            attention_weights: 어텐션 가중치 (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            (활성 헤드 인덱스, 헤드 중요도 점수) 튜플
        """
        with torch.no_grad():
            # 1. 헤드 중요도 계산 (배치 평균)
            head_scores = attention_weights.float().mean(dim=[0, 2, 3])
            
            # 2. 헤드 중요도와 게이트 결합
            combined_scores = head_scores * self.head_gates
            
            # 3. 상위 k개 헤드 선택 (k는 복잡도에 따라 동적 결정)
            complexity_score = torch.sigmoid(combined_scores.mean())
            k = max(1, int(self.max_num_heads * complexity_score.item()))
            
            # 4. 결정론적 헤드 선택
            _, active_heads = torch.topk(combined_scores, k)
            
            # 5. 헤드 중요도 업데이트
            self.head_importance.data = 0.9 * self.head_importance + 0.1 * head_scores
            
            return active_heads, head_scores

    def _compute_entropy(self, x: torch.Tensor) -> float:
        """결정론적 엔트로피 계산"""
        with torch.no_grad():
            # 입력을 float32로 변환하여 일관성 보장
            x_float = x.detach().float()
            
            # 결정론적 정규화
            probs = F.softmax(x_float, dim=-1)
            
            # 결정론적 엔트로피 계산
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            
            return float(entropy.mean().item())

    def _compute_attention_sparsity(self, x: torch.Tensor) -> float:
        """결정론적 어텐션 희소성 계산"""
        with torch.no_grad():
            # 결정론적 어텐션 맵 계산
            attn_map = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.size(-1))
            attn_weights = F.softmax(attn_map, dim=-1)
            
            # 결정론적 희소성 계산
            threshold = 0.1
            sparsity = 1.0 - (attn_weights > threshold).float().mean()
            
            return float(sparsity.item())

    def _compute_feature_importance(self, x: torch.Tensor) -> float:
        """결정론적 특징 중요도 계산"""
        with torch.no_grad():
            # 배치 평균 계산
            if x.dim() == 3:
                x = x.mean(dim=0)
            
            # float32로 변환하여 일관성 보장
            x_float = x.float()
            
            # 결정론적 상관관계 계산
            mean = x_float.mean(dim=0, keepdim=True)
            centered = x_float - mean
            cov = torch.matmul(centered.t(), centered) / (x_float.size(0) - 1)
            std = torch.sqrt(torch.diag(cov) + 1e-8)
            corr = cov / torch.outer(std, std)
            
            # 결정론적 중요도 점수 계산
            importance = torch.abs(corr).mean()
            
            return float(importance.item())

    def _compute_sequence_complexity(self, x: torch.Tensor) -> float:
        """결정론적 시퀀스 복잡도 계산"""
        with torch.no_grad():
            # 배치 평균 계산
            if x.dim() == 3:
                x = x.mean(dim=0)
            
            # float32로 변환하여 일관성 보장
            x_float = x.float()
            
            # 결정론적 자기상관 계산
            mean = x_float.mean(dim=0, keepdim=True)
            centered = x_float - mean
            auto_corr = torch.matmul(centered, centered.t()) / (torch.norm(centered, dim=1, keepdim=True) ** 2)
            
            # 결정론적 복잡도 점수 계산
            complexity = 1.0 - torch.abs(auto_corr).mean()
            
            return float(complexity.item())

    def _estimate_optimal_heads(self, complexity_metrics: Dict[str, float]) -> int:
        """결정론적 최적 헤드 수 추정"""
        # 결정론적 가중치 적용
        complexity_score = (
            0.4 * complexity_metrics['entropy'] +
            0.3 * complexity_metrics['attention_sparsity'] +
            0.2 * complexity_metrics['feature_importance'] +
            0.1 * complexity_metrics['sequence_complexity']
        )
        
        # 결정론적 헤드 수 결정
        base_heads = self.max_num_heads
        if complexity_score < 0.3:
            return max(1, base_heads // 4)
        elif complexity_score < 0.6:
            return max(2, base_heads // 2)
        else:
            return base_heads

    def _process_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """최적화된 배치 처리"""
        batch_size = x.shape[0]
        
        # 1. 배치 크기별 최적 처리 전략 결정
        if batch_size <= 4:
            return self._process_small_batch(x)
        else:
            return self._process_large_batch(x)

    def _process_small_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """작은 배치 최적화 처리 (4 이하)"""
        start_time = time.time()
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 메모리 할당
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.max_num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # 2. QKV 분리 및 스케일링
        q = qkv[0] * self.scaling
        k = qkv[1]
        v = qkv[2]
        
        # 3. 어텐션 계산 (최적화)
        attention_weights = torch.matmul(
            q.reshape(-1, seq_len, self.head_dim),
            k.reshape(-1, seq_len, self.head_dim).transpose(-2, -1)
        )
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        if self.training and self.dropout > 0:
            attention_weights = F.dropout(attention_weights, p=self.dropout, training=True)
        
        # 4. 활성 헤드 선택
        active_heads, head_scores = self._select_active_heads(attention_weights)
        active_head_mask = torch.zeros(
            self.max_num_heads,
            device=x.device,
            dtype=torch.float32
        )
        active_head_mask[active_heads] = 1.0
        active_head_mask = active_head_mask.reshape(1, -1, 1, 1)
        
        # 5. 출력 계산 (최적화)
        attention_output = torch.matmul(
            attention_weights.reshape(-1, seq_len, seq_len),
            v.reshape(-1, seq_len, self.head_dim)
        ).reshape(batch_size, self.max_num_heads, seq_len, self.head_dim)
        
        attention_output = attention_output * active_head_mask
        attention_output = attention_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        output = self.out_proj(attention_output)
        
        # 6. 메타데이터 수집
        end_time = time.time()
        metadata = {
            'active_heads': active_heads.tolist(),
            'head_scores': head_scores.tolist(),
            'num_active_heads': len(active_heads),
            'processing_time': end_time - start_time,
            'cache_stats': self.cache_manager.get_stats(batch_size)
        }
        
        return output, metadata

    def _process_large_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """큰 배치 최적화 처리 (4 초과)"""
        start_time = time.time()
        batch_size, seq_len, hidden_dim = x.shape
        
        # 1. 동적 청크 크기 계산
        chunk_size = self._compute_optimal_chunk_size(batch_size)
        num_chunks = (batch_size + chunk_size - 1) // chunk_size
        
        # 2. 결과 저장용 텐서 할당
        output = torch.empty(batch_size, seq_len, hidden_dim, device=x.device)
        
        # 3. 청크 단위 처리
        active_heads_list = []
        head_scores_list = []
        
        for i in range(num_chunks):
            chunk_start = i * chunk_size
            chunk_end = min((i + 1) * chunk_size, batch_size)
            x_chunk = x[chunk_start:chunk_end]
            chunk_size_actual = chunk_end - chunk_start
            
            # 3.1 QKV 계산
            qkv = self.qkv(x_chunk)
            qkv = qkv.reshape(chunk_size_actual, seq_len, 3, self.max_num_heads, -1)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            
            # 3.2 QKV 분리 및 스케일링
            q = qkv[0] * self.scaling
            k = qkv[1]
            v = qkv[2]
            
            # 3.3 어텐션 계산 (최적화)
            attention_weights = torch.matmul(
                q.reshape(-1, seq_len, self.head_dim),
                k.reshape(-1, seq_len, self.head_dim).transpose(-2, -1)
            ).reshape(chunk_size_actual, self.max_num_heads, seq_len, seq_len)
            
            attention_weights = F.softmax(attention_weights, dim=-1)
            
            if self.training and self.dropout > 0:
                attention_weights = F.dropout(attention_weights, p=self.dropout, training=True)
            
            # 3.4 활성 헤드 선택
            active_heads, head_scores = self._select_active_heads(attention_weights)
            active_heads_list.append(active_heads)
            head_scores_list.append(head_scores)
            
            active_head_mask = torch.zeros(
                self.max_num_heads,
                device=x.device,
                dtype=torch.float32
            )
            active_head_mask[active_heads] = 1.0
            active_head_mask = active_head_mask.reshape(1, -1, 1, 1)
            
            # 3.5 출력 계산 (최적화)
            attention_output = torch.matmul(
                attention_weights.reshape(-1, seq_len, seq_len),
                v.reshape(-1, seq_len, self.head_dim)
            ).reshape(chunk_size_actual, self.max_num_heads, seq_len, self.head_dim)
            
            attention_output = attention_output * active_head_mask
            attention_output = attention_output.transpose(1, 2).reshape(chunk_size_actual, seq_len, hidden_dim)
            chunk_output = self.out_proj(attention_output)
            
            # 3.6 결과 저장
            output[chunk_start:chunk_end] = chunk_output
        
        # 4. 메타데이터 수집
        end_time = time.time()
        
        # 활성 헤드 통합 (다수결)
        all_active_heads = torch.cat(active_heads_list)
        head_counts = torch.bincount(all_active_heads, minlength=self.max_num_heads)
        final_active_heads = torch.where(head_counts > num_chunks // 2)[0]
        
        # 헤드 점수 평균
        all_head_scores = torch.stack(head_scores_list)
        mean_head_scores = all_head_scores.mean(dim=0)
        
        metadata = {
            'active_heads': final_active_heads.tolist(),
            'head_scores': mean_head_scores.tolist(),
            'num_active_heads': len(final_active_heads),
            'processing_time': end_time - start_time,
            'cache_stats': self.cache_manager.get_stats(batch_size),
            'num_chunks': num_chunks,
            'chunk_size': chunk_size
        }
        
        return output, metadata

    def _compute_optimal_chunk_size(self, batch_size: int) -> int:
        """배치 크기에 따른 최적 청크 크기 계산"""
        if batch_size <= 8:
            return 4
        elif batch_size <= 16:
            return 8
        else:
            return min(16, batch_size // 2)

    def update_head_importance(self, gradient_norm: torch.Tensor) -> None:
        """헤드 중요도 업데이트 (안정성 개선)
        
        Args:
            gradient_norm: 각 헤드의 그래디언트 노름
        """
        with torch.no_grad():
            # 1. 그래디언트 클리핑
            max_norm = 1.0
            gradient_norm = torch.clamp(gradient_norm, -max_norm, max_norm)
            
            # 2. 이동 평균으로 중요도 업데이트 (안정성 개선)
            momentum = 0.9
            self.head_importance.data = (
                momentum * self.head_importance +
                (1 - momentum) * gradient_norm
            )
            
            # 3. 중요도 정규화
            self.head_importance.data = F.softmax(self.head_importance, dim=0)
            
            # 4. 게이트 업데이트
            gate_threshold = 0.1
            self.head_gates.data = (self.head_importance > gate_threshold).float()
    
    def prune_heads(self, head_mask: torch.Tensor) -> None:
        """헤드 가지치기
        
        Args:
            head_mask: 유지할 헤드를 나타내는 이진 마스크
        """
        with torch.no_grad():
            self.head_gates.data *= head_mask
            
    def reset_cache(self) -> None:
        """캐시 초기화"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_manager.clear()
        self.memory_pool.cleanup_old_tensors()
        torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """순전파
        
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, seq_len, embed_dim]
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - 출력 텐서 [batch_size, seq_len, embed_dim]
                - 메타데이터 (헤드 선택, 복잡도 메트릭 등)
        """
        batch_size = x.size(0)
        start_time = time.time()
        
        # 캐시 키 생성 및 조회
        cache_key = self._generate_cache_key(x)
        cached_result = self.cache_manager.get(cache_key, batch_size)
        
        if cached_result is not None:
            self._update_cache_stats(cache_hit=True)
            output, metadata = cached_result
            metadata['cache_hit'] = True
            return output, metadata
            
        self._update_cache_stats(cache_hit=False)
        
        # 배치 크기에 따른 처리 방법 선택
        if batch_size <= 8:
            output, metadata = self._process_small_batch(x)
        else:
            output, metadata = self._process_large_batch(x)
        
        # 캐시 저장
        self.cache_manager.put(cache_key, (output, metadata), batch_size)
        
        # 성능 통계 업데이트
        processing_time = time.time() - start_time
        self.performance_stats['processing_time'].append(processing_time)
        self.performance_stats['batch_sizes'].append(batch_size)
        
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.performance_stats['memory_usage'].append(memory_usage)
        
        metadata.update({
            'processing_time': processing_time,
            'cache_hit': False
        })
        
        return output, metadata

    def _process_small_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """작은 배치 처리 (최적화된 버전)"""
        batch_size, seq_len, _ = x.shape
        
        # QKV 계산
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.max_num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 스케일링된 닷-프로덕트 어텐션
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 드롭아웃
        if self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # 헤드 선택
        active_heads, head_weights = self._select_active_heads(attn_weights)
        
        # 선택된 헤드만 사용하여 출력 계산
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output * active_heads.view(1, -1, 1, 1)
        
        # 차원 변환 및 출력 투영
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        
        # 복잡도 메트릭 계산
        complexity_metrics = {
            'entropy': self._compute_entropy(x),
            'attention_sparsity': self._compute_attention_sparsity(attn_weights),
            'feature_importance': self._compute_feature_importance(x),
            'sequence_complexity': self._compute_sequence_complexity(x)
        }
        
        metadata = {
            'active_heads': active_heads,
            'head_weights': head_weights,
            'complexity_metrics': complexity_metrics,
            'attention_weights': attn_weights.detach()
        }
        
        return output, metadata

    def _process_large_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """큰 배치 처리 (청크 기반 최적화)"""
        batch_size = x.size(0)
        chunk_size = self._compute_optimal_chunk_size(batch_size)
        num_chunks = math.ceil(batch_size / chunk_size)
        
        outputs = []
        all_metadata = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            chunk = x[start_idx:end_idx]
            
            # 청크 처리
            chunk_output, chunk_metadata = self._process_small_batch(chunk)
            outputs.append(chunk_output)
            all_metadata.append(chunk_metadata)
        
        # 결과 병합
        output = torch.cat(outputs, dim=0)
        
        # 메타데이터 통합
        merged_metadata = {
            'active_heads': torch.stack([m['active_heads'] for m in all_metadata]).mean(0),
            'head_weights': torch.stack([m['head_weights'] for m in all_metadata]).mean(0),
            'complexity_metrics': {
                k: sum(m['complexity_metrics'][k] for m in all_metadata) / len(all_metadata)
                for k in all_metadata[0]['complexity_metrics']
            },
            'chunked_processing': True,
            'num_chunks': num_chunks
        }
        
        return output, merged_metadata

    def _compute_optimal_chunk_size(self, batch_size: int) -> int:
        """최적의 청크 크기 계산"""
        if batch_size <= 8:
            return batch_size
            
        # GPU 메모리에 따른 동적 조정
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = free_memory / (1024 * 1024 * 1024)  # GB
            
            if free_memory < 4:
                return 4
            elif free_memory < 8:
                return 8
            elif free_memory < 16:
                return 16
            else:
                return 32
        
        # CPU의 경우 보수적으로 설정
        return 8

    def _compute_entropy(self, x: torch.Tensor) -> float:
        """입력의 엔트로피 계산"""
        # 정규화된 확률 분포 계산
        x_flat = x.view(-1, x.size(-1))
        probs = F.softmax(x_flat, dim=-1)
        
        # 엔트로피 계산
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy.mean().item()

    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> float:
        """어텐션 맵의 희소성 계산"""
        # 임계값 기반 희소성
        threshold = 0.1
        sparsity = (attention_weights < threshold).float().mean().item()
        return sparsity

    def _compute_feature_importance(self, x: torch.Tensor) -> float:
        """특징 중요도 계산"""
        with torch.no_grad():
            # 특징 간 상관관계 계산
            x_centered = x - x.mean(dim=1, keepdim=True)
            cov = torch.bmm(x_centered.transpose(1, 2), x_centered)
            cov = cov / (x.size(1) - 1)
            
            # 특징 중요도 = 대각 요소의 상대적 크기
            importance = torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=0)
            importance = F.softmax(importance, dim=-1)
            
            return importance.max().item()

    def _compute_sequence_complexity(self, x: torch.Tensor) -> float:
        """시퀀스 복잡도 계산"""
        # 시퀀스 내 변화율 계산
        diffs = torch.abs(x[:, 1:] - x[:, :-1])
        complexity = diffs.mean().item()
        return complexity

    def _generate_cache_key(self, x: torch.Tensor) -> str:
        """결정론적 캐시 키 생성 (최적화 버전)
        
        Args:
            x: 입력 텐서
            
        Returns:
            캐시 키 문자열
        """
        with torch.no_grad():
            # 1. 배치 무관 통계 계산
            x_float = x.float()
            
            # 배치 차원 제외한 평균
            mean = x_float.mean(dim=0).mean().item()
            std = x_float.std(dim=0).mean().item()
            
            # 형상 정보 (배치 제외)
            shape = tuple(x.shape[1:])
            
            # 2. 결정론적 해시 입력 생성
            stats_str = f"{mean:.4f}_{std:.4f}_{shape}"
            
            # 3. SHA-256 해시 계산
            content_hash = hashlib.sha256(
                stats_str.encode()
            ).hexdigest()[:8]
            
            return f"s{shape}_h{content_hash}"

    def _select_active_heads(self, attention_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """활성 헤드 선택 (최적화 버전)
        
        Args:
            attention_weights: 어텐션 가중치 (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            (활성 헤드 인덱스, 헤드 중요도 점수) 튜플
        """
        with torch.no_grad():
            # 1. 헤드 중요도 계산 (배치 평균)
            head_scores = attention_weights.float().mean(dim=[0, 2, 3])
            
            # 2. 헤드 중요도와 게이트 결합
            combined_scores = head_scores * self.head_gates
            
            # 3. 상위 k개 헤드 선택 (k는 복잡도에 따라 동적 결정)
            complexity_score = torch.sigmoid(combined_scores.mean())
            k = max(1, int(self.max_num_heads * complexity_score.item()))
            
            # 4. 결정론적 헤드 선택
            _, active_heads = torch.topk(combined_scores, k)
            
            # 5. 헤드 중요도 업데이트
            self.head_importance.data = 0.9 * self.head_importance + 0.1 * head_scores
            
            return active_heads, head_scores

    def _update_cache_stats(self, cache_hit: bool) -> None:
        """캐시 통계 업데이트 (최적화 버전)
        
        Args:
            cache_hit: 캐시 히트 여부
        """
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def _process_large_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """큰 배치 처리 (청크 기반 최적화)"""
        batch_size = x.size(0)
        chunk_size = self._compute_optimal_chunk_size(batch_size)
        num_chunks = math.ceil(batch_size / chunk_size)
        
        outputs = []
        all_metadata = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, batch_size)
            chunk = x[start_idx:end_idx]
            
            # 청크 처리
            chunk_output, chunk_metadata = self._process_small_batch(chunk)
            outputs.append(chunk_output)
            all_metadata.append(chunk_metadata)
        
        # 결과 병합
        output = torch.cat(outputs, dim=0)
        
        # 메타데이터 통합
        merged_metadata = {
            'active_heads': torch.stack([m['active_heads'] for m in all_metadata]).mean(0),
            'head_weights': torch.stack([m['head_weights'] for m in all_metadata]).mean(0),
            'complexity_metrics': {
                k: sum(m['complexity_metrics'][k] for m in all_metadata) / len(all_metadata)
                for k in all_metadata[0]['complexity_metrics']
            },
            'chunked_processing': True,
            'num_chunks': num_chunks
        }
        
        return output, merged_metadata 