"""
기본 트랜스포머 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Union, Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['attention_heads']
        self.attention_head_size = int(self.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # QKV 변환
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # 출력 변환
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 드롭아웃
        self.dropout = nn.Dropout(config['attention_probs_dropout_prob'])
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = hidden_states.size()[:2]
        
        # QKV 변환
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 어텐션 스코어 계산
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=torch.float32))
        
        if attention_mask is not None:
            # 어텐션 마스크 차원 확장
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(-1, self.num_attention_heads, seq_length, -1)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # 어텐션 확률 계산
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 컨텍스트 계산
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 출력 변환
        attention_output = self.output(context_layer)
        
        return attention_output, attention_probs

class Transformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.intermediate = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.output = nn.Linear(config['intermediate_size'], config['hidden_size'])
        
        # 레이어 정규화
        self.attention_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        self.output_layer_norm = nn.LayerNorm(config['hidden_size'], eps=config['layer_norm_eps'])
        
        # 드롭아웃
        self.dropout = nn.Dropout(config['hidden_dropout_prob'])
        
        # 활성화 함수
        self.activation = F.gelu
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        # 셀프 어텐션
        attention_output, attention_probs = self.attention(
            hidden_states,
            attention_mask=attention_mask
        )
        
        # 첫 번째 잔차 연결 및 레이어 정규화
        attention_output = self.dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        # 피드포워드 네트워크
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        
        # 출력 변환
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        # 두 번째 잔차 연결 및 레이어 정규화
        layer_output = self.output_layer_norm(layer_output + attention_output)
        
        outputs = {
            'last_hidden_state': layer_output,
            'attention_probs': attention_probs
        }
        
        return outputs 