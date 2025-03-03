"""
행렬 연산 최적화 테스트
"""

import pytest
import torch
import numpy as np
import time
from ..blocks.optimized_matrix import TriangularMatrixOptimizer

@pytest.fixture
def config():
    return {
        'hidden_size': 512,
        'attention_heads': 8,
        'intermediate_size': 2048,
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'layer_norm_eps': 1e-12
    }

@pytest.fixture
def matrix_optimizer(config):
    return TriangularMatrixOptimizer(config)

def test_triangular_matrix_multiplication(matrix_optimizer):
    """삼각 행렬 곱셈 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 테스트 데이터 생성
    matrix_a = torch.randn(batch_size, seq_len, hidden_size)
    matrix_b = torch.randn(batch_size, hidden_size, seq_len)
    
    # 삼각 행렬 곱셈
    start_time = time.time()
    result = matrix_optimizer._triangular_matrix_multiply(matrix_a, matrix_b)
    tri_time = time.time() - start_time
    
    # 기준 구현
    start_time = time.time()
    baseline = torch.matmul(matrix_a, matrix_b)
    mask = torch.tril(torch.ones(seq_len, seq_len))
    baseline = baseline * mask
    baseline_time = time.time() - start_time
    
    # 정확도 검증
    assert torch.allclose(result, baseline, rtol=1e-4), "삼각 행렬 곱셈 결과가 정확하지 않습니다"
    
    # 성능 검증
    print(f"\n성능 비교:")
    print(f"최적화 버전: {tri_time:.4f}초")
    print(f"기준 구현: {baseline_time:.4f}초")
    print(f"속도 향상: {baseline_time/tri_time:.2f}배")

def test_matrix_multiplication(matrix_optimizer):
    """일반 행렬 곱셈 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 테스트 데이터 생성
    matrix_a = torch.randn(batch_size, seq_len, hidden_size)
    matrix_b = torch.randn(batch_size, hidden_size, seq_len)
    
    # 최적화된 행렬 곱셈
    start_time = time.time()
    result = matrix_optimizer.optimize_matrix_multiplication(matrix_a, matrix_b)
    opt_time = time.time() - start_time
    
    # 기준 구현
    start_time = time.time()
    baseline = torch.bmm(matrix_a, matrix_b)
    baseline_time = time.time() - start_time
    
    # 정확도 검증
    assert torch.allclose(result, baseline, rtol=1e-4), "행렬 곱셈 결과가 정확하지 않습니다"
    
    # 성능 검증
    print(f"\n성능 비교:")
    print(f"최적화 버전: {opt_time:.4f}초")
    print(f"기준 구현: {baseline_time:.4f}초")
    print(f"속도 향상: {baseline_time/opt_time:.2f}배")

def test_attention_score_computation(matrix_optimizer):
    """어텐션 스코어 계산 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 테스트 데이터 생성
    query = torch.randn(batch_size, seq_len, hidden_size)
    key = torch.randn(batch_size, seq_len, hidden_size)
    value = torch.randn(batch_size, seq_len, hidden_size)
    mask = torch.zeros(batch_size, 1, 1, seq_len)
    
    # 최적화된 어텐션 계산
    start_time = time.time()
    context, probs = matrix_optimizer.compute_attention_scores(query, key, value, mask)
    opt_time = time.time() - start_time
    
    # 기준 구현
    start_time = time.time()
    scaling = 1.0 / np.sqrt(hidden_size)
    scores = torch.matmul(query * scaling, key.transpose(-2, -1))
    scores = scores + mask
    probs_baseline = torch.softmax(scores, dim=-1)
    context_baseline = torch.matmul(probs_baseline, value)
    baseline_time = time.time() - start_time
    
    # 정확도 검증
    assert torch.allclose(context, context_baseline, rtol=1e-4), "어텐션 계산 결과가 정확하지 않습니다"
    assert torch.allclose(probs, probs_baseline, rtol=1e-4), "어텐션 확률이 정확하지 않습니다"
    
    # 성능 검증
    print(f"\n성능 비교:")
    print(f"최적화 버전: {opt_time:.4f}초")
    print(f"기준 구현: {baseline_time:.4f}초")
    print(f"속도 향상: {baseline_time/opt_time:.2f}배")

def test_memory_efficiency(matrix_optimizer):
    """메모리 효율성 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 초기 메모리 사용량
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # 테스트 데이터 생성
    matrix_a = torch.randn(batch_size, seq_len, hidden_size)
    matrix_b = torch.randn(batch_size, hidden_size, seq_len)
    
    # 최적화된 버전 메모리 사용량
    matrix_optimizer.optimize_matrix_multiplication(matrix_a, matrix_b, triangular=True)
    opt_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # 기준 구현 메모리 사용량
    torch.matmul(matrix_a, matrix_b)
    baseline_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    print(f"\n메모리 사용량 비교:")
    print(f"최적화된 버전: {(opt_memory-initial_memory)/1024/1024:.2f}MB")
    print(f"기준 구현: {(baseline_memory-initial_memory)/1024/1024:.2f}MB")
    print(f"메모리 절감: {(baseline_memory-opt_memory)/1024/1024:.2f}MB")

def test_cache_effectiveness(matrix_optimizer):
    """캐시 효과 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 테스트 데이터 생성
    matrix_a = torch.randn(batch_size, seq_len, hidden_size)
    matrix_b = torch.randn(batch_size, hidden_size, seq_len)
    
    # 캐시 초기화
    matrix_optimizer.clear_cache()
    
    # 첫 번째 실행 (캐시 미스)
    start_time = time.time()
    result1 = matrix_optimizer._triangular_matrix_multiply(matrix_a, matrix_b)
    first_time = time.time() - start_time
    
    # 두 번째 실행 (캐시 히트)
    start_time = time.time()
    result2 = matrix_optimizer._triangular_matrix_multiply(matrix_a, matrix_b)
    second_time = time.time() - start_time
    
    # 결과 일관성 검증
    assert torch.allclose(result1, result2, rtol=1e-4), "캐시된 결과가 일관되지 않습니다"
    
    # 캐시 효과 검증
    print(f"\n캐시 성능:")
    print(f"첫 번째 실행: {first_time:.4f}초")
    print(f"두 번째 실행: {second_time:.4f}초")
    print(f"속도 향상: {first_time/second_time:.2f}배")
    
    # 메모리 사용량 확인
    stats = matrix_optimizer.get_memory_stats()
    print(f"\n메모리 통계:")
    print(f"캐시 크기: {stats['cache_size']}")
    print(f"마스크 캐시 크기: {stats['mask_cache_size']}")
    print(f"총 메모리: {stats['total_memory']/1024/1024:.2f}MB")

def test_memory_management(matrix_optimizer):
    """메모리 관리 테스트"""
    batch_size, seq_len, hidden_size = 32, 128, 512
    
    # 초기 메모리 상태
    initial_stats = matrix_optimizer.get_memory_stats()
    
    # 여러 크기의 행렬로 테스트
    for size in [64, 128, 256]:
        matrix_a = torch.randn(batch_size, size, hidden_size)
        matrix_b = torch.randn(batch_size, hidden_size, size)
        
        _ = matrix_optimizer._triangular_matrix_multiply(matrix_a, matrix_b)
    
    # 최종 메모리 상태
    final_stats = matrix_optimizer.get_memory_stats()
    
    # 메모리 사용량 검증
    print(f"\n메모리 관리:")
    print(f"초기 캐시 크기: {initial_stats['cache_size']}")
    print(f"최종 캐시 크기: {final_stats['cache_size']}")
    print(f"초기 마스크 캐시 크기: {initial_stats['mask_cache_size']}")
    print(f"최종 마스크 캐시 크기: {final_stats['mask_cache_size']}")
    
    # 캐시 정리
    matrix_optimizer.clear_cache()
    cleared_stats = matrix_optimizer.get_memory_stats()
    assert cleared_stats['cache_size'] == 0, "캐시가 완전히 정리되지 않았습니다"
    assert cleared_stats['mask_cache_size'] == 0, "마스크 캐시가 완전히 정리되지 않았습니다"

if __name__ == "__main__":
    pytest.main([__file__]) 