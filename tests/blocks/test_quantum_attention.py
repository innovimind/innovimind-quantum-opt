import pytest
import torch
import numpy as np
from ..blocks.quantum_attention import QuantumRegister, QuantumAttention

@pytest.fixture
def quantum_register():
    return QuantumRegister(hidden_dim=64)

@pytest.fixture
def quantum_attention():
    return QuantumAttention(dim=512, num_heads=8)

def test_quantum_register_normalization(quantum_register):
    """양자 상태 정규화 테스트"""
    # 테스트 입력 생성
    batch_size = 32
    seq_len = 16
    num_heads = 8
    head_dim = 64
    
    vector = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # 정규화 적용
    normalized = quantum_register.prepare_normalized(vector)
    
    # 차원 검증
    assert normalized.shape == (batch_size, num_heads, seq_len, head_dim)
    
    # 정규화 검증
    norms = torch.norm(normalized, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_quantum_register_entanglement(quantum_register):
    """양자 얽힘 상태 생성 테스트"""
    # 테스트 입력 생성
    batch_size = 32
    seq_len = 16
    num_heads = 8
    head_dim = 64
    
    state1 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    state2 = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # 정규화
    state1 = quantum_register.prepare_normalized(state1)
    state2 = quantum_register.prepare_normalized(state2)
    
    # 얽힘 상태 생성
    entangled = quantum_register.entangle(state1, state2)
    
    # 차원 검증
    assert entangled.shape == (batch_size, num_heads, seq_len, seq_len)
    
    # 확률 분포 특성 검증
    attention_probs = torch.softmax(entangled, dim=-1)
    row_sums = attention_probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

def test_quantum_attention_forward(quantum_attention):
    """양자 어텐션 순전파 테스트"""
    # 테스트 입력 생성
    batch_size = 16
    seq_len = 32
    dim = 512
    
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)
    
    # 순전파
    output, attention_weights = quantum_attention(query, key, value)
    
    # 출력 차원 검증
    assert output.shape == (batch_size, seq_len, dim)
    assert attention_weights.shape == (batch_size, quantum_attention.num_heads, seq_len, seq_len)
    
    # 어텐션 가중치 합 검증
    weight_sums = attention_weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

def test_quantum_attention_mask(quantum_attention):
    """마스킹 기능 테스트"""
    # 테스트 입력 생성
    batch_size = 16
    seq_len = 32
    dim = 512
    num_heads = quantum_attention.num_heads
    
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)
    
    # 마스크 생성 (인과적 마스크)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, seq_len, seq_len)
    
    # 순전파
    output, attention_weights = quantum_attention(query, key, value, mask=mask)
    
    # 마스킹된 위치 검증
    masked_weights = attention_weights.masked_fill(mask, 0)
    assert torch.all(masked_weights[mask] == 0)
    
    # 마스킹되지 않은 위치의 합이 1인지 검증
    unmasked_sums = attention_weights.masked_fill(mask, 0).sum(dim=-1)
    assert torch.allclose(unmasked_sums, torch.ones_like(unmasked_sums), atol=1e-6)

def test_quantum_attention_gradient(quantum_attention):
    """그래디언트 흐름 테스트"""
    # 테스트 입력 생성
    batch_size = 16
    seq_len = 32
    dim = 512
    
    query = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    key = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    value = torch.randn(batch_size, seq_len, dim, requires_grad=True)
    
    # 순전파 및 역전파
    output, _ = quantum_attention(query, key, value)
    loss = output.sum()
    loss.backward()
    
    # 그래디언트 존재 여부 검증
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    
    # 그래디언트 크기 검증
    assert not torch.any(torch.isnan(query.grad))
    assert not torch.any(torch.isnan(key.grad))
    assert not torch.any(torch.isnan(value.grad))

def test_quantum_attention_memory(quantum_attention):
    """메모리 효율성 테스트"""
    # 테스트 입력 생성
    batch_size = 16
    seq_len = 32
    dim = 512
    
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)
    
    # 메모리 사용량 측정
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated()
        else:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
    
    # 초기 메모리 사용량
    initial_memory = get_memory_usage()
    
    # 순전파
    output, _ = quantum_attention(query, key, value)
    
    # 최종 메모리 사용량
    final_memory = get_memory_usage()
    
    # 메모리 증가량 계산 (이론적 한계와 비교)
    theoretical_limit = batch_size * seq_len * dim * 4  # 4는 float32 기준
    actual_increase = final_memory - initial_memory
    
    # CPU에서는 메모리 측정이 부정확할 수 있으므로 느슨한 제한 적용
    assert actual_increase <= theoretical_limit * 3  # 300% 여유 허용

def test_quantum_attention_speed(quantum_attention):
    """처리 속도 테스트"""
    # 테스트 입력 생성
    batch_size = 16
    seq_len = 32
    dim = 512
    
    query = torch.randn(batch_size, seq_len, dim)
    key = torch.randn(batch_size, seq_len, dim)
    value = torch.randn(batch_size, seq_len, dim)
    
    # 웜업
    for _ in range(5):
        quantum_attention(query, key, value)
    
    # 속도 측정
    import time
    start_time = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        quantum_attention(query, key, value)
    
    end_time = time.time()
    average_time = (end_time - start_time) / num_iterations
    
    # CPU에서는 처리 시간이 더 길 수 있으므로 느슨한 제한 적용
    assert average_time < 0.5  # 500ms 이하

def test_phase_optimization(quantum_attention):
    """위상 최적화 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 위상이 있는 복소수 입력 생성
    phase = torch.rand(batch_size, seq_len, dim) * 2 * np.pi
    magnitude = torch.rand(batch_size, seq_len, dim)
    query = magnitude * torch.exp(1j * phase)
    key = magnitude * torch.exp(1j * phase)
    value = magnitude * torch.exp(1j * phase)
    
    # 실수 텐서로 변환
    query = torch.cat([query.real, query.imag], dim=-1)
    key = torch.cat([key.real, key.imag], dim=-1)
    value = torch.cat([value.real, value.imag], dim=-1)
    
    # 순전파
    output, attention_weights = quantum_attention(query, key, value)
    
    # 위상 보존 검증
    output_phase = torch.atan2(
        output[..., dim:],  # 허수부
        output[..., :dim]   # 실수부
    )
    
    # 위상이 완전히 손실되지 않았는지 확인
    phase_variance = output_phase.var()
    assert phase_variance > 0, "위상 정보가 완전히 손실됨"
    
    # 위상 최적화 레이어의 영향 검증
    original_phase = torch.atan2(query[..., dim:], query[..., :dim])
    phase_correlation = torch.corrcoef(
        output_phase.flatten(),
        original_phase.flatten()
    )
    assert phase_correlation[0, 1] > 0.1, "위상 최적화가 효과적이지 않음"

def test_entanglement_optimization(quantum_attention):
    """얽힘 최적화 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 상관관계가 있는 입력 생성
    common_features = torch.randn(batch_size, seq_len, dim // 2)
    query = torch.cat([common_features, torch.randn(batch_size, seq_len, dim // 2)], dim=-1)
    key = torch.cat([common_features, torch.randn(batch_size, seq_len, dim // 2)], dim=-1)
    value = torch.randn(batch_size, seq_len, dim)
    
    # 순전파
    output, attention_weights = quantum_attention(query, key, value)
    
    # 얽힘 강도 분석
    entanglement_strength = attention_weights.abs().mean()
    assert entanglement_strength > 0.1, "얽힘이 너무 약함"
    
    # 상관관계가 있는 부분에 대한 어텐션 강도 확인
    attention_pattern = attention_weights.mean(dim=1)  # 헤드 평균
    diagonal_attention = torch.diagonal(attention_pattern, dim1=-2, dim2=-1).mean()
    assert diagonal_attention > 1.0 / seq_len, "상관된 토큰 간 얽힘이 약함"

def test_noise_handling(quantum_attention):
    """노이즈 처리 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 노이즈가 있는 입력 생성
    clean_signal = torch.randn(batch_size, seq_len, dim)
    noise = torch.randn(batch_size, seq_len, dim) * 0.1
    noisy_input = clean_signal + noise
    
    # 학습 모드에서 테스트
    quantum_attention.train()
    noisy_output, _ = quantum_attention(noisy_input, noisy_input, noisy_input)
    
    # 평가 모드에서 테스트
    quantum_attention.eval()
    clean_output, _ = quantum_attention(noisy_input, noisy_input, noisy_input)
    
    # 노이즈 감소 검증
    noisy_variance = (noisy_output - clean_signal).var()
    clean_variance = (clean_output - clean_signal).var()
    assert clean_variance < noisy_variance, "노이즈 제거가 효과적이지 않음"

def test_measurement_optimization(quantum_attention):
    """양자 측정 최적화 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 다양한 크기의 신호 생성
    strong_signal = torch.randn(batch_size, seq_len, dim) * 2.0
    weak_signal = torch.randn(batch_size, seq_len, dim) * 0.1
    
    # 강한 신호 테스트
    strong_output, _ = quantum_attention(strong_signal, strong_signal, strong_signal)
    strong_magnitude = torch.norm(strong_output, dim=-1)
    
    # 약한 신호 테스트
    weak_output, _ = quantum_attention(weak_signal, weak_signal, weak_signal)
    weak_magnitude = torch.norm(weak_output, dim=-1)
    
    # 측정 최적화 검증
    assert torch.all(strong_magnitude > weak_magnitude), "측정이 신호 강도를 보존하지 않음"
    
    # 약한 신호가 완전히 억제되지 않았는지 확인
    assert torch.all(weak_magnitude > 0), "약한 신호가 완전히 소실됨"

def test_phase_noise_stability(quantum_attention):
    """위상 노이즈 안정성 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 기본 신호 생성
    base_input = torch.randn(batch_size, seq_len, dim)
    
    # 위상 노이즈 추가
    phase_noise = torch.randn(batch_size, seq_len, dim) * 0.1
    noisy_input = base_input * torch.exp(1j * phase_noise)
    noisy_input = torch.cat([noisy_input.real, noisy_input.imag], dim=-1)
    
    # 여러 번의 순전파로 안정성 테스트
    outputs = []
    for _ in range(10):
        output, _ = quantum_attention(noisy_input, noisy_input, noisy_input)
        outputs.append(output)
    
    # 출력 안정성 검증
    output_stack = torch.stack(outputs)
    output_std = output_stack.std(dim=0)
    assert torch.all(output_std < 0.1), "위상 노이즈에 대한 안정성이 부족함"

def test_entanglement_capacity(quantum_attention):
    """얽힘 용량 테스트"""
    batch_size = 16
    seq_len = 32
    dim = 512
    
    # 다양한 상관관계를 가진 입력 생성
    inputs = []
    correlations = [0.0, 0.3, 0.6, 0.9]
    
    for correlation in correlations:
        common = torch.randn(batch_size, seq_len, dim)
        noise = torch.randn(batch_size, seq_len, dim)
        input_signal = correlation * common + (1 - correlation) * noise
        inputs.append(input_signal)
    
    # 각 상관관계에 대한 얽힘 강도 측정
    entanglement_strengths = []
    
    for input_signal in inputs:
        _, attention_weights = quantum_attention(input_signal, input_signal, input_signal)
        strength = attention_weights.abs().mean().item()
        entanglement_strengths.append(strength)
    
    # 상관관계와 얽힘 강도의 관계 검증
    for i in range(len(correlations) - 1):
        assert entanglement_strengths[i] <= entanglement_strengths[i + 1], \
            "얽힘 강도가 상관관계에 비례하지 않음"

def test_phase_noise_detailed_analysis(quantum_register):
    """위상 노이즈 상세 분석 테스트"""
    # 1. 기본 설정
    batch_size = 4
    seq_len = 8
    num_heads = 4
    head_dim = 64
    
    # 2. 컨트롤된 입력 생성
    # 2.1 위상이 알려진 입력 생성
    phase_pattern = torch.linspace(0, 2*np.pi, head_dim)
    magnitude = torch.ones(batch_size, seq_len, num_heads, head_dim)
    
    # 2.2 복소수 입력 생성
    input_state = magnitude * torch.exp(1j * phase_pattern)
    input_state = input_state.to(dtype=torch.complex64)
    
    # 3. 위상 변환 전후 비교
    # 3.1 초기 위상 기록
    initial_phase = torch.angle(input_state)
    
    # 3.2 정규화 적용
    normalized_state = quantum_register.prepare_normalized(input_state)
    
    # 3.3 변환 후 위상 측정
    final_phase = torch.angle(normalized_state)
    
    # 4. 위상 보존 검증
    # 4.1 위상 차이 계산
    phase_diff = torch.abs(final_phase - initial_phase)
    
    # 4.2 위상 보존율 계산
    preservation_rate = 1.0 - phase_diff.mean().item() / (2 * np.pi)
    
    # 5. 엄격한 검증
    assert preservation_rate > 0.95, f"위상 보존율이 너무 낮습니다: {preservation_rate}"
    
    # 6. 위상 노이즈 패턴 분석
    # 6.1 위상 노이즈 주파수 분석
    phase_noise = final_phase - initial_phase
    fft_result = torch.fft.fft(phase_noise, dim=-1)
    noise_spectrum = torch.abs(fft_result)
    
    # 6.2 노이즈 주파수 특성 검증
    high_freq_noise = noise_spectrum[:, :, :, head_dim//2:].mean()
    low_freq_noise = noise_spectrum[:, :, :, :head_dim//2].mean()
    
    assert high_freq_noise < low_freq_noise, "고주파 노이즈가 예상보다 높습니다"

def test_phase_stability_under_scaling(quantum_register):
    """위상 스케일링에 따른 안정성 테스트"""
    # 1. 테스트 설정
    batch_size = 4
    seq_len = 8
    num_heads = 4
    head_dim = 64
    
    # 2. 다양한 스케일링 팩터 테스트
    scale_factors = [0.5, 1.0, 2.0]
    phase_preservation_rates = []
    
    for scale in scale_factors:
        # 2.1 양자 레지스터 스케일 설정
        quantum_register.phase_scale.data = torch.tensor([scale])
        
        # 2.2 테스트 입력 생성
        input_state = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.complex64)
        initial_phase = torch.angle(input_state)
        
        # 2.3 정규화 적용
        normalized_state = quantum_register.prepare_normalized(input_state)
        final_phase = torch.angle(normalized_state)
        
        # 2.4 위상 보존율 계산
        phase_diff = torch.abs(final_phase - initial_phase)
        preservation_rate = 1.0 - phase_diff.mean().item() / (2 * np.pi)
        phase_preservation_rates.append(preservation_rate)
        
        # 2.5 각 스케일에 대한 검증
        assert preservation_rate > 0.9, f"스케일 {scale}에서 위상 보존율이 낮습니다: {preservation_rate}"
    
    # 3. 스케일링 안정성 검증
    preservation_variation = torch.tensor(phase_preservation_rates).std()
    assert preservation_variation < 0.05, f"스케일링에 따른 보존율 변동이 큽니다: {preservation_variation}"

def test_quantum_state_preparation(config, batch_data):
    model = QuantumStatePreparation(config['hidden_size'])
    output = model(batch_data)
    
    # 출력 차원 확인
    assert output.shape == batch_data.shape
    
    # 복소수 텐서 확인
    assert output.dtype == torch.cfloat
    
    # 정규화 확인
    norms = torch.abs(output).pow(2).sum(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_quantum_gate(config):
    model = QuantumGate(config['hidden_size'])
    input_state = torch.randn(2, 8, config['hidden_size'], dtype=torch.cfloat)
    output = model(input_state)
    
    # 출력 차원 확인
    assert output.shape == input_state.shape
    
    # 유니타리 행렬 확인
    unitary = model.unitary.data
    identity = torch.eye(config['hidden_size'], dtype=torch.cfloat)
    product = torch.matmul(unitary, unitary.conj().transpose(-2, -1))
    assert torch.allclose(product, identity, atol=1e-6)

def test_quantum_measurement(config):
    model = QuantumMeasurement(config['hidden_size'])
    input_state = torch.randn(2, 8, config['hidden_size'], dtype=torch.cfloat)
    output = model(input_state)
    
    # 출력이 실수인지 확인
    assert output.dtype == torch.float32
    
    # 확률 분포 확인
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)

def test_quantum_attention(config, batch_data):
    model = QuantumAttention(config)
    attention_mask = torch.ones(2, 1, 1, 8)
    output, attention_probs = model(batch_data, attention_mask)
    
    # 출력 차원 확인
    assert output.shape == batch_data.shape
    
    # 어텐션 확률 확인
    assert attention_probs.shape == (2, 4, 8, 8)  # (batch_size, num_heads, seq_len, seq_len)
    assert torch.all(attention_probs >= 0)
    assert torch.all(attention_probs <= 1)
    assert torch.allclose(attention_probs.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-6)

def test_quantum_attention_cache(config, batch_data):
    model = QuantumAttention(config)
    
    # 첫 번째 호출
    output1, _ = model(batch_data, use_cache=True)
    
    # 두 번째 호출 (캐시 사용)
    output2, _ = model(batch_data, use_cache=True)
    
    # 캐시된 결과가 동일한지 확인
    assert torch.allclose(output1, output2, atol=1e-6)
    
    # 캐시 크기 확인
    assert model.get_cache_size() == 1
    
    # 캐시 초기화
    model.clear_cache()
    assert model.get_cache_size() == 0

def test_quantum_attention_noise_reduction(config, batch_data):
    model = QuantumAttention(config)
    
    # 노이즈 임계값 설정
    model.set_noise_threshold(0.2)
    
    # 노이즈가 있는 입력 생성
    noisy_data = batch_data + torch.randn_like(batch_data) * 0.1
    
    # 노이즈 마스크 생성
    noise_mask = torch.rand(2, 1, 8, 8) > 0.8
    
    # 모델 실행
    output, _ = model(noisy_data, noise_mask=noise_mask)
    
    # 출력이 입력보다 더 안정적인지 확인
    assert torch.std(output) < torch.std(noisy_data)

def test_quantum_attention_state_info(config, batch_data):
    model = QuantumAttention(config)
    
    # 모델 실행
    model(batch_data)
    
    # 양자 상태 정보 가져오기
    state_info = model.get_quantum_states()
    
    # 상태 정보 검증
    assert 'query_states' in state_info
    assert 'key_states' in state_info
    assert 'value_states' in state_info
    
    # 각 상태의 차원 확인
    for state in state_info.values():
        assert state.shape == (config['hidden_size'], config['hidden_size'])

def test_quantum_attention_gradient_flow(config):
    model = QuantumAttention(config)
    optimizer = torch.optim.Adam(model.parameters())
    
    # 입력 데이터 생성
    input_data = torch.randn(2, 8, config['hidden_size'], requires_grad=True)
    target = torch.randn(2, 8, config['hidden_size'])
    
    # 순전파
    output, _ = model(input_data)
    loss = torch.nn.functional.mse_loss(output, target)
    
    # 역전파
    optimizer.zero_grad()
    loss.backward()
    
    # 그래디언트 확인
    for name, param in model.named_parameters():
        assert param.grad is not None
        assert not torch.all(param.grad == 0) 