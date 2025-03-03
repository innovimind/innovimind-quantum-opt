# InnoviMind 프로젝트 기여 가이드라인

InnoviMind 프로젝트에 관심을 가져주셔서 감사합니다. 이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 기여 방법

1. 이슈 제기
   - 버그 리포트
   - 기능 제안
   - 문서 개선 제안

2. 풀 리퀘스트 제출
   - 버그 수정
   - 새로운 기능 구현
   - 문서 개선

## 개발 환경 설정

1. 저장소 클론
```bash
git clone https://github.com/innovimind/innovimind-quantum-opt.git
cd innovimind-quantum-opt
```

2. 가상환경 설정
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 코드 스타일

- Python 코드는 PEP 8 스타일 가이드를 따릅니다.
- 모든 함수와 클래스에 docstring을 작성해주세요.
- 타입 힌트를 사용해주세요.
- 테스트 코드를 작성해주세요.

## 커밋 메시지 규칙

```
<type>: <description>

[optional body]

[optional footer]
```

커밋 타입:
- feat: 새로운 기능
- fix: 버그 수정
- docs: 문서 수정
- style: 코드 포맷팅
- refactor: 코드 리팩토링
- test: 테스트 코드
- chore: 빌드 프로세스 변경

## 테스트

풀 리퀘스트를 제출하기 전에 모든 테스트가 통과하는지 확인해주세요:

```bash
pytest tests/
```

## 문서화

- 새로운 기능을 추가할 때는 관련 문서도 함께 업데이트해주세요.
- API 문서는 Google 스타일의 docstring을 사용합니다.
- 예제 코드를 포함해주세요.

## 리뷰 프로세스

1. 풀 리퀘스트 제출
2. CI 테스트 통과 확인
3. 코드 리뷰 진행
4. 필요한 수정 사항 반영
5. 승인 및 머지

## 라이선스

기여하신 모든 코드는 프로젝트의 Apache 2.0 라이선스를 따르게 됩니다.

## 질문이나 도움이 필요하신가요?

- 이메일: innovimindcompany@gmail.com
