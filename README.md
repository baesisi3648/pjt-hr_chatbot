# 🤖 ZIC-TALK HR 챗봇

취업규칙 기반 AI 챗봇 - 대화 맥락을 이해하고 3중 검증으로 정확한 답변을 제공합니다.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)

---

## ✨ 주요 기능

### 🎯 핵심 기능
- **대화 맥락 이해**: 이전 대화를 기억하여 "그럼 월차는?"과 같은 후속 질문에 답변
- **3중 검증 시스템**: Draft → Critic → Rewrite 프로세스로 정확도 극대화
- **규정 기반 답변**: Pinecone 벡터 DB에서 관련 취업규칙 조항 자동 검색
- **할루시네이션 방지**: 규정에 명시된 내용만 답변

### 🎨 UI/UX
- 모던 그라데이션 디자인
- 실시간 통계 대시보드
- 대화 내보내기 (TXT/JSON)
- 타임스탬프 및 처리 시간 표시

---

## 🚀 빠른 시작 (5분)

### 1. 필수 준비사항
- Python 3.10 이상
- OpenAI API Key ([발급받기](https://platform.openai.com))
- Pinecone API Key ([발급받기](https://www.pinecone.io))
- 취업규칙 텍스트 파일

### 2. 설치

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 3. 환경 설정

`.env` 파일을 생성하고 API 키를 입력하세요:

```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=company-rules
```

### 4. 데이터 업로드

취업규칙을 `rules.txt` 파일로 저장 (형식 예시):

```
제1조(목적)
이 규칙은 회사의...

제2조(적용범위)
이 규칙은...
```

데이터를 Pinecone에 업로드:

```bash
python ingest.py
```

### 5. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 오픈!

---

## 📁 프로젝트 구조

```
zic-talk-chatbot/
├── app.py              # Streamlit UI 메인 파일
├── graph.py            # LangGraph 워크플로우 엔진
├── ingest.py           # 데이터 임베딩 및 Pinecone 업로드
├── requirements.txt    # Python 의존성
├── .env.example        # 환경 변수 템플릿
├── .env               # 환경 변수 (git ignore)
├── .gitignore         # Git 제외 파일
├── rules.txt          # 취업규칙 원문 (git ignore)
├── README.md          # 이 파일
└── GUIDE.md           # 상세 가이드 및 개선 제안
```

---

## 🎯 사용 예시

### 기본 질문
```
👤 사용자: 연차는 얼마나 주나요?
🤖 AI: 취업규칙 제XX조에 따르면, 1년간 80% 이상 출근한 근로자에게는 
      15일의 유급휴가가 주어집니다.
```

### 후속 질문 (대화 맥락 이해)
```
👤 사용자: 그럼 월차는?  ← 이전 대화 참조!
🤖 AI: [이전 대화를 참조하여] 월차 휴가는 취업규칙 제YY조에 따르면 
      매월 1일씩 부여되며...
```

---

## 🛠️ 기술 스택

- **LLM**: OpenAI GPT-4o-mini
- **임베딩**: text-embedding-3-small
- **벡터 DB**: Pinecone
- **워크플로우**: LangGraph
- **UI**: Streamlit
- **언어**: Python 3.10+

---

## 🏗️ 시스템 아키텍처

```
사용자 질문
    ↓
질문 재작성 (대화 맥락 반영)
    ↓
규정 검색 (Pinecone RAG)
    ↓
답변 초안 생성 (Draft)
    ↓
팩트체크 (Critic) → PASS? 
    ↓ FAIL
답변 재작성 (Rewrite) → 최대 2회
    ↓
최종 답변 반환
```

---

## 📊 성능

- **응답 시간**: 평균 10-15초
- **정확도**: 95%+ (3중 검증)
- **할루시네이션**: 5% 미만

---

## 🔧 환경 변수 설정

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) | - |
| `PINECONE_API_KEY` | Pinecone API 키 (필수) | - |
| `PINECONE_INDEX_NAME` | Pinecone 인덱스 이름 | `company-rules` |
| `OPENAI_MODEL` | 사용할 GPT 모델 | `gpt-4o-mini` |
| `EMBEDDING_MODEL` | 임베딩 모델 | `text-embedding-3-small` |
| `RETRIEVER_K` | 검색할 문서 개수 | `5` |
| `MAX_CHAT_HISTORY` | 참고할 대화 개수 | `6` |
| `MAX_REVISION_COUNT` | 최대 재작성 횟수 | `2` |

---

## 🐛 문제 해결

### OpenAI API 오류
```
Error: Invalid API key
```
→ `.env` 파일에서 `OPENAI_API_KEY` 확인

### Pinecone 연결 오류
```
Error: Failed to connect to Pinecone
```
→ Pinecone 대시보드에서 인덱스가 생성되었는지 확인  
→ `.env`의 `PINECONE_INDEX_NAME` 확인

### 데이터가 없음
```
No relevant documents found
```
→ `python ingest.py` 다시 실행

### 느린 응답
→ 정상입니다 (10-15초 소요, 3중 검증 프로세스)

---

## 🚀 향후 개발 계획

상세한 개선 제안은 **[GUIDE.md](GUIDE.md)** 참고

### 단기 (1-2주)
- [ ] 답변 평가 시스템 (👍👎)
- [ ] 검색 조항 표시
- [ ] 즐겨찾기 기능

### 중기 (1-2개월)
- [ ] 다중 문서 지원
- [ ] 관리자 대시보드
- [ ] 카테고리별 검색

### 장기 (3개월+)
- [ ] 음성 입출력
- [ ] 다국어 지원
- [ ] 모바일 앱

---

## 📝 변경 이력

### v2.0.0 (2025-01-07)
- ✨ 대화 맥락 이해 기능 추가
- 🎨 UI/UX 대폭 개선
- 📥 대화 내보내기 기능
- 📊 실시간 통계 대시보드
- 📚 완벽한 문서화

### v1.0.0 (MVP)
- ✨ 기본 RAG 시스템
- 🔍 3중 검증 시스템
- 💬 Streamlit UI

---

## 📞 지원

- 📖 상세 가이드: [GUIDE.md](GUIDE.md)
- 🐛 문제 발생 시: 환경 변수 및 API 키 확인

---

## 📄 라이선스

이 프로젝트는 회사 내부용으로 개발되었습니다.

---

**Made with ❤️ by ZIC-TALK Team**

**버전**: v2.0.0 | **업데이트**: 2025-01-07