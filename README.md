

# pjt-hr-chatbot

**LangGraph 기반 3중 팩트체크 에이전트 (Zero Hallucination RAG)**

이 프로젝트는 인천메트로서비스 취업규칙 데이터를 바탕으로, LangGraph 기반의 3중 팩트체크 로직을 구현하여 할루시네이션(거짓 답변)이 없는 사내 규정 QA 에이전트를 구축한 AI/MLOps 프로젝트입니다. 이는 Code-First(LangChain) 방식을 통해 복잡한 추론 흐름을 제어하고 시스템 견고성을 확보하는 데 중점을 둡니다.

-----

### 🧭 핵심 요약

목표: **규정집에 없는 내용은 절대 답변하지 않는** 에이전트 구현을 통해 **Zero Hallucination** 기반의 사내 규정 QA 시스템 구축.

산출물: Python LangGraph 엔진 (`graph.py`), Streamlit 웹 UI (`app.py`), 커스텀 Regex 기반 데이터 인제스트 파이프라인.

주요 통찰 예

  * **3중 검증 로직:** `Draft` → `Critic` → `Rewrite` 순환 구조를 LangGraph로 구현하여 답변 오류율 획기적 감소.
  * **Semantic Gap 해소:** 사용자의 구어체 질문("짤려?")을 규정집 용어("직권면직")로 변환하는 **Query Rewriting** 기술 적용.
  * **모델 최적화:** 단계별 난이도에 따라 `GPT-4o`와 `GPT-4o-mini`를 분리 배치하여 속도와 비용 절감 달성.

⚠️ **주의사항:** Local 환경에서는 HTTPS/OAuth 문제로 Slack 대신 **Streamlit 내부 Chat**으로 테스트합니다.

-----

### 📚 목차

1.  프로젝트 소개 (Zero Hallucination RAG)
2.  프로젝트 구조 및 기술 스택
3.  데이터
4.  설치 & 실행 (Ingestion 포함)
5.  엔진 가이드 (LangGraph Flow)
6.  한계 & 향후 과제

-----

### 🔎 프로젝트 소개

이 프로젝트는 [인천메트로서비스 취업규칙]을 활용하여 다음을 수행합니다.

  * **문제 해결:** LLM의 기본 성향인 **할루시네이션(거짓 답변)** 경향을 구조적으로 차단합니다.
  * **아키텍처:** **LangGraph StateGraph**를 기반으로 검색(Retrieval), 생성(Generation), 비평(Critique) 단계를 직렬화하여 \*\*규정 위반 시 재작성(Rewrite)\*\*을 강제하는 사이클을 구현했습니다.
  * **비교 우위:** 기존 No-Code 툴(n8n)에서 검증이 어려웠던 **복잡한 조건부 루프**를 Python 코드로 구현하여 **제어의 투명성**을 확보했습니다.

### 🗂️ 프로젝트 구조

```text
pjt-hr-chatbot/
├─ README.md
├─ .env                 # API Keys (OpenAI, Pinecone)
├─ requirements.txt     # 설치 목록
├─ rules.txt            # [원본] 취업규칙 전체 텍스트 (PDF 내용)
├─ ingest.py            # [ETL] 데이터 전처리 및 Pinecone Upsert
├── graph.py             # [ENGINE] LangGraph State, Nodes, Edges (The Brain)
└── app.py               # [UI] Streamlit Chat Interface
```

### 🗃️ 데이터

출처: 인천메트로서비스주식회사 취업규칙(2025)

  * **전처리:** Python `re` 모듈을 사용하여 **`제O조` 단위**로 텍스트를 분할(Chunking)하여 문맥 유실을 방지.
  * **저장소:** Pinecone Index (`company-rules`), Namespace (`rules-2025`).
  * **핵심 가치:** 각 조항은 메타데이터와 함께 저장되어 **"제25조에 따르면..."** 형식의 답변 근거 제시를 가능하게 함.

### 🛠️ 설치 & 실행 방법 (End-to-End)

본 프로젝트는 \*\*데이터 적재(Ingestion)\*\*와 **챗봇 실행(App)** 두 단계로 나뉘어 진행됩니다.

1)  저장소 클론 및 패키지 설치 (터미널)

<!-- end list -->

```bash
# REPO_URL은 본인의 GitHub 주소로 대체
git clone <REPO_URL>
cd pjt-hr_chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2)  환경 변수 설정

`.env` 파일에 `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` 등을 정확히 입력합니다.

3)  데이터 적재 (Ingestion)

**주의:** 이 단계는 Pinecone DB에 데이터를 넣는 과정으로, **딱 한 번만 실행합니다.**

```bash
python ingest.py
```

4)  챗봇 실행 및 테스트 (Streamlit UI)

<!-- end list -->

```bash
streamlit run app.py
```

### 🧠 엔진 가이드 (LangGraph Flow)

이 봇은 4개의 노드와 1개의 조건부 루프로 구성됩니다.

1.  **Transform Query:** 사용자의 은어를 공식 용어로 번역 ("짤려?" → "직권면직").
2.  **Retrieve:** 변환된 쿼리로 Pinecone 검색.
3.  **Draft Generator:** 검색된 Context만을 사용해 초안 작성.
4.  **Critic Agent:** 초안을 검증 (`PASS/FAIL`).
      * **If FAIL:** $\rightarrow$ `Rewrite Agent` $\rightarrow$ `Critic Agent` (재검증 루프 실행)
      * **If PASS:** $\rightarrow$ 최종 답변 출력.

-----

### ⚠️ 한계 & 향후 과제

| 구분 | 이슈 및 필요성 | 개선 방향 |
| :--- | :--- | :--- |
| **UX / Memory** | 단발성 질문만 가능하여 복잡한 대화 맥락 유지 불가. | `SqliteSaver` 또는 `Redis`를 활용한 **대화 맥락 기억(History Aware System)** 구현. |
| **Accuracy** | Dense Vector 검색만으로는 '제25조' 같은 정확한 조항 번호 검색에 취약. | \*\*Hybrid Search (BM25 + Dense)\*\*를 적용하여 키워드 매칭 능력을 보강. |
| **Deployment** | Streamlit은 포트폴리오용이며, 실무 사용을 위한 Slack 연동 실패 이슈 잔존. | **FastAPI** 기반의 봇 서버 구축 및 **HTTPS 환경**에서 Slack/Discord 연동 재시도. |

### 👥 만든이

배성우

-----