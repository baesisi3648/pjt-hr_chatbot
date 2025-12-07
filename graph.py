"""
ZIC-TALK HR 챗봇 - LangGraph 워크플로우 엔진
대화 맥락을 이해하고 3중 검증(Draft-Critic-Rewrite)을 수행합니다.
"""
import os
from typing import TypedDict, Literal, List, Dict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# 환경 설정
load_dotenv()

# ========== 설정 상수 ==========
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "company-rules")
PINECONE_NAMESPACE = "rules-2025"
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
MAX_CHAT_HISTORY = int(os.getenv("MAX_CHAT_HISTORY", "6"))
MAX_REVISION_COUNT = int(os.getenv("MAX_REVISION_COUNT", "2"))

# ========== 시스템 프롬프트 ==========
REWRITE_SYSTEM_PROMPT = """당신은 대화 맥락을 이해하여 질문을 재작성하는 전문가입니다.

사용자의 현재 질문이 이전 대화를 참조하는 경우(예: "그럼 그건?", "더 알려줘"), 
이전 대화 내용을 바탕으로 **독립적이고 명확한 질문**으로 재작성하세요.

예시:
- 이전 대화에서 "연차"에 대해 이야기했고, 현재 질문이 "그럼 월차는?"이면
  → "취업규칙에서 월차 휴가는 어떻게 되나요?"
  
- 이전 대화 없이 "연차는 몇일인가요?"라고 물으면
  → "취업규칙에서 연차 휴가는 몇일인가요?" (그대로 유지 또는 명확화)

**중요**: 
- 재작성된 질문만 출력하세요. 설명이나 부가 문구 없이.
- 취업규칙/인사규정 맥락을 유지하세요."""

DRAFT_SYSTEM_PROMPT = """당신은 회사 취업규칙 전문 상담사입니다.

주어진 규정 원문을 바탕으로 정확하고 친절하게 답변하세요.

**답변 원칙**:
1. 규정에 명시된 내용만 답변 (추측 금지)
2. 조항 번호와 함께 근거를 명확히 제시
3. 사용자 친화적인 설명 추가
4. 규정에 없는 내용이면 "해당 내용은 규정에 명시되어 있지 않습니다"라고 답변"""

CRITIQUE_SYSTEM_PROMPT = """당신은 엄격한 사실 검증 전문가입니다.

주어진 답변이 규정 원문에 **정확히 일치**하는지 검증하세요.

**평가 기준**:
- PASS: 모든 내용이 규정에 근거하며 사실과 일치
- FAIL: 규정에 없는 내용 추측, 잘못된 해석, 조항 번호 오류 등

**출력 형식**:
평가: PASS 또는 FAIL
이유: (FAIL인 경우 구체적인 문제점 지적)"""

# ========== 상태 정의 ==========
class GraphState(TypedDict):
    question: str                       # 현재 처리 중인 질문 (변환된 쿼리)
    original_question: str              # 사용자의 원래 질문
    context: str                        # 검색된 규정 원문
    draft: str                          # 생성된 답변 초안
    critique: str                       # 감사관의 지적사항
    grade: str                          # 평가 결과 (PASS / FAIL)
    revision_count: int                 # 수정 횟수
    chat_history: List[Dict[str, str]]  # 대화 기록

# ========== 컴포넌트 초기화 ==========
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=PINECONE_NAMESPACE
)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

# ========== 노드 함수들 ==========
def rewrite_question(state: GraphState) -> GraphState:
    """대화 기록을 참고하여 현재 질문을 독립적인 질문으로 재작성"""
    question = state["original_question"]
    chat_history = state.get("chat_history", [])
    
    if chat_history and len(chat_history) > 0:
        # 최근 대화만 참고
        recent_history = chat_history[-MAX_CHAT_HISTORY:]
        history_text = "\n".join([
            f"{'사용자' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in recent_history
        ])
        
        messages = [
            SystemMessage(content=REWRITE_SYSTEM_PROMPT),
            HumanMessage(content=f"""이전 대화:
{history_text}

현재 질문: {question}

재작성된 질문:""")
        ]
        
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        
        print(f"\n🔄 [질문 재작성]")
        print(f"   원본: {question}")
        print(f"   재작성: {rewritten}")
        
        state["question"] = rewritten
    else:
        state["question"] = question
        print(f"\n📝 [첫 질문] {question}")
    
    return state


def retrieve_context(state: GraphState) -> GraphState:
    """벡터 DB에서 관련 규정을 검색"""
    question = state["question"]
    print(f"\n🔍 [규정 검색] '{question}'에 대한 관련 조항 검색 중...")
    
    docs = retriever.invoke(question)
    
    context_parts = []
    for i, doc in enumerate(docs, 1):
        article_title = doc.metadata.get("article_title", "Unknown")
        content = doc.page_content
        context_parts.append(f"[문서 {i}] {article_title}\n{content}")
    
    context = "\n\n---\n\n".join(context_parts)
    state["context"] = context
    
    print(f"   ✅ 총 {len(docs)}개의 관련 조항을 찾았습니다.")
    return state


def generate_draft(state: GraphState) -> GraphState:
    """검색된 규정을 바탕으로 초안을 작성"""
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])
    
    print(f"\n✍️  [초안 작성] 답변 생성 중...")
    
    # 대화 기록을 간단히 요약하여 프롬프트에 포함
    history_context = ""
    if chat_history and len(chat_history) > 0:
        recent = chat_history[-4:]
        history_context = "\n\n이전 대화 참고:\n" + "\n".join([
            f"- {msg['role']}: {msg['content'][:100]}..."
            for msg in recent
        ])
    
    messages = [
        SystemMessage(content=f"""{DRAFT_SYSTEM_PROMPT}

**검색된 규정**:
{context}
{history_context}
"""),
        HumanMessage(content=question)
    ]
    
    response = llm.invoke(messages)
    draft = response.content
    state["draft"] = draft
    
    print(f"   ✅ 초안 작성 완료 (길이: {len(draft)} 글자)")
    return state


def critique_answer(state: GraphState) -> GraphState:
    """작성된 답변을 팩트체크하고 평가"""
    draft = state["draft"]
    context = state["context"]
    question = state["question"]
    
    print(f"\n🔍 [팩트체크] 답변 검증 중...")
    
    messages = [
        SystemMessage(content=CRITIQUE_SYSTEM_PROMPT),
        HumanMessage(content=f"""질문: {question}

답변:
{draft}

규정 원문:
{context}

평가를 시작하세요:""")
    ]
    
    response = llm.invoke(messages)
    critique = response.content
    state["critique"] = critique
    
    if "PASS" in critique.split('\n')[0].upper():
        state["grade"] = "PASS"
        print(f"   ✅ 검증 통과!")
    else:
        state["grade"] = "FAIL"
        print(f"   ❌ 검증 실패 - 수정 필요")
        print(f"   사유: {critique[:100]}...")
    
    return state


def rewrite_answer(state: GraphState) -> GraphState:
    """피드백을 반영하여 답변을 수정"""
    draft = state["draft"]
    critique = state["critique"]
    context = state["context"]
    question = state["question"]
    
    state["revision_count"] = state.get("revision_count", 0) + 1
    
    print(f"\n🔧 [답변 수정] {state['revision_count']}차 수정 중...")
    
    messages = [
        SystemMessage(content=f"""당신은 피드백을 받아 답변을 개선하는 전문가입니다.

**검증 피드백**:
{critique}

**규정 원문**:
{context}

위 피드백을 반영하여 답변을 수정하세요. 반드시 규정에 근거한 내용만 포함하세요.
"""),
        HumanMessage(content=f"""질문: {question}

기존 답변:
{draft}

수정된 답변:""")
    ]
    
    response = llm.invoke(messages)
    revised = response.content
    state["draft"] = revised
    
    print(f"   ✅ 수정 완료")
    return state


def should_continue(state: GraphState) -> Literal["rewrite", "end"]:
    """답변이 통과했는지, 재작성이 필요한지 판단"""
    if state["grade"] == "PASS":
        return "end"
    
    if state.get("revision_count", 0) >= MAX_REVISION_COUNT:
        print("\n⚠️  최대 수정 횟수 도달 - 현재 답변으로 종료합니다.")
        return "end"
    
    return "rewrite"


# ========== 그래프 구성 ==========
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_draft)
workflow.add_node("critique", critique_answer)
workflow.add_node("rewrite", rewrite_answer)

# 엣지 연결
workflow.set_entry_point("rewrite_question")
workflow.add_edge("rewrite_question", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")

# 조건부 엣지
workflow.add_conditional_edges(
    "critique",
    should_continue,
    {
        "rewrite": "rewrite",
        "end": END
    }
)
workflow.add_edge("rewrite", "critique")

# 컴파일
app = workflow.compile()


# ========== 실행 헬퍼 함수 ==========
def run_workflow(question: str, chat_history: List[Dict[str, str]] = None):
    """
    워크플로우를 실행하고 최종 답변을 반환
    
    Args:
        question: 사용자 질문
        chat_history: 이전 대화 기록 [{"role": "user", "content": "..."}, ...]
    
    Returns:
        최종 답변 문자열
    """
    if chat_history is None:
        chat_history = []
    
    inputs = {
        "original_question": question,
        "question": question,
        "context": "",
        "draft": "",
        "critique": "",
        "grade": "",
        "revision_count": 0,
        "chat_history": chat_history
    }
    
    result = app.invoke(inputs)
    return result["draft"]


# ========== 테스트 코드 ==========
if __name__ == "__main__":
    print("="*80)
    print("ZIC-TALK 챗봇 테스트")
    print("="*80)
    
    # 첫 번째 질문
    history = []
    q1 = "연차는 얼마나 주나요?"
    print(f"\n👤 사용자: {q1}")
    answer1 = run_workflow(q1, history)
    print(f"\n🤖 AI: {answer1}")
    
    # 대화 기록에 추가
    history.append({"role": "user", "content": q1})
    history.append({"role": "assistant", "content": answer1})
    
    # 후속 질문 (대화 맥락 참조)
    q2 = "그럼 월차는?"
    print(f"\n👤 사용자: {q2}")
    answer2 = run_workflow(q2, history)
    print(f"\n🤖 AI: {answer2}")
