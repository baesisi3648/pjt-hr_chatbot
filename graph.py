import os
import json
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# 1. 환경 설정
load_dotenv()

# 2. 상태(State) 정의
class GraphState(TypedDict):
    question: str           # 현재 처리 중인 질문 (변환된 쿼리)
    original_question: str  # 사용자의 원래 질문 (참고용)
    context: str            # 검색된 규정 원문
    draft: str              # 생성된 답변 초안
    critique: str           # 감사관의 지적사항
    grade: str              # 평가 결과 (PASS / FAIL)
    revision_count: int     # 수정 횟수 (무한 루프 방지)

# 3. 컴포넌트 초기화
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = PineconeVectorStore.from_existing_index(
    index_name=os.environ.get("PINECONE_INDEX_NAME", "company-rules"),
    embedding=embeddings,
    namespace="rules-2025"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 모델 설정
# llm_draft: 복잡한 추론이 필요한 초안 작성용 (GPT-4o)
# llm_critic & transformer: 단순 작업 및 검증용 (GPT-4o-mini) - 속도/비용 최적화
llm_draft = ChatOpenAI(model="gpt-4o", temperature=0)
llm_critic = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
llm_transformer = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ==========================================
# 4. 노드(Node) 함수 정의
# ==========================================

def transform_query_node(state: GraphState):
    """
    [0단계] 사용자의 질문을 분석하여, 검색 확률을 높이는 '최적의 검색어'로 확장/변환합니다.
    (하드코딩된 단어장 없이 LLM의 추론 능력을 활용합니다.)
    """
    print("\n🔄 [0] 질문 확장(Query Expansion) 중...")
    question = state["question"]
    
    # 변환용 LLM (gpt-4o-mini)
    llm_transformer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""
    당신은 기업 인사(HR) 규정 검색을 위한 '검색어 최적화 전문가'입니다.
    사용자의 질문은 구어체나 비공식 용어(은어)가 섞여 있어, 규정집 검색(Vector DB) 시 정확도가 떨어질 수 있습니다.
    
    [당신의 임무]
    1. 사용자의 질문 의도를 파악하세요.
    2. 질문에 포함된 핵심 단어를 **'기업 취업규칙'에서 주로 쓰이는 공식 법률/행정 용어**로 변환하세요.
    3. 혹시 모를 상황에 대비해 **유의어(Synonyms)**도 함께 포함하여 검색 쿼리를 풍성하게 만드세요.
    4. 결과는 오직 **변환된 검색어 문장**만 출력하세요. (설명 금지)

    [예시]
    User: "회사 며칠 안 나오면 잘려?"
    AI: "무단결근 시 직권면직 기준 및 징계 해고 사유 (결근, 무계결근)"
    
    User: "애 낳으면 언제까지 쉬어?"
    AI: "출산전후휴가 기간 및 육아휴직 신청 가능 기간 (모성보호)"

    [사용자 질문]
    {question}
    """
    
    # LLM이 스스로 생각해서 검색어를 만듭니다.
    better_question = llm_transformer.invoke([HumanMessage(content=prompt)]).content
    print(f"   👉 확장된 쿼리: '{better_question}'")
    
    return {"question": better_question, "original_question": question}

def retrieve_node(state: GraphState):
    """[1단계] 변환된 질문으로 규정을 검색합니다."""
    print("\n🔍 [1] 검색 중...")
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    return {"context": context, "revision_count": 0}

def draft_node(state: GraphState):
    print("\n📝 [2] 초안 작성 중...")
    question = state["question"]
    context = state["context"]

    # 할루시네이션 방지 + 유연한 해석을 위한 시스템 프롬프트
    system_prompt = """
    당신은 인천메트로서비스 규정집 기반의 팩트체크 봇입니다. 
    당신의 임무는 제공된 [참고할 취업규칙]을 바탕으로 사용자 질문에 답하는 것입니다.

    [답변 작성 원칙 (중요)]
    1. **[유의어 해석 허용]:** 사용자는 '무단결근', '짤린다', '월급' 같은 일상 용어를 쓰지만, 규정집은 '무계결근', '직권면직', '보수' 같은 행정 용어를 사용합니다.
       - 질문의 단어가 규정의 단어와 100% 일치하지 않더라도, **의미가 동일하다면 관련 규정으로 판단하고 답변하세요.**
       - 예: 질문 "무단결근" -> 규정 "무계결근" (답변 가능 O)
    
    2. **[할루시네이션 방지]:** 위 유의어 해석을 적용했음에도 불구하고, 전혀 관련 없는 내용(예: 재택근무)이라면 "규정에 없습니다"라고 답하고 종료하세요.
    
    3. **[답변 스타일]:**
       - 핵심 결론을 먼저 말하고, 문장 끝에 **근거 조항(예: 제12조 제4항)**을 괄호로 명시하세요.
       - "일반적으로", "통상적으로" 같은 사족은 붙이지 마세요.
    """
    
    user_message = f"""
    [참고할 취업규칙]
    {context}

    [질문]
    {question}
    """
    
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
    response = llm_draft.invoke(messages)
    return {"draft": response.content}

def critic_node(state: GraphState):
    """[3단계] 답변을 검증합니다 (JSON 출력)."""
    print("\n🕵️ [3] 팩트체크 중...")
    context = state["context"]
    draft = state["draft"]

    prompt = f"""
    당신은 엄격한 규정 준수 감사관입니다. 초안이 다음 기준을 위반했는지 검사하세요.
    
    [검증 기준]
    1. [규정 원문]에 없는 내용(외부 지식, 일반 상식)이 포함되었는가? -> 포함되면 **FAIL**
    2. "일반적으로", "통상적으로", "권장합니다" 같은 **뇌피셜 조언**이 포함되었는가? -> 포함되면 **FAIL**
    3. 규정에 없는 질문에 대해 "규정에 없습니다"라고 깔끔하게 거절했는가? -> 거절 후 사족을 붙였다면 **FAIL**
    4. 근거 조항(제O조)이 명시되었는가? (규정에 있는 경우)
    
    [규정 원문]
    {context}
    
    [초안 답변]
    {draft}

    [출력 형식 - JSON]
    {{
        "grade": "PASS" 또는 "FAIL",
        "critique": "PASS면 '적합', FAIL이면 구체적인 지적 사항"
    }}
    """
    response = llm_critic.invoke([HumanMessage(content=prompt)])
    result = json.loads(response.content)
    
    return {"grade": result["grade"], "critique": result["critique"]}

def rewrite_node(state: GraphState):
    """[4단계] 지적사항을 반영하여 답변을 수정합니다."""
    print("\n✏️ [4] 답변 수정 중...")
    draft = state["draft"]
    critique = state["critique"]
    revision_count = state["revision_count"]

    prompt = f"""
    당신은 편집자입니다. 감사관의 지적을 반영하여 답변을 수정하세요.
    
    [기존 초안]
    {draft}
    
    [지적 사항]
    {critique}
    
    위 내용을 반영하여 더 완벽한 답변을 작성하세요. (외부 지식 금지 원칙 준수)
    """
    response = llm_draft.invoke([HumanMessage(content=prompt)])
    
    return {"draft": response.content, "revision_count": revision_count + 1}

# ==========================================
# 5. 그래프(Workflow) 연결
# ==========================================

def check_pass_or_fail(state: GraphState):
    """조건부 엣지: 검증 결과에 따라 다음 단계 결정"""
    grade = state["grade"]
    count = state["revision_count"]

    if grade == "PASS":
        print("   ✅ 검증 통과!")
        return "pass"
    elif count >= 3:
        print("   🛑 수정 횟수 초과 (그냥 반환)")
        return "max_retries"
    else:
        print(f"   ❌ 검증 실패 (이유: {state['critique']}) -> 재작성")
        return "rewrite"

# 그래프 생성
workflow = StateGraph(GraphState)

# 노드 추가
workflow.add_node("transform_query", transform_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("draft", draft_node)
workflow.add_node("critic", critic_node)
workflow.add_node("rewrite", rewrite_node)

# 엣지 연결
workflow.set_entry_point("transform_query") # 시작점: 질문 변환
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("retrieve", "draft")
workflow.add_edge("draft", "critic")

# 조건부 분기
workflow.add_conditional_edges(
    "critic",
    check_pass_or_fail,
    {
        "pass": END,
        "max_retries": END,
        "rewrite": "rewrite"
    }
)

# 루프 연결
workflow.add_edge("rewrite", "critic")

# 컴파일
app = workflow.compile()

# ==========================================
# 6. 테스트 실행 코드
# ==========================================
if __name__ == "__main__":
    print("🤖 HR 챗봇 엔진 시동 (Query Rewriting 포함)...")
    
    # 은어가 포함된 테스트 질문
    test_query = "회사 며칠 안가면 짤려?"
    
    inputs = {"question": test_query}
    final_state = app.invoke(inputs)
    
    print("\nFINAL ANSWER:")
    print(final_state["draft"])