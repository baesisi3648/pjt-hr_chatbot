"""
ZIC-TALK HR 챗봇 - Streamlit UI
대화 맥락을 이해하고 취업규칙 기반 정확한 답변을 제공합니다.
"""
import streamlit as st
from graph import run_workflow
import time
from datetime import datetime
import json

# ========== 유틸리티 함수 ==========
def get_timestamp():
    """현재 시간을 HH:MM 형식으로 반환"""
    return datetime.now().strftime("%H:%M")


def export_chat_to_txt(messages):
    """대화 내역을 텍스트로 변환"""
    lines = []
    lines.append("=" * 80)
    lines.append("ZIC-TALK HR 챗봇 대화 내역")
    lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    for i, msg in enumerate(messages, 1):
        role = "👤 사용자" if msg["role"] == "user" else "🤖 AI"
        timestamp = msg.get("timestamp", "")
        
        lines.append(f"[{i}] {role} ({timestamp})")
        lines.append("-" * 80)
        lines.append(msg["content"])
        lines.append("")
    
    return "\n".join(lines)


def export_chat_to_json(messages):
    """대화 내역을 JSON으로 변환"""
    export_data = {
        "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_messages": len(messages),
        "messages": messages
    }
    return json.dumps(export_data, ensure_ascii=False, indent=2)


# ========== 페이지 설정 ==========
st.set_page_config(
    page_title="ZIC-TALK HR 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 커스텀 CSS ==========
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .sub-title {
        font-size: 1.2rem;
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ========== 헤더 ==========
st.markdown('<h1 class="main-title">🤖 ZIC-TALK HR 챗봇</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">🚀 LangGraph 기반 3중 검증 | 대화 맥락 이해 가능</p>', unsafe_allow_html=True)

# ========== 세션 상태 초기화 ==========
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "안녕하세요! 👋\n\n저는 **ZIC-TALK HR 챗봇**입니다.\n\n취업규칙에 대해 궁금한 점을 물어보시면, 관련 규정을 검색하고 3중 팩트체크를 거쳐 정확한 답변을 드립니다.\n\n**예시 질문:**\n- 연차는 얼마나 주나요?\n- 퇴직금 계산 방법은?\n- 육아휴직 조건이 어떻게 되나요?\n\n편하게 질문해주세요! 😊",
        "timestamp": get_timestamp()
    }]

if "total_questions" not in st.session_state:
    st.session_state.total_questions = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# ========== 사이드바 ==========
with st.sidebar:
    st.markdown("## 📊 대시보드")
    
    # 통계 정보
    col1, col2 = st.columns(2)
    with col1:
        st.metric("💬 총 질문 수", st.session_state.total_questions)
    with col2:
        elapsed = int(time.time() - st.session_state.start_time)
        st.metric("⏱️ 세션 시간", f"{elapsed//60}분")
    
    st.markdown("---")
    
    # 기능 안내
    st.markdown("## 🎯 주요 기능")
    st.markdown("""
    ✅ **대화 맥락 이해**  
    이전 대화를 기억하여 후속 질문에 답변
    
    🔍 **3중 검증 시스템**  
    Draft → Critic → Rewrite 프로세스
    
    📚 **규정 기반 답변**  
    Pinecone 벡터 DB에서 관련 조항 검색
    
    🎨 **사용자 친화적 UI**  
    깔끔하고 직관적인 인터페이스
    """)
    
    st.markdown("---")
    
    # 대화 초기화 버튼
    if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "대화 기록이 초기화되었습니다. 새로운 질문을 시작해주세요! 😊",
            "timestamp": get_timestamp()
        }]
        st.session_state.total_questions = 0
        st.session_state.start_time = time.time()
        st.rerun()
    
    st.markdown("---")
    
    # 대화 내보내기 기능
    st.markdown("## 📥 대화 내보내기")
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        if st.button("📄 TXT", use_container_width=True):
            if len(st.session_state.messages) > 1:
                txt_content = export_chat_to_txt(st.session_state.messages)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ 다운로드",
                    data=txt_content,
                    file_name=f"chat_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("저장할 대화가 없습니다.")
    
    with col_export2:
        if st.button("📊 JSON", use_container_width=True):
            if len(st.session_state.messages) > 1:
                json_content = export_chat_to_json(st.session_state.messages)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ 다운로드",
                    data=json_content,
                    file_name=f"chat_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.warning("저장할 대화가 없습니다.")
    
    st.markdown("---")
    
    # 시스템 정보
    with st.expander("⚙️ 시스템 정보"):
        st.markdown("""
        **버전:** v2.0  
        **모델:** GPT-4o-mini  
        **임베딩:** text-embedding-3-small  
        **벡터DB:** Pinecone  
        **프레임워크:** LangGraph
        """)

# ========== 메인 채팅 영역 ==========
st.markdown("## 💬 대화창")

# 채팅 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "timestamp" in msg:
            st.caption(f"🕐 {msg['timestamp']}")

# ========== 사용자 입력 처리 ==========
if prompt := st.chat_input("질문을 입력하세요... (예: 연차는 얼마나 주나요?)"):
    current_time = get_timestamp()
    
    # 사용자 메시지 추가
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": current_time
    })
    
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"🕐 {current_time}")
    
    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("🔍 규정 검색 및 팩트체크 중... (약 10~15초 소요)"):
            try:
                # 대화 기록 준비 (시스템 메시지 제외)
                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[:-1]
                    if msg["role"] in ["user", "assistant"]
                ]
                
                # 워크플로우 실행
                start = time.time()
                answer = run_workflow(prompt, chat_history)
                elapsed = time.time() - start
                
                # 답변 표시
                st.markdown(answer)
                st.caption(f"🕐 {get_timestamp()} | ⏱️ 처리 시간: {elapsed:.1f}초")
                
                # 답변 저장
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "timestamp": get_timestamp()
                })
                
                # 통계 업데이트
                st.session_state.total_questions += 1
                
            except Exception as e:
                error_msg = f"❌ 오류가 발생했습니다: {str(e)}\n\n다시 시도해주세요."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": get_timestamp()
                })

# ========== 하단 정보 ==========
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("💡 **TIP:** 이전 대화를 참조하여 후속 질문을 할 수 있습니다!")

with col2:
    st.success("✅ **신뢰도:** 3중 팩트체크로 할루시네이션 최소화")

with col3:
    st.warning("⚠️ **주의:** 최종 결정은 인사팀과 상의하세요")