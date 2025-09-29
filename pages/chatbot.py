import os
import tempfile
import shutil
import streamlit as st
from dotenv import load_dotenv

# LangChain (최신 구조)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def _sidebar_dark_and_slider_fix():
    st.markdown(
        """
    <style>
      :root{
        /* Sidebar dark palette */
        --sb-bg:#0b1220;         /* 다크 네이비 */
        --sb-fg:#e2e8f0;         /* 본문 텍스트 */
        --sb-muted:#e5e7eb;      /* 🔸보조 텍스트: 더 밝게/진하게 */
        --sb-line:#1f2a44;

        --accent:#f1f5f9;   /* 거의 흰색 (상단) */
        --accent-2:#cbd5e1; /* 밝은 회색 (하단) */

        /* Main content neutrals */
        --ink:#0f172a;
        --muted:#475569;
        --line:#e2e8f0;
      }

      /* Sidebar Dark */
      section[data-testid="stSidebar"]{
        background:var(--sb-bg)!important; color:var(--sb-fg)!important;
        border-right:1px solid var(--sb-line);
      }
      section[data-testid="stSidebar"] *{ color:var(--sb-fg)!important; }
      section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,section[data-testid="stSidebar"] h3{
        color:var(--sb-fg)!important;
      }

      /* 🔸보조 텍스트/라벨: 더 선명 + 약간 굵게 */
      section[data-testid="stSidebar"] .stMarkdown p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] .stSelectbox label{
        color:var(--sb-muted)!important;
        font-weight:600!important;
      }

      /* Inputs in sidebar */
      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] textarea,
      section[data-testid="stSidebar"] select,
      section[data-testid="stSidebar"] .stTextInput input,
      section[data-testid="stSidebar"] .stNumberInput input{
        background:rgba(255,255,255,0.06)!important;
        border:1px solid var(--sb-line)!important;
      }

      /* 🔧 Slider cutoff fix */
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ padding-right:12px; }
      section[data-testid="stSidebar"] div[data-testid="stSlider"]{
        padding-right:12px; margin-right:2px; overflow:visible;
      }
      section[data-testid="stSidebar"] div[role="slider"]{
        box-shadow:0 0 0 2px rgba(20,184,166,0.25); border-radius:999px;
      }

      /* ✅ Radio: 색/정렬 깔끔하게 (red → teal, 정중앙 정렬) */
      /* Streamlit 라디오 인풋 컬러를 액센트로 통일 */
      input[type="radio"]{ accent-color: var(--accent); }
      /* 라벨/원형이 수직 중앙 정렬되도록 라벨 플렉스 정렬 */
      div[role="radiogroup"] label{
        display:flex; align-items:center; gap:.5rem;
        line-height:1.2; margin: .1rem 0;
      }
      /* 일부 환경에서 라디오 원이 1px 내려가 보이는 현상 보정 */
      div[role="radiogroup"] input[type="radio"]{
        transform: translateY(0px);
      }

      /* Buttons (sidebar/main 공통) */
      section[data-testid="stSidebar"] .stButton>button,
      [data-testid="stAppViewContainer"] .stButton>button{
        background:linear-gradient(180deg,var(--accent),var(--accent-2))!important;
        color:#001018!important;
        border:0!important; font-weight:800!important; letter-spacing:.2px;
        border-radius:10px; padding:.55rem 1rem;
      }
      section[data-testid="stSidebar"] .stButton>button:hover,
      [data-testid="stAppViewContainer"] .stButton>button:hover{
        filter:brightness(1.05);
      }

      /* 이미지 여백 (겹침 방지) */
      [data-testid="stImage"]{ margin:6px 0 18px!important; }
      [data-testid="stImage"] img{ display:block; }

        span[label="app main"] {
      font-size: 0 !important;          /* 기존 글자 숨김 */
      position: relative;
  }
  span[label="app main"]::after {
      content: "메인";                  /* 원하는 표시 이름 */
      font-size: 1rem !important;       /* 기본 폰트 크기로 복원 */
      color: #fff !important;           /* 사이드바 글씨 색 (흰색) */
      font-weight: 700 !important;      /* 굵게 */
      position: absolute;
      left: 0;
      top: 0;
  }
    </style>
    """,
        unsafe_allow_html=True,
    )


# call once
_sidebar_dark_and_slider_fix()

# ---------------------------------------
# 환경설정
# ---------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY가 .env에 없습니다."

st.set_page_config(page_title="시방서 Q&A 챗봇", page_icon="🛁", layout="wide")
st.title("🛁 시방서 Q&A 챗봇")

# ---------------------------------------
# ✅ 상태 초기화 (세션 상태를 사용하기 전에!)
# ---------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ---------------------------------------
# 사이드바: 모델/옵션
# ---------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ 옵션")
    model_name = "gpt-5"
    st.markdown("⚙️ LLM 모델: gpt-5")
    k_ctx = st.slider("검색 문서 수(k)", 2, 8, 4, 1)
    chunk_size = st.slider("청크 크기", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("오버랩", 50, 400, 150, 25)
    st.markdown("---")
    st.markdown("**파일 업로드 후, [인덱스 생성]을 눌러주세요.**")


# ---------------------------------------
# 공용: 업로드 파일을 임시경로로 저장
# ---------------------------------------
def _save_uploaded_to_temp(uploaded_file, suffix):
    """Streamlit UploadedFile -> temp file path"""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(uploaded_file, tmp)  # .read()/getvalue() 대신 스트림 복사
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


# ---------------------------------------
# 함수: 문서 로딩 (PDF/Text 모두 임시경로 경유)
# ---------------------------------------
def load_docs(uploaded_files):
    docs = []
    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1].lower()

        if suffix == ".pdf":
            tmp_path = _save_uploaded_to_temp(f, ".pdf")
            try:
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                for d in loaded:
                    d.metadata["display_name"] = f.name
                docs.extend(loaded)
            finally:
                os.unlink(tmp_path)

        elif suffix in [".txt", ".md"]:
            tmp_path = _save_uploaded_to_temp(f, suffix)
            try:
                loader = TextLoader(tmp_path, encoding="utf-8")
                loaded = loader.load()
                for d in loaded:
                    d.metadata["display_name"] = f.name
                docs.extend(loaded)
            finally:
                os.unlink(tmp_path)

        else:
            st.warning(f"지원하지 않는 형식: {f.name}")

    return docs


# ---------------------------------------
# 함수: 청크 분할
# ---------------------------------------
def split_docs(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ---------------------------------------
# 시스템 프롬프트 (욕실 공사 시방서 전용)
# ---------------------------------------
SYSTEM_INSTRUCTIONS = """\
너는 욕실(UBR) 공사 시방서 전용 전문가 어시스턴트다.
- 반드시 업로드된 시방서/도면(컨텍스트)에 근거해 대답하라.
- 근거가 불충분하면 '해당사항 없음' 또는 '시방서에 명시 없음'이라고 답하고 추측하지 마라.
- 질문이 시방서 범위를 벗어나면 '본 챗봇은 시방서 기반 질의만 답변합니다'라고 안내하라.
- 수량이나 치수 계산이 필요한 경우, 문서 근거(페이지/문구)를 요약해서 함께 제시하라.
- 답변은 한국어로, 항목형/표형 정리 선호.
"""

USER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_INSTRUCTIONS),
        (
            "human",
            """\
다음은 검색된 시방서 컨텍스트입니다. 이를 참고하여 질문에 답하라.

[컨텍스트]
{context}

[대화 히스토리 요약]
{chat_history}

[질문]
{question}

요구사항:
- 문서 근거의 핵심 문구를 인용(요약)하고, 가능한 경우 페이지/섹션을 함께 제시.
- 모호하면 명시적으로 '해당사항 없음' 기재.
- 최종에 '요약' 섹션으로 3줄 이내 핵심만 재정리.
""",
        ),
    ]
)

# ---------------------------------------
# 업로더/인덱서
# ---------------------------------------
st.subheader("1) 시방서 업로드")
uploaded = st.file_uploader(
    "PDF(.pdf) 또는 텍스트(.txt/.md) 시방서를 업로드하세요 (복수 가능)",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("📚 인덱스 생성", use_container_width=True, type="primary"):
        if not uploaded:
            st.warning("먼저 파일을 업로드하세요.")
        else:
            with st.spinner("문서 로딩/청크 분할/임베딩 중..."):
                raw_docs = load_docs(uploaded)
                chunks = split_docs(
                    raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vs = FAISS.from_documents(chunks, embeddings)
                st.session_state["vectorstore"] = vs
            st.success(f"인덱스 생성 완료! (청크 수: {len(chunks)})")

with col_b:
    if st.button("🗑 인덱스 초기화", use_container_width=True):
        st.session_state["vectorstore"] = None
        st.session_state["chat_history"] = []
        st.success("초기화 완료.")


# ---------------------------------------
# RAG 체인 구성
# ---------------------------------------
def make_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": k_ctx, "fetch_k": max(10, k_ctx * 4)}
    )
    llm = ChatOpenAI(model=model_name)

    def format_docs(docs):
        formatted = []
        for d in docs:
            src_path = d.metadata.get("source", "")
            page = d.metadata.get("page", None)
            disp = d.metadata.get(
                "display_name", os.path.basename(src_path) or "document"
            )
            head = f"[source: {disp}"
            if page is not None:
                head += f", page: {page+1}"
            head += "]"
            formatted.append(f"{head}\n{d.page_content}")
        return "\n\n---\n\n".join(formatted)

    rag = (
        {
            # ✅ retriever에는 '질문' 문자열만 흘려보내기
            "context": (lambda x: x["question"]) | retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        | USER_PROMPT
        | llm
    )
    return rag, retriever


# ---------------------------------------
# 질의 영역
# ---------------------------------------
st.subheader("2) 질문하기")
q = st.text_input(
    "시방서 관련 질문을 입력하세요 (예: 'UBR 공사에서 벽체 타일 규격은?')"
)

if st.session_state["vectorstore"] is None:
    st.info("먼저 시방서를 업로드하고 인덱스를 생성하세요.")
else:
    rag_chain, retriever = make_rag_chain(st.session_state["vectorstore"])

    if st.button("🔎 질의 실행", type="primary") and q.strip():
        with st.spinner("검색 및 답변 생성 중..."):
            # Deprecated API 교체: get_relevant_documents -> invoke
            docs = retriever.invoke(q)

            # Runnable 바깥에서 안전하게 chat_history 문자열 생성
            chat_history_str = (
                "\n".join(
                    [
                        f"Q: {qq}\nA: {aa}"
                        for qq, aa in st.session_state["chat_history"]
                    ][-6:]
                )
                if st.session_state["chat_history"]
                else "없음"
            )

            # 체인에 명시적으로 입력 전달
            answer_msg = rag_chain.invoke(
                {"question": q, "chat_history": chat_history_str}
            )

        # 히스토리 저장
        st.session_state["chat_history"].append((q, answer_msg.content))

        # 출력
        st.markdown("### 🧠 답변")
        st.markdown(answer_msg.content)

        with st.expander("🔎 사용한 근거(상위 검색 결과) 보기"):
            for i, d in enumerate(docs, 1):
                src_path = d.metadata.get("source", "")
                page = d.metadata.get("page", None)
                disp = d.metadata.get(
                    "display_name", os.path.basename(src_path) or "document"
                )
                st.markdown(
                    f"**[{i}] {disp}**  (page: {page+1 if page is not None else 'N/A'})"
                )
                st.write(
                    d.page_content[:1200]
                    + ("..." if len(d.page_content) > 1200 else "")
                )

# ---------------------------------------
# 히스토리 표시
# ---------------------------------------
if st.session_state["chat_history"]:
    st.markdown("---")
    st.markdown("### 💬 대화 히스토리")
    for i, (qq, aa) in enumerate(reversed(st.session_state["chat_history"][-8:]), 1):
        st.markdown(f"**Q{i}.** {qq}")
        st.markdown(f"**A{i}.** {aa}")
