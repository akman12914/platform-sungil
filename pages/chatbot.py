import os
import tempfile
import shutil
import re
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

SEOUL_TZ = ZoneInfo("Asia/Seoul")

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
        --sb-bg:#0b1220;
        --sb-fg:#e2e8f0;
        --sb-muted:#e5e7eb;
        --sb-line:#1f2a44;

        --accent:#f1f5f9;
        --accent-2:#cbd5e1;

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

      /* helper labels */
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

      /* Slider cutoff fix */
      section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{ padding-right:12px; }
      section[data-testid="stSidebar"] div[data-testid="stSlider"]{
        padding-right:12px; margin-right:2px; overflow:visible;
      }
      section[data-testid="stSidebar"] div[role="slider"]{
        box-shadow:0 0 0 1px rgba(20,184,166,0.25); border-radius:999px;
      }

      input[type="radio"]{ accent-color: var(--accent); }
      div[role="radiogroup"] label{
        display:flex; align-items:center; gap:.5rem;
        line-height:1.2; margin: .1rem 0;
      }

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

      [data-testid="stImage"]{ margin:6px 0 18px!important; }
      [data-testid="stImage"] img{ display:block; }

      /* 사이드바 페이지 라벨 바꾸기 (예시) */
      span[label="app main"] { font-size:0 !important; position:relative; }
      span[label="app main"]::after {
        content:"메인"; font-size:1rem !important; color:#fff !important; font-weight:700 !important;
        position:absolute; left:0; top:0;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )


# call once
_sidebar_dark_and_slider_fix()

# 요약 카드 공통 스타일 (둥근 모서리 + 그림자)
st.markdown(
    """
<style>
  .summary-card{
    border:1px solid var(--line);
    border-radius:14px;
    padding:16px 20px;
    background:#ffffff;
    margin-top:.5rem;
    margin-bottom:3.5rem;
  }
  /* 카드 내부 elements 살짝 정돈 */
  .summary-card h1, .summary-card h2, .summary-card h3{ margin-top:.6rem; }
  .summary-card hr{ border:none; border-top:1px solid #e5e7eb; margin:12px 0; }
  .summary-card details{
    margin-top:.5rem;
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:10px;
    padding:.5rem .75rem;
  }
  .summary-card summary{ cursor:pointer; font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------
# 환경설정
# ---------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY가 .env에 없습니다."

st.set_page_config(page_title="시방서 Q&A 챗봇", page_icon="🛁", layout="wide")
st.title("🛁 시방서 Q&A 챗봇")

# ---------------------------------------
# ✅ 상태 초기화
# ---------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
# 새로 추가: 마지막 인덱스 배치와 그 요약
if "last_index_batch_docs" not in st.session_state:
    st.session_state["last_index_batch_docs"] = []
if "last_index_summary" not in st.session_state:
    st.session_state["last_index_summary"] = None

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
        shutil.copyfileobj(uploaded_file, tmp)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


# ---------------------------------------
# 함수: 문서 로딩 (PDF/Text)
# ---------------------------------------
def load_docs(uploaded_files):
    docs = []
    batch_id = datetime.now(tz=SEOUL_TZ).strftime("%Y%m%d-%H%M%S")
    batch_ts = datetime.now(tz=SEOUL_TZ).isoformat()

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
                    d.metadata["batch_id"] = batch_id
                    d.metadata["timestamp"] = batch_ts
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

# -------------------------------
# 🔴 요점(볼드/경고) 추출 유틸
# -------------------------------
HIGHLIGHT_PATTERNS = [
    r"\*\*(.+?)\*\*",  # **bold**
    r"(?:\(|\[|【)?\s*중요\s*(?:\)|\]|】)?[:：]?\s*(.+)",  # (중요) / [중요] / 중요: ...
    r"(?:\(|\[|【)?\s*주의\s*(?:\)|\]|】)?[:：]?\s*(.+)",  # (주의) ...
    r"※\s*(.+)",  # ※ ...
    r"(?:필수|엄수|경고)[:：]?\s*(.+)",  # 필수:, 경고:
    r"\bMUST\b[:：]?\s*(.+)",  # MUST: ...
]


def extract_highlights_from_text(text: str, limit=15):
    points = []
    # 1) 마크다운 bold 자체를 요점으로도 취급
    for m in re.finditer(r"\*\*(.+?)\*\*", text):
        t = m.group(1).strip()
        if 2 <= len(t) <= 120:  # 너무 짧거나 긴건 제외
            points.append(("bold", t))

    # 2) 중요/주의/※ 등
    for pat in HIGHLIGHT_PATTERNS[1:]:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            t = m.group(1).strip() if m.groups() else m.group(0).strip()
            if 2 <= len(t) <= 160:
                points.append(("red", t))

    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for typ, t in points:
        key = (typ, t)
        if key not in seen:
            seen.add(key)
            uniq.append((typ, t))
        if len(uniq) >= limit:
            break
    return uniq


def collect_batch_highlights(docs, per_doc_limit=6, total_limit=20):
    bag = []
    for d in docs:
        pts = extract_highlights_from_text(d.page_content, limit=per_doc_limit)
        bag.extend(pts)
        if len(bag) >= total_limit:
            break
    # total limit
    return bag[:total_limit]


# -------------------------------
# 🧾 요약 생성 (LLM)
# -------------------------------
SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 업로드된 시방서 묶음을 한국어로 간결하고 정확하게 요약하는 기술문서 보조자다. "
            "가능하면 조목조목 항목형으로, 수치/치수/재료/시공순서/검수기준을 구분해 정리하라. "
            "입력으로 전달되는 '요점 후보'는 굵게 강조해서 상단에 먼저 정리하라."
            "제목 마크다운 외 본문에 이모티콘은 사용하지 마라.",
        ),
        (
            "human",
            """다음은 이번 배치에 포함된 문서들의 발췌 텍스트다.

[요점 후보]
{points}

[문서 내용(샘플)]
{content}

원하는 출력 형식(마크다운):

- 문서 목록: 파일명1, 파일명2, ...

### 🔴 요점
- **굵게 표시** 항목으로 5~12개 핵심만.

---

### 📌 주요 사양
- <strong>재료</strong>:
- <strong>치수/규격</strong>:
- <strong>시공 절차/순서</strong>:
- <strong>품질/검수/유의</strong>:

---

### 📎 참고 근거
<details>
  <summary><b>🔎 근거 펼치기 / 접기</b></summary>

- [파일/페이지] 핵심문장 요약
- [파일/페이지] 핵심문장 요약
- (필요 시 추가)

</details>

---

### 요약
- 1)
- 2)
- 3)

---

주의: 문서에 없는 내용은 추측하지 말고 비워두거나 '해당사항 없음'으로 표기.
""",
        ),
    ]
)


def make_batch_summary(docs, model="gpt-5"):
    # 파일명 리스트
    names = []
    for d in docs:
        disp = (
            d.metadata.get("display_name")
            or os.path.basename(d.metadata.get("source", "") or "")
            or "document"
        )
        if disp not in names:
            names.append(disp)
    names_str = ", ".join(names[:12]) + (" ..." if len(names) > 12 else "")

    # 요점 후보 수집
    key_points = collect_batch_highlights(docs, per_doc_limit=6, total_limit=20)

    # 컨텐츠 샘플(너무 길면 앞부분만)
    samples = []
    for d in docs:
        t = d.page_content.strip().replace("\n\n", "\n")
        if not t:
            continue
        samples.append(t[:700])  # 샘플 길이 적당히 제한
    sample_text = "\n\n---\n\n".join(samples)[:4000]

    # 요점 후보를 마크다운/HTML 섞어서 미리 정리
    pts_lines = []
    for typ, t in key_points:
        pts_lines.append(
            f"- **{t}**" if typ == "bold" else f'- <span class="red-point">{t}</span>'
        )
    pts_block = "\n".join(pts_lines) if pts_lines else "- (자동 추출된 요점 없음)"

    # ✅ 파이프 체인으로 안전 호출
    llm = ChatOpenAI(model=model)
    summary_chain = SUMMARY_PROMPT | llm
    msg = summary_chain.invoke({"points": pts_block, "content": sample_text})

    rendered_inner = f"<h3>이번 배치 문서:{names_str}</h3>\n\n{msg.content}"
    rendered = f'<div class="summary-card">{rendered_inner}</div>'

    return rendered


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
                # 이번 업로드 배치만 별도로 로딩
                raw_docs = load_docs(uploaded)
                chunks = split_docs(
                    raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )

                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vs = FAISS.from_documents(chunks, embeddings)
                st.session_state["vectorstore"] = vs

                # 🔹 이번 배치를 저장(요약은 '이번 배치 우선' 생성)
                st.session_state["last_index_batch_docs"] = raw_docs

                # 🔹 바로 요약 생성
                st.session_state["last_index_summary"] = make_batch_summary(
                    raw_docs, model=model_name
                )

            st.success(f"인덱스 생성 완료! (청크 수: {len(chunks)})")

with col_b:
    if st.button("🗑 인덱스 초기화", use_container_width=True):
        st.session_state["vectorstore"] = None
        st.session_state["chat_history"] = []
        st.session_state["last_index_batch_docs"] = []
        st.session_state["last_index_summary"] = None
        st.success("초기화 완료.")

# ---------------------------------------
# ✅ 업로드 직후 요약본 출력 (새 인덱스 우선)
# ---------------------------------------
if st.session_state.get("last_index_summary"):
    st.markdown("### 업로드 배치 요약본")
    st.markdown(st.session_state["last_index_summary"], unsafe_allow_html=True)
    # 필요시 재생성(옵션 바꾼 후)
    # if st.button("🔁 요약 다시 생성", help="이번 업로드 배치 내용을 기준으로 재요약"):
    #     with st.spinner("요약 재생성 중..."):
    #         st.session_state["last_index_summary"] = make_batch_summary(
    #             st.session_state.get("last_index_batch_docs", []),
    #             model=model_name,
    #         )
    #     st.success("요약을 갱신했습니다.")
    #     st.markdown(st.session_state["last_index_summary"], unsafe_allow_html=True)


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
            docs = retriever.invoke(q)

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

            answer_msg = rag_chain.invoke(
                {"question": q, "chat_history": chat_history_str}
            )

        st.session_state["chat_history"].append((q, answer_msg.content))

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
