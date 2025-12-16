from common_styles import apply_common_styles, set_page_config
import auth

import os
import tempfile
import shutil
import re
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import math
from typing import List, Tuple


SEOUL_TZ = ZoneInfo("Asia/Seoul")


# === [ADD] 여러 형태의 섹션 헤더 줄바꿈 보정 ===
HEADER_MARKERS = ("※", "■", "◆", "●", "▶", "▷", "▲", "▸", "•")
BULLET_MARKERS = ("- ", "• ", "ㆍ", "·", "* ", "— ", "– ")

HEADER_KEYWORDS = (
    "개요",
    "공사 범위",
    "공사범위",
    "견적조건",
    "적용범위",
    "UBR 공사분",
    "재료",
    "자재",
    "치수",
    "규격",
    "시공 절차",
    "시공절차",
    "품질",
    "검수",
    "유의",
    "유의사항",
    "안전",
    "기타",
    "참고",
    "근거",
)

_hdr_colon_re = re.compile(r"^\s*[^:：]{1,80}\s*[:：]\s*$")
_hdr_number_re = re.compile(r"^\s*(제?\d+(?:\.\d+)*[)\.]?)\s+[^\s].{0,60}$")
_hdr_keyword_re = re.compile(
    r"^\s*(" + "|".join(map(re.escape, HEADER_KEYWORDS)) + r")\s*$"
)


def _is_header_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    if s.startswith(HEADER_MARKERS):  # 기호형 헤더
        return True
    if _hdr_colon_re.match(s):  # 콜론형 헤더 (예: "재료:")
        return True
    if _hdr_number_re.match(s):  # 번호형 헤더 (예: "1. 개요", "제2. 품질")
        return True
    if _hdr_keyword_re.match(s):  # 키워드 단독 헤더 (예: "공사 범위")
        return True
    return False


def _is_bullet_line(s: str) -> bool:
    ss = s.lstrip()
    return any(ss.startswith(m) for m in BULLET_MARKERS)


def _normalize_multiline_sections_enhanced(text: str) -> str:
    """
    PDF 추출 과정에서 헤더가 줄바꿈으로 끊긴 것을 보정.
    - 헤더 라인 인식(기호/콜론/번호/키워드)
    - 바로 다음 라인이 불릿/헤더/빈줄이 아니고 너무 길지 않으면(<=120자) 최대 3줄까지 이어붙임.
    """
    lines = text.splitlines()
    out = []
    buf = None
    tail_joined = 0

    for raw in lines:
        s = raw.strip()

        if _is_header_line(s):
            if buf is not None:
                out.append(buf)
            buf = s
            tail_joined = 0
            continue

        if buf is not None:
            if (
                s
                and not _is_header_line(s)
                and not _is_bullet_line(s)
                and len(s) <= 120
                and tail_joined < 3
            ):
                buf += " " + s
                tail_joined += 1
                continue
            else:
                out.append(buf)
                buf = None
                tail_joined = 0

        out.append(raw)

    if buf is not None:
        out.append(buf)

    return "\n".join(out)


# === [/ADD] ===


# LangChain (최신 구조)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ---------------------------------------
# 환경설정
# ---------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY가 .env에 없습니다."

set_page_config(page_title="시방서 Q&A 챗봇", page_icon="🛁", layout="wide")
apply_common_styles()

auth.require_auth()

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
    model_name = "gpt-5-mini"
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
# 함수: 최신우선 가중치(신선도) + 유사도 재랭킹 검색
# ---------------------------------------


def _parse_ts(ts: str) -> float:
    # ISO8601 → epoch seconds
    try:
        return datetime.fromisoformat(ts).timestamp()
    except Exception:
        return 0.0


def search_with_recency_rerank(
    vs,
    query: str,
    k: int = 4,
    fetch_k: int = 32,
    w_recency: float = 0.35,
    half_life_days: float = 14.0,
) -> List[Document]:
    """
    벡터 유사도 + 신선도(지수감쇠) 결합 점수로 재랭크.
    FAISS.similarity_search_with_score 를 사용하고, 점수정규화 후 결합.
    """
    # 1) 충분히 넓게 후보 수집
    try:
        pairs: List[Tuple[Document, float]] = vs.similarity_search_with_score(
            query, k=fetch_k
        )
        # 일부 구현은 score가 "작을수록 유사"(거리)일 수 있으므로 뒤에서 정규화로 보정
    except Exception:
        # fallback
        docs = vs.similarity_search(query, k=fetch_k)
        pairs = [(d, 0.0) for d in docs]

    now = datetime.now(tz=SEOUL_TZ).timestamp()
    # 2) score 정규화 (min-max → 유사도 방향으로 뒤집기)
    scores = [s for _, s in pairs]
    if scores:
        s_min, s_max = min(scores), max(scores)
        # 거리를 유사도로 변환: 작은게 더 유사 → inv_norm
        sim_norm = []
        for doc, s in pairs:
            if s_max == s_min:
                inv = 1.0
            else:
                # 0~1로 정규화 후 뒤집기
                inv = 1.0 - ((s - s_min) / (s_max - s_min))
            sim_norm.append((doc, inv))
    else:
        sim_norm = [(doc, 1.0) for doc, _ in pairs]

    # 3) recency 점수: half-life 기반 지수 감쇠
    hl_secs = half_life_days * 86400.0
    ranked = []
    for doc, sim in sim_norm:
        ts = _parse_ts(doc.metadata.get("timestamp", ""))  # epoch
        # 시간이 없으면 0점
        if ts <= 0:
            rec = 0.0
        else:
            age = max(0.0, now - ts)
            rec = math.exp(-age / hl_secs)  # 최근일수록 1에 가까움

        combined = (1.0 - w_recency) * sim + (w_recency) * rec
        ranked.append((combined, doc, sim, rec))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d, _, _ in ranked[:k]]


# ---------------------------------------
# 함수: 문서 로딩 (PDF/Text)
# ---------------------------------------
def load_docs(uploaded_files):
    docs = []
    batch_id = datetime.now(tz=SEOUL_TZ).strftime("%Y%m%d-%H%M%S")
    base_ts = datetime.now(tz=SEOUL_TZ)
    step = 1  # 파일 간 1초 간격

    for idx, f in enumerate(uploaded_files):
        suffix = os.path.splitext(f.name)[1].lower()
        file_ts = (base_ts - timedelta(seconds=step * idx)).isoformat()

        if suffix == ".pdf":
            tmp_path = _save_uploaded_to_temp(f, ".pdf")
            try:
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                for d in loaded:
                    d.metadata["display_name"] = f.name
                    d.metadata["batch_id"] = batch_id
                    d.metadata["timestamp"] = file_ts
                    d.page_content = _normalize_multiline_sections_enhanced(
                        d.page_content
                    )
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
                    d.metadata["timestamp"] = file_ts
                    d.page_content = _normalize_multiline_sections_enhanced(
                        d.page_content
                    )
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

요약/병합 규칙:
- 동일 항목에서 서로 다른 값이 있으면 **문서 메타데이터의 timestamp가 가장 최근인 값만** 채택한다.
- v1/v2 같은 **버전 라벨을 본문에 쓰지 말라**. 과거값은 '참고 근거'에만 필요시 요약-비교하라.
- 즉, 최종 본문은 **최신 기준으로 병합된 단일 사양**만 적는다.


원하는 출력 형식(마크다운):

- 문서 목록: 파일명1, 파일명2, ...

### 🔴 요점
- **굵게 표시** 항목으로 5~12개 핵심만.

---

### 📌 주요 사양
- **재료:**
- **치수/규격:**
- **시공 절차/순서:**
- **품질/검수/유의:**

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


def make_batch_summary(docs, model="gpt-5-mini"):
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
# ✅ 모순(충돌) 감지/병합 규칙
# ---------------------------------------

# ---- 간단 규칙 기반 추출기 (숫자/부등호/단위 & 긍/부정 서술)
NUM_PAT = re.compile(
    r"(?P<key>[가-힣A-Za-z0-9\s\-/\(\)·]+?)\s*"
    r"(?P<op>≥|<=|≤|>=|=|>|<|≈|~)?\s*"
    r"(?P<val>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>mm|cm|m|W|kW|%|EA|MPa|CMH|A|V|mmH2O|dB\(A\))?",
    flags=re.UNICODE,
)
NEG_PAT = re.compile(r"(금지|무|아님|아니다|없음|불가)")
POS_PAT = re.compile(r"(필수|포함|설치|적용|필요|있음)")


def _normalize_key(raw: str) -> str:
    t = re.sub(r"[\s/()·]+", " ", raw).strip().lower()
    # 너무 긴 키는 컷
    return t[:120]


def extract_facts(doc) -> list[dict]:
    facts = []
    text = doc.page_content
    for m in NUM_PAT.finditer(text):
        facts.append(
            {
                "type": "numeric",
                "key": _normalize_key(m.group("key")),
                "op": m.group("op") or "=",
                "val": float(m.group("val")),
                "unit": (m.group("unit") or "").lower(),
                "source": doc.metadata.get("display_name", "document"),
                "page": doc.metadata.get("page"),
                "ts": doc.metadata.get("timestamp"),
            }
        )
    # 서술형 (+/-) 존재성
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        key = _normalize_key(line)
        if POS_PAT.search(line):
            facts.append(
                {
                    "type": "bool",
                    "key": key,
                    "polarity": True,
                    "source": doc.metadata.get("display_name", "document"),
                    "page": doc.metadata.get("page"),
                    "ts": doc.metadata.get("timestamp"),
                }
            )
        if NEG_PAT.search(line):
            facts.append(
                {
                    "type": "bool",
                    "key": key,
                    "polarity": False,
                    "source": doc.metadata.get("display_name", "document"),
                    "page": doc.metadata.get("page"),
                    "ts": doc.metadata.get("timestamp"),
                }
            )
    return facts


def detect_conflicts(docs: list) -> dict:
    """
    반환:
    {
      "numeric_conflicts": [ {key, entries:[...], merged} ],
      "boolean_conflicts": [ {key, positives:[...], negatives:[...], resolution} ],
      "constraint_violations": [ {rule, evidence:[...]} ]
    }
    병합 규칙:
      - 최신(timestamp 큰) 값을 우선
      - 단위 동일 시 값이 다르면 '충돌'
      - 부등호/조건 충돌도 표기
    """
    by_key_num = {}
    by_key_bool = {}

    for d in docs:
        for f in extract_facts(d):
            if f["type"] == "numeric":
                by_key_num.setdefault((f["key"], f["unit"]), []).append(f)
            else:
                by_key_bool.setdefault(f["key"], []).append(f)

    numeric_conflicts = []
    for (key, unit), items in by_key_num.items():
        # 서로 다른 값/연산자가 존재하면 충돌 후보
        vals = {(it["op"], it["val"]) for it in items}
        if len(vals) > 1:
            # 최신 우선 병합안: 가장 최신 ts
            items_sorted = sorted(items, key=lambda x: (x["ts"] or "",), reverse=True)
            merged = {
                "op": items_sorted[0]["op"],
                "val": items_sorted[0]["val"],
                "unit": unit,
                "ts": items_sorted[0]["ts"],
                "source": items_sorted[0]["source"],
            }
            numeric_conflicts.append(
                {"key": key, "unit": unit, "entries": items_sorted, "merged": merged}
            )

    boolean_conflicts = []
    for key, items in by_key_bool.items():
        pos = [it for it in items if it["polarity"]]
        neg = [it for it in items if not it["polarity"]]
        if pos and neg:
            # 최신 우선: 더 최신 쪽 채택
            newest_pos_ts = max((p["ts"] or "" for p in pos), default="")
            newest_neg_ts = max((n["ts"] or "" for n in neg), default="")
            resolution = True if newest_pos_ts >= newest_neg_ts else False
            boolean_conflicts.append(
                {
                    "key": key,
                    "positives": pos,
                    "negatives": neg,
                    "resolution": resolution,  # True 채택/ False 채택
                }
            )

    # 제약 위반: 간단 규칙 예) "A < B"인데 "= B" 등장
    # 텍스트 기반이라 키 매핑이 어려워 보수적으로 탐지
    constraint_violations = []
    # 예시 규칙: 같은 key/unit에서 (< 또는 ≤) vs (= 또는 >, ≥)가 공존하고 값이 동일/역전
    for (key, unit), items in by_key_num.items():
        ops = set(it["op"] for it in items)
        if any(op in ops for op in ["<", "≤"]) and any(
            op in ops for op in ["=", ">", "≥"]
        ):
            # 간단: 값들의 min/max가 서로 모순인지 체크
            vals = [it["val"] for it in items]
            if vals:
                mn, mx = min(vals), max(vals)
                if mn == mx or mn > mx:
                    constraint_violations.append(
                        {
                            "rule": f"{key} 제약 충돌({unit}): '< or ≤' 와 '= or > or ≥' 혼재",
                            "evidence": items,
                        }
                    )

    return {
        "numeric_conflicts": numeric_conflicts,
        "boolean_conflicts": boolean_conflicts,
        "constraint_violations": constraint_violations,
    }


# ---------------------------------------
# ✅ 업로드 직후 요약본 출력 (새 인덱스 우선)
# ---------------------------------------
if st.session_state.get("last_index_summary"):
    st.markdown("### 업로드 배치 요약본")
    st.markdown(st.session_state["last_index_summary"], unsafe_allow_html=True)

# conflicts = detect_conflicts(st.session_state["last_index_batch_docs"])
# st.session_state["last_batch_conflicts"] = conflicts

# if st.session_state.get("last_batch_conflicts"):
#     cf = st.session_state["last_batch_conflicts"]
#     st.markdown("#### 🧩 문서 충돌/모순 감지 결과")
#     with st.expander("🔎 상세 보기 (수치/서술/제약 위반)"):
#         # 수치형
#         st.markdown("**수치형 충돌 (numeric)**")
#         if cf["numeric_conflicts"]:
#             for c in cf["numeric_conflicts"]:
#                 st.write(f"- 키: `{c['key']}` [{c['unit'] or '-'}]")
#                 for e in c["entries"]:
#                     page = (e["page"] + 1) if isinstance(e["page"], int) else "N/A"
#                     st.write(
#                         f"   • {e['source']} p.{page}: {e['op']} {e['val']} {e['unit'] or ''} @ {e['ts']}"
#                     )
#                 m = c["merged"]
#                 st.write(
#                     f"   → **병합 권고(최신우선)**: {m['op']} {m['val']} {m['unit'] or ''} (from {m['source']}, {m['ts']})"
#                 )
#         else:
#             st.write("- 없음")

#         st.markdown("---")
#         # 서술형
#         st.markdown("**서술/범주 충돌 (boolean)**")
#         if cf["boolean_conflicts"]:
#             for c in cf["boolean_conflicts"]:
#                 st.write(f"- 키: `{c['key']}`")
#                 st.write(
#                     "  • 긍정 근거 수: "
#                     + str(len(c["positives"]))
#                     + " / 부정 근거 수: "
#                     + str(len(c["negatives"]))
#                 )
#                 st.write(
#                     f"  → **채택(최신우선)**: {'긍정' if c['resolution'] else '부정'}"
#                 )
#         else:
#             st.write("- 없음")

#         st.markdown("---")
#         # 제약 위반
#         st.markdown("**제약 위반 (constraints)**")
#         if cf["constraint_violations"]:
#             for v in cf["constraint_violations"]:
#                 st.write(f"- {v['rule']}")
#                 for e in v["evidence"]:
#                     page = (e["page"] + 1) if isinstance(e["page"], int) else "N/A"
#                     st.write(
#                         f"   • {e['source']} p.{page}: {e['op']} {e['val']} {e['unit'] or ''} @ {e['ts']}"
#                     )
#         else:
#             st.write("- 없음")


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
            docs = search_with_recency_rerank(
                st.session_state["vectorstore"],
                q,
                k=k_ctx,
                fetch_k=max(24, k_ctx * 6),
                w_recency=0.35,
                half_life_days=14,
            )
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
