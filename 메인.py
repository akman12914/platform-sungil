import streamlit as st

# 안전한 set_page_config
try:
    st.set_page_config(page_title="성일 통합 시스템", page_icon="⚙️", layout="wide")
except Exception:
    pass


# --- reuse the sidebar dark/pro mood from other pages (paste once per app) ---
def _sidebar_dark_and_slider_fix():
    st.markdown(
        """
    <style>
      :root{ --sb-bg:#0b1220; --sb-fg:#e2e8f0; --sb-muted:#cbd5e1; --sb-line:#1f2a44;
             --accent:#22d3ee; --accent-2:#06b6d4; --ink:#0f172a; --muted:#475569; --line:#e2e8f0; }
      section[data-testid="stSidebar"]{ background:var(--sb-bg)!important; color:var(--sb-fg)!important; border-right:1px solid var(--sb-line); }
      section[data-testid="stSidebar"] *{ color:var(--sb-fg)!important; }
      section[data-testid="stSidebar"] .stMarkdown p, section[data-testid="stSidebar"] label{ color:var(--sb-muted)!important; font-weight:600!important; }
      [data-testid="stAppViewContainer"] .stButton>button{
        background:linear-gradient(180deg,var(--accent),var(--accent-2))!important; color:#001018!important;
        border:0!important; font-weight:800!important; letter-spacing:.2px;
      }
      [data-testid="stAppViewContainer"] .stButton>button:hover{ filter:brightness(1.05); }
      .hero{
        border:1px solid var(--line); border-radius:18px; padding:28px 26px; margin:12px 0 32px;
        background:linear-gradient(180deg,#f8fafc, #f1f5f9);
      }
      .hero h1{ margin:0 0 .5rem 0; color:var(--ink); font-size:1.6rem; }
      .hero p{ margin:.25rem 0 0; color:var(--muted); }
      .tile:hover{
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(0, 0, 0, .08) !important;   
      }
      .tile{
        border:1px solid var(--line); border-radius:16px; padding:18px; background:#fff;
        transition: transform .08s ease, box-shadow .2s ease;
        box-shadow:0 1px 3px rgba(0,0,0,.06);
      }
      .tile h3{ margin:.25rem 0 .5rem; font-size:1.05rem; color:#0f172a; }
      .tile p{ margin:0; color:#475569; font-size:.95rem; }
      .tile .cta{ margin-top:12px; }

  div[data-testid^="stPageLink"] > *,
  div[data-testid="stPageLink"] > *{
    display:block;
    border:1px solid var(--line) !important;
    border-radius:16px !important;
    padding:18px !important;
    background:#fff !important;
    box-shadow:0 1px 3px rgba(0,0,0,.06) !important;
    transition: transform .08s ease, box-shadow .2s ease !important;
  }
  div[data-testid^="stPageLink"] a,
  div[data-testid^="stPageLink"] p,
  div[data-testid="stPageLink"] a,
  div[data-testid="stPageLink"] p,
  div[data-testid^="stPageLink"] > *,
  div[data-testid="stPageLink"] > * {

    color:var(--ink) !important;
    white-space:pre-line !important;  /* \n 줄바꿈 반영 */
    font-size:.95rem !important;
    margin:0 !important;
    cursor:pointer !important;
    text-decoration:none !important;  /* a일 때 밑줄 제거 */
  }



    .stpagelink div[data-testid="stPageLink"]:hover > * {
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(0, 0, 0, .08) !important;
    }

  /* 첫 줄(타이틀) 강조 */
  div[data-testid^="stPageLink"] a::first-line,
  div[data-testid^="stPageLink"] p::first-line,
  div[data-testid="stPageLink"] a::first-line,
  div[data-testid="stPageLink"] p::first-line,
  div[data-testid^="stPageLink"] > *::first-line,
  div[data-testid="stPageLink"] > *::first-line {
    font-weight:800;
    font-size:1.08rem;
  }


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


_sidebar_dark_and_slider_fix()
# --- end reuse ---

# Hero
st.markdown(
    """
<div class="hero">
  <h1>성일 시스템</h1>
  <p>바닥/벽/천장 계산 도구로 바로 이동하세요.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Tiles
spL, c1, c2, c3, spR = st.columns([1, 3, 3, 3, 1])
with c1:
    st.page_link(
        "pages/바닥판_계산.py",
        label="🟦 바닥판 계산기\n바닥 규격 산출 및 미리보기",
        icon=None,
    )

with c2:
    st.page_link(
        "pages/벽판_계산.py",
        label="🟩 벽판 계산기\n문/젠다이/분할 규칙 반영",
        icon=None,
    )

with c3:
    st.page_link(
        "pages/천장판_계산.py",
        label="🟨 천장판 최적화\n패턴 전수/최소비용 조합",
        icon=None,
    )
