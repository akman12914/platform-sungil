import streamlit as st
from st_pages import Page, add_page_title, hide_pages
import auth


# 안전한 set_page_config
try:
    st.set_page_config(page_title="성일 통합 시스템", page_icon="⚙️", layout="wide")
except Exception:
    pass


hide_pages(["관리자", "로그인"])
# 로그인 체크

auth.require_auth()


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
      .hero h1{ margin:0 0 .5rem 0; color:var(--ink); font-size:1.8rem; font-weight:800; }
      .hero p{ margin:.25rem 0 0; color:var(--muted); font-size:1.05rem; }

      /* ========== 섹션 타이틀 스타일 ========== */
      h3 {
        color:var(--ink) !important;
        font-weight:700 !important;
        margin-top:2rem !important;
        margin-bottom:1rem !important;
        position:relative;
        padding-bottom:0.5rem;
      }
      h3::after {
        content:'';
        position:absolute;
        bottom:0;
        left:0;
        width:60px;
        height:3px;
        background:linear-gradient(90deg, var(--accent), var(--accent-2));
        border-radius:2px;
      }

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

  /* ========== 페이지 링크 카드 스타일 ========== */
  div[data-testid^="stPageLink"] > *,
  div[data-testid="stPageLink"] > *{
    display:block;
    border:1px solid var(--line) !important;
    border-radius:16px !important;
    padding:20px !important;
    background:#fff !important;
    box-shadow:0 2px 8px rgba(0,0,0,.08) !important;
    transition: all .2s ease !important;
    position:relative;
    overflow:hidden;
  }

  /* 왼쪽 컬러 액센트 바 */
  div[data-testid^="stPageLink"] > *::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    opacity: 0;
    transition: opacity 0.2s ease;
  }

  div[data-testid^="stPageLink"]:hover > *::before,
  div[data-testid="stPageLink"]:hover > *::before {
    opacity: 1;
  }

  div[data-testid^="stPageLink"] a,
  div[data-testid^="stPageLink"] p,
  div[data-testid="stPageLink"] a,
  div[data-testid="stPageLink"] p,
  div[data-testid^="stPageLink"] > *,
  div[data-testid="stPageLink"] > * {
    color:var(--ink) !important;
    white-space:pre-line !important;
    font-size:.95rem !important;
    margin:0 !important;
    padding-left:12px !important;
    cursor:pointer !important;
    text-decoration:none !important;
    line-height:1.6 !important;
  }

  div[data-testid^="stPageLink"]:hover > *,
  div[data-testid="stPageLink"]:hover > * {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 0, 0, .12) !important;
  }

  /* 첫 줄(타이틀) 강조 */
  div[data-testid^="stPageLink"] a::first-line,
  div[data-testid^="stPageLink"] p::first-line,
  div[data-testid="stPageLink"] a::first-line,
  div[data-testid="stPageLink"] p::first-line,
  div[data-testid^="stPageLink"] > *::first-line,
  div[data-testid="stPageLink"] > *::first-line {
    font-weight:800;
    font-size:1.15rem;
    letter-spacing: -0.02em;
  }

  /* 두 번째 줄(설명) 스타일 */
  div[data-testid^="stPageLink"] a,
  div[data-testid="stPageLink"] a {
    display:block !important;
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
  section[data-testid="stSidebarNav"] a[aria-label*="관리자"],
  section[data-testid="stSidebarNav"] a[aria-label*="로그인"] {
      display: none !important;
  }

    </style>
    """,
        unsafe_allow_html=True,
    )


_sidebar_dark_and_slider_fix()
# --- end reuse ---

# Sidebar: 사용자 정보 및 로그아웃
with st.sidebar:
    st.markdown("---")
    current_user = auth.get_current_user()
    user_info = auth.get_user_info(current_user)

    if user_info:
        st.markdown(f"**👤 {user_info['name']}**")
        role_text = "관리자" if user_info["role"] == "admin" else "사용자"
        st.caption(f"{role_text} • {user_info['username']}")

        if st.button("🚪 로그아웃", use_container_width=True):
            auth.logout()
            st.rerun()

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

# 섹션 제목
st.markdown("### 🔧 계산 도구")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Row 1: 바닥판 / 벽판(3단계) / 천장판
c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.page_link(
        "pages/1_바닥판_계산.py",
        label="🟦 바닥판 계산기\n바닥 규격 산출 및 미리보기",
        icon=None,
    )

with c2:
    # 벽판 계산 (3단계 파이프라인) - page_link 스타일과 통일
    st.markdown("""
    <div class="wall-panel-card">
        <div class="wall-panel-title">🟩 벽판 계산 (3단계)</div>
        <div class="wall-panel-desc">규격 → 타일 → 원가 순서로 진행</div>
        <div class="wall-panel-steps">
            <a href="/벽판_규격" target="_self" class="step-link">① 규격</a>
            <span class="step-arrow">→</span>
            <a href="/타일_개수" target="_self" class="step-link">② 타일</a>
            <span class="step-arrow">→</span>
            <a href="/벽판_원가" target="_self" class="step-link">③ 원가</a>
        </div>
    </div>
    <style>
    .wall-panel-card {
        border: 1px solid var(--line, #e2e8f0);
        border-radius: 16px;
        padding: 20px;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,.08);
        transition: all .2s ease;
        position: relative;
        overflow: hidden;
    }
    .wall-panel-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #22d3ee, #06b6d4);
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    .wall-panel-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,.12);
    }
    .wall-panel-card:hover::before {
        opacity: 1;
    }
    .wall-panel-title {
        font-weight: 800;
        font-size: 1.15rem;
        color: #0f172a;
        margin-bottom: 4px;
        padding-left: 12px;
    }
    .wall-panel-desc {
        font-size: 0.9rem;
        color: #475569;
        padding-left: 12px;
        margin-bottom: 12px;
    }
    .wall-panel-steps {
        display: flex;
        align-items: center;
        gap: 8px;
        padding-left: 12px;
    }
    .step-link {
        background: linear-gradient(180deg, #22d3ee, #06b6d4);
        color: #001018 !important;
        padding: 6px 12px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        text-decoration: none;
        transition: filter 0.15s ease;
    }
    .step-link:hover {
        filter: brightness(1.1);
        text-decoration: none;
    }
    .step-arrow {
        color: #94a3b8;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

with c3:
    st.page_link(
        "pages/5_천장판_계산.py",
        label="🟨 천장판 최적화\n패턴 전수/최소비용 조합",
        icon=None,
    )

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# 섹션 제목
st.markdown("### 📊 견적 및 지원")
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Row 2: Quotation & Chatbot (중앙 정렬을 위한 레이아웃)
sp_l, c4, c5, sp_r = st.columns([1, 3, 3, 1], gap="medium")

with c4:
    st.page_link(
        "pages/6_견적서_생성.py",
        label="📋 견적서 생성\n바닥/벽/천장 결과 종합",
        icon=None,
    )

with c5:
    st.page_link(
        "pages/0_chatbot.py",
        label="💬 시방서 Q&A\nAI 챗봇 (PDF 검색)",
        icon=None,
    )
