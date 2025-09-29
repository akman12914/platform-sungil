# ui_theme.py
import streamlit as st


def apply():
    st.markdown(
        """
    <style>
      :root{
        --brand:#4F46E5; --ink:#0f172a; --muted:#f6f7fb; --border:#e5e7eb; --sub:#64748b;
      }
      .block-container{ padding-top: 2rem; }

      /* 사이드바 */
      [data-testid="stSidebar"]{ background:#0f172a !important; }
      [data-testid="stSidebar"] *{ color:#e5e7eb !important; }

      /* 링크/버튼 */
      a, a:visited{ color:var(--brand); }
      button[kind="primary"]{ background:var(--brand) !important; border:none !important; }
      .muted{ color:var(--sub); }

      /* 히어로/배지 */
      .hero{
        padding:22px; border-radius:20px;
        background:linear-gradient(135deg,#eef2ff,#f8fafc);
        border:1px solid var(--border); margin-bottom:12px;
      }
      .hero h1{ margin:8px 0 6px; }
      .pill{
        display:inline-block; padding:4px 10px; border-radius:999px;
        background:#EEF2FF; color:#3730a3; font-weight:600;
      }

      /* KPI 카드 */
      .kpi{ display:flex; gap:16px; flex-wrap:wrap; }
      .kpi .item{
        background:#fff; border:1px solid var(--border);
        border-radius:14px; padding:12px 14px; min-width:180px;
        box-shadow:0 2px 0 rgba(2,6,23,0.03);
      }
      .kpi .item b{ display:block; font-size:22px; line-height:1.2; }

      .hr{ height:1px; background:var(--border); margin:12px 0; }

      /* 페이지 링크 카드 */
      [data-testid="stPageLink-NavLink"]{
        display:block; padding:14px 16px; border:1px solid var(--border);
        border-radius:14px; background:#fff; text-decoration:none !important;
        box-shadow:0 2px 0 rgba(2,6,23,0.03);
      }
      [data-testid="stPageLink-NavLink"]:hover{
        border-color:var(--brand);
        box-shadow:0 8px 24px rgba(79,70,229,.15);
      }

      /* ★ Expander를 카드처럼 — 최신/구버전 DOM 모두 커버 */
      /* (A) 최신 Streamlit: 컨테이너 div에 data-testid */
      [data-testid="stExpander"]{
        border:1px solid var(--border);
        border-radius:16px;
        background:var(--muted);
        box-shadow:0 2px 0 rgba(2,6,23,0.03);
        margin-bottom:14px;
      }
      [data-testid="stExpander"] details{ padding:10px 12px; }
      [data-testid="stExpander"] summary{ font-weight:700; color:var(--ink); }
      [data-testid="stExpander"] summary::-webkit-details-marker{ display:none; }

      /* (B) 구버전/환경별: details 자체를 직접 스타일 */
      details{
        border:1px solid var(--border);
        border-radius:16px;
        background:var(--muted);
        box-shadow:0 2px 0 rgba(2,6,23,0.03);
        margin-bottom:14px;
      }
      details > summary{
        list-style:none; padding:10px 12px; font-weight:700; color:var(--ink);
      }
      details > summary::-webkit-details-marker{ display:none; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = ""):
    st.markdown(
        f"""
    <div class="hero">
      <span class="pill">UBR</span>
      <h1>{title}</h1>
      <div class="muted">{subtitle}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def divider():
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)


def card(title: str, expanded: bool = True):
    return st.expander(title, expanded=expanded)
