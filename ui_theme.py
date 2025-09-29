# ui_theme.py
import streamlit as st


def apply():
    """앱 전체에 공통 스타일 적용"""
    st.markdown(
        """
    <style>
      :root{
        --brand:#4F46E5;
        --ink:#0f172a;
        --muted:#f6f7fb;
        --border:#e5e7eb;
        --sub:#64748b;
      }
      .block-container{ padding-top: 2rem; }
      /* 사이드바 스타일 */
      [data-testid="stSidebar"]{
        background:#0f172a !important;
      }
      [data-testid="stSidebar"] *{
        color:#e5e7eb !important;
      }

      /* 버튼 & 링크 강조 */
      button[kind="primary"]{
        background:var(--brand) !important; border:none !important;
      }
      a, a:visited{ color:var(--brand); }

      /* 페이지 링크를 카드처럼 보이게 */
      [data-testid="stPageLink-NavLink"]{
        display:block; padding:14px 16px; border:1px solid var(--border);
        border-radius:14px; background:#fff; text-decoration:none !important;
        box-shadow:0 2px 0 rgba(2,6,23,0.03);
      }
      [data-testid="stPageLink-NavLink"]:hover{
        border-color:var(--brand);
        box-shadow:0 8px 24px rgba(79,70,229,.15);
      }

      /* 표/카드 기본 */
      .card{
        background:var(--muted); border:1px solid var(--border);
        border-radius:16px; padding:18px; margin-bottom:14px;
      }
      .pill{
        display:inline-block; padding:4px 10px; border-radius:999px;
        background:#EEF2FF; color:#3730a3; font-weight:600;
      }
      .muted{ color:var(--sub); }
      .hero{
        padding:22px; border-radius:20px;
        background:linear-gradient(135deg,#eef2ff,#f8fafc);
        border:1px solid var(--border);
        margin-bottom:12px;
      }
      .hero h1{ margin:8px 0 6px; }
    </style>
    """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str = ""):
    """상단 히어로 블록"""
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
