"""
로그인 페이지
사용자 인증 및 로그인/로그아웃 기능
"""

import streamlit as st
from common_styles import apply_common_styles, set_page_config
import auth

set_page_config(page_title="로그인", layout="centered")
apply_common_styles()

st.title("🔐 로그인")

# 이미 로그인된 경우
if auth.is_authenticated():
    current_user = auth.get_current_user()
    user_info = auth.get_user_info(current_user)

    st.success(f"✅ {user_info['name']}님, 환영합니다!")

    # 현재 사용자 정보 표시
    st.markdown("### 현재 로그인 정보")
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**사용자 ID:** {user_info['username']}")
    with col2:
        role_text = "관리자" if user_info['role'] == "admin" else "일반 사용자"
        st.info(f"**권한:** {role_text}")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # 로그아웃 버튼
    col_spacer, col_btn, col_spacer2 = st.columns([1, 2, 1])
    with col_btn:
        if st.button("🚪 로그아웃", use_container_width=True):
            auth.logout()
            st.rerun()

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # 메인 페이지로 이동 버튼
    col_spacer3, col_btn2, col_spacer4 = st.columns([1, 2, 1])
    with col_btn2:
        st.page_link("메인.py", label="🏠 메인 페이지로 이동", icon=None)

else:
    # 로그인 폼
    st.markdown(
        """
    <div style="
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    ">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
            <span style="font-size: 24px;">👤</span>
            <h3 style="margin: 0; color: #0f172a; font-weight: 700;">로그인</h3>
        </div>
        <p style="margin: 0 0 12px 36px; color: #475569; line-height: 1.6;">
            성일 시스템을 사용하려면 로그인이 필요합니다.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.form("login_form"):
        username = st.text_input("사용자 ID", placeholder="아이디를 입력하세요")
        password = st.text_input("비밀번호", type="password", placeholder="비밀번호를 입력하세요")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            submit = st.form_submit_button("🔐 로그인", use_container_width=True)

        if submit:
            if not username or not password:
                st.error("❌ 사용자 ID와 비밀번호를 모두 입력해주세요.")
            else:
                # 로그인 시도
                if auth.login(username, password):
                    st.success("✅ 로그인 성공!")
                    st.rerun()
                else:
                    st.error("❌ 사용자 ID 또는 비밀번호가 올바르지 않습니다.")

    # 안내 메시지
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    st.info("💡 **안내:** 초기 관리자 계정은 ID: `admin`, 비밀번호: `admin123` 입니다. 로그인 후 관리자 페이지에서 비밀번호를 변경하세요.")
