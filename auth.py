"""
auth.py
사용자 인증 및 관리 모듈
- 로그인/로그아웃 기능
- 사용자 데이터 관리 (JSON 파일 기반)
- 비밀번호 해싱 (SHA256)
"""

import json
import hashlib
import os
from typing import Optional, Dict, List
from datetime import datetime
import streamlit as st

# 사용자 데이터 파일 경로
USERS_FILE = "users.json"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # 초기 비밀번호

# Session state keys
SESSION_USER_KEY = "authenticated_user"
SESSION_ROLE_KEY = "user_role"


def _hash_password(password: str) -> str:
    """비밀번호를 SHA256으로 해싱"""
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> Dict[str, Dict]:
    """사용자 데이터 로드 (users.json)"""
    if not os.path.exists(USERS_FILE):
        # 파일이 없으면 기본 관리자 계정 생성
        default_users = {
            DEFAULT_ADMIN_USERNAME: {
                "password": _hash_password(DEFAULT_ADMIN_PASSWORD),
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "name": "관리자"
            }
        }
        _save_users(default_users)
        return default_users

    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"사용자 데이터 로드 실패: {e}")
        return {}


def _save_users(users: Dict[str, Dict]) -> None:
    """사용자 데이터 저장"""
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"사용자 데이터 저장 실패: {e}")


def authenticate(username: str, password: str) -> Optional[Dict]:
    """
    사용자 인증

    Args:
        username: 사용자 ID
        password: 비밀번호

    Returns:
        인증 성공 시 사용자 정보, 실패 시 None
    """
    users = _load_users()

    if username not in users:
        return None

    user_data = users[username]
    password_hash = _hash_password(password)

    if user_data["password"] == password_hash:
        return {
            "username": username,
            "role": user_data.get("role", "user"),
            "name": user_data.get("name", username)
        }

    return None


def login(username: str, password: str) -> bool:
    """
    로그인 처리

    Args:
        username: 사용자 ID
        password: 비밀번호

    Returns:
        로그인 성공 여부
    """
    user_info = authenticate(username, password)

    if user_info:
        st.session_state[SESSION_USER_KEY] = user_info["username"]
        st.session_state[SESSION_ROLE_KEY] = user_info["role"]
        return True

    return False


def logout() -> None:
    """로그아웃 처리"""
    if SESSION_USER_KEY in st.session_state:
        del st.session_state[SESSION_USER_KEY]
    if SESSION_ROLE_KEY in st.session_state:
        del st.session_state[SESSION_ROLE_KEY]


def is_authenticated() -> bool:
    """현재 사용자가 로그인되어 있는지 확인"""
    return SESSION_USER_KEY in st.session_state


def get_current_user() -> Optional[str]:
    """현재 로그인한 사용자 ID 반환"""
    return st.session_state.get(SESSION_USER_KEY)


def get_current_role() -> Optional[str]:
    """현재 로그인한 사용자의 역할 반환"""
    return st.session_state.get(SESSION_ROLE_KEY)


def is_admin() -> bool:
    """현재 사용자가 관리자인지 확인"""
    return get_current_role() == "admin"


def require_auth():
    """
    로그인이 필요한 페이지에서 호출
    로그인되지 않은 경우 로그인 페이지로 안내하고 페이지 실행 중단
    """
    if not is_authenticated():
        st.warning("⚠️ 이 페이지에 접근하려면 로그인이 필요합니다.")

        st.markdown(
            """
        <div style="
            border: 1px solid #f59e0b;
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        ">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 24px;">🔐</span>
                <h3 style="margin: 0; color: #0f172a; font-weight: 700;">로그인 필요</h3>
            </div>
            <p style="margin: 0 0 12px 36px; color: #78350f; line-height: 1.6;">
                성일 시스템을 사용하려면 로그인이 필요합니다.
            </p>
            <div style="margin-left: 36px; padding: 12px; background: white; border-radius: 8px; border: 1px solid #f59e0b;">
                <p style="margin: 0; color: #92400e; font-size: 0.95rem; line-height: 1.6;">
                    아래 버튼을 클릭하여 로그인 페이지로 이동하세요.
                </p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_spacer, col_btn, col_spacer2 = st.columns([1, 2, 1])
        with col_btn:
            st.page_link("pages/로그인.py", label="🔐 로그인 페이지로 이동", icon=None)

        st.stop()


def require_admin():
    """
    관리자 권한이 필요한 페이지에서 호출
    관리자가 아닌 경우 에러 메시지 표시하고 페이지 실행 중단
    """
    require_auth()  # 먼저 로그인 확인

    if not is_admin():
        st.error("❌ 이 페이지는 관리자만 접근할 수 있습니다.")
        st.stop()


# 사용자 관리 함수 (관리자 전용)
def add_user(username: str, password: str, name: str, role: str = "user") -> bool:
    """
    새 사용자 추가 (관리자 전용)

    Args:
        username: 사용자 ID
        password: 비밀번호
        name: 사용자 이름
        role: 역할 (admin/user)

    Returns:
        성공 여부
    """
    users = _load_users()

    if username in users:
        return False  # 이미 존재하는 사용자

    users[username] = {
        "password": _hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "name": name
    }

    _save_users(users)
    return True


def delete_user(username: str) -> bool:
    """
    사용자 삭제 (관리자 전용)

    Args:
        username: 사용자 ID

    Returns:
        성공 여부
    """
    users = _load_users()

    if username not in users:
        return False

    # 관리자 계정 삭제 방지
    if users[username].get("role") == "admin" and username == DEFAULT_ADMIN_USERNAME:
        return False

    del users[username]
    _save_users(users)
    return True


def update_password(username: str, new_password: str) -> bool:
    """
    비밀번호 변경

    Args:
        username: 사용자 ID
        new_password: 새 비밀번호

    Returns:
        성공 여부
    """
    users = _load_users()

    if username not in users:
        return False

    users[username]["password"] = _hash_password(new_password)
    users[username]["updated_at"] = datetime.now().isoformat()

    _save_users(users)
    return True


def get_all_users() -> List[Dict]:
    """
    모든 사용자 목록 반환 (관리자 전용)

    Returns:
        사용자 정보 리스트 (비밀번호 제외)
    """
    users = _load_users()

    user_list = []
    for username, data in users.items():
        user_list.append({
            "username": username,
            "name": data.get("name", username),
            "role": data.get("role", "user"),
            "created_at": data.get("created_at", ""),
        })

    return user_list


def get_user_info(username: str) -> Optional[Dict]:
    """
    특정 사용자 정보 반환 (비밀번호 제외)

    Args:
        username: 사용자 ID

    Returns:
        사용자 정보 또는 None
    """
    users = _load_users()

    if username not in users:
        return None

    data = users[username]
    return {
        "username": username,
        "name": data.get("name", username),
        "role": data.get("role", "user"),
        "created_at": data.get("created_at", ""),
    }
