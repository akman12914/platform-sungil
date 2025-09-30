# streamlit run app.py
import io
from typing import Optional, Dict, Any


# --- design refresh (prettier inline) ---
import streamlit as st

# --- Floor/Walls 연동용 상태키 ---
FLOOR_DONE_KEY = "floor_done"
FLOOR_RESULT_KEY = "floor_result"


def _init_state():
    st.session_state.setdefault(FLOOR_DONE_KEY, False)
    st.session_state.setdefault(FLOOR_RESULT_KEY, None)


_init_state()


def _design_refresh():

    st.markdown(
        """
    <style>
      :root{
        /* Sidebar dark palette */
        --sb-bg:#0b1220;         /* 다크 네이비 */
        --sb-fg:#e2e8f0;         /* 본문 텍스트 */
        --sb-muted:#475569;      /* 🔸보조 텍스트: 더 밝게/진하게 */
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
        color:var(--sb-muted)!important;
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
        color:#0f172a !important;
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

        /* NumberInput - stepper 버튼 아이콘 색상 */
      button[data-testid="stNumberInputStepUp"] svg,
      button[data-testid="stNumberInputStepDown"] svg {
          color: var(--sb-muted) !important;   /* 보조색 */
          fill: var(--sb-muted) !important;    /* 일부 환경에서 필요 */
      }

      /* 버튼 자체 hover/focus 시에도 색 유지 */
      button[data-testid="stNumberInputStepUp"]:hover svg,
      button[data-testid="stNumberInputStepDown"]:hover svg {
          color: var(--sb-muted) !important;
          fill: var(--sb-muted) !important;
      }

            /* Selectbox: 선택된 값 텍스트 */
      div[data-baseweb="select"] div[role="combobox"],
      div[data-baseweb="select"] div[role="combobox"] input,
      div[data-baseweb="select"] div[value] {
          color: var(--sb-muted) !important;   /* 보조색 */
          font-weight: 600 !important;         /* 조금 더 굵게 */
      }

      /* Selectbox: 드롭다운 아이콘 (열림/닫힘 화살표) */
      div[data-baseweb="select"] svg {
          color: var(--sb-muted) !important;
          fill: var(--sb-muted) !important;
      }

      /* Hover 시에도 색 유지 */
      div[data-baseweb="select"]:hover div[value],
      div[data-baseweb="select"]:hover svg {
          color: var(--sb-muted) !important;
          fill: var(--sb-muted) !important;
      }

            /* 🔹 FileUploader 전체 영역 */
      section[data-testid="stFileUploaderDropzone"] {
          border: 2px dashed var(--sb-line) !important;
          background: rgba(255,255,255,0.03) !important;
          color: var(--sb-muted) !important;
          border-radius: 10px !important;
          padding: 12px !important;
      }

      /* 아이콘 색상 */
      section[data-testid="stFileUploaderDropzone"] svg {
          color: var(--sb-muted) !important;
          fill: var(--sb-muted) !important;
      }

      /* 안내 텍스트 */
      section[data-testid="stFileUploaderDropzone"] span {
          color: var(--sb-muted) !important;
          font-weight: 600 !important;
      }

      /* 버튼 */
      section[data-testid="stFileUploaderDropzone"] button {
          background: linear-gradient(180deg,var(--accent),var(--accent-2)) !important;
          color: #001018 !important;
          border: 0 !important;
          font-weight: 700 !important;
          border-radius: 8px !important;
          padding: .4rem .9rem !important;
      }
      section[data-testid="stFileUploaderDropzone"] button:hover {
          filter: brightness(1.05);
      }

            /* 계산하기 버튼 텍스트 색 변경 */
      button[data-testid="stBaseButton-primary"] p {
          color: var(--ink) !important;  /* 보조색 계열 */
          font-weight: 700 !important;        /* 더 굵게 */
      }

     div[data-testid="stImageContainer"] {
          margin-bottom: 2rem !important; /* 이미지+캡션 아래쪽 간격 */
      }

      /* 모든 stImage(도형, 미리보기 등) 출력은 강제로 블록 배치 */
      div[data-testid="stImage"] {
          display: block !important;
          width: 100% !important;          /* 한 줄 전용 */
          margin: 2rem auto !important;    /* 위/아래 넉넉히 띄움 */
          text-align: center !important;   /* 중앙 정렬 */
          z-index: 1 !important;           /* 텍스트보다 위 */
          position: relative !important;   /* 겹침 방지 */
      }

      /* 이미지와 캡션 간격 */
      div[data-testid="stImageCaption"] {
          margin-top: 1rem !important;
      }

    /* 사이드바 Alert 전용 스타일 */
    section[data-testid="stSidebar"] div[data-testid="stAlertContainer"] {
        background: transparent !important;   /* 배경 무채색 */
        border: 1px solid #555 !important;
        color: #e2e2e2 !important;        /* 텍스트 색 */
        border-radius: 6px !important;
        padding: 0.6rem !important;
    }

    /* 내부 텍스트/아이콘은 상위 색을 따르도록 */
    section[data-testid="stSidebar"] div[data-testid="stAlertContainer"] * {
        color: inherit !important;
        fill: inherit !important;
    }

    /* 아이콘만 살짝 연한 회색 */
    section[data-testid="stSidebar"] div[data-testid="stAlertContainer"] svg {
        color: #bbb !important;
        fill: #bbb !important;
    }

    </style>

    """,
        unsafe_allow_html=True,
    )


# --- end design refresh ---

_design_refresh()


import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# ---------------------------
# UI: Sidebar (왼쪽 입력 인터페이스)
# ---------------------------
st.set_page_config(page_title="바닥판 규격/옵션 산출", layout="wide")

st.sidebar.header("입력값 (왼쪽 인터페이스)")
uploaded = st.sidebar.file_uploader(
    "엑셀 업로드 (시트명: 바닥판)", type=["xlsx", "xls"]
)

units = st.sidebar.number_input("공사 세대수", min_value=1, step=1, value=100)

st.sidebar.subheader("기본 조건")
central = st.sidebar.radio("중앙배수 여부", ["No", "Yes"], horizontal=True)
shape = st.sidebar.radio("욕실 형태", ["사각형", "코너형"], horizontal=True)
btype = st.sidebar.radio("욕실 유형", ["샤워형", "욕조형", "구분없음"], horizontal=True)

st.sidebar.subheader("치수 입력 (mm)")
bw = st.sidebar.number_input("욕실 폭", min_value=400, step=10, value=1500)
bl = st.sidebar.number_input("욕실 길이", min_value=400, step=10, value=2200)

# 세면/샤워 비활성 조건: 중앙배수 Yes 또는 유형 '구분없음'
disable_sink_shower = (central == "Yes") or (btype == "구분없음")

col_ss1, col_ss2 = st.sidebar.columns(2)
with col_ss1:
    sw = st.sidebar.number_input(
        "세면부 폭", min_value=0, step=10, value=1300, disabled=disable_sink_shower
    )
with col_ss2:
    sl = st.sidebar.number_input(
        "세면부 길이", min_value=0, step=10, value=1500, disabled=disable_sink_shower
    )

col_sh1, col_sh2 = st.sidebar.columns(2)
with col_sh1:
    shw = st.sidebar.number_input(
        "샤워부 폭", min_value=0, step=10, value=800, disabled=disable_sink_shower
    )
with col_sh2:
    shl = st.sidebar.number_input(
        "샤워부 길이", min_value=0, step=10, value=900, disabled=disable_sink_shower
    )

# 비활성일 때는 None으로 전달 → 비교 생략
if disable_sink_shower:
    sw = None
    sl = None
    shw = None
    shl = None

# --- 샤워부 1000×900 예외처리 UI (유효값으로만 반영) ---
EXC_KEY = "exc_1000_900_choice"

# 기본 유효값은 입력값 그대로
shw_eff, shl_eff = shw, shl
exception_applied = False

if (not disable_sink_shower) and (shw is not None) and (shl is not None):
    if shw == 1000 and shl == 900:
        st.sidebar.warning(
            "샤워부 1000×900은 예외 규격으로 사용될 수 있습니다. 900×1000으로 간주할까요?"
        )
        choice = st.sidebar.radio(
            "예외처리",
            ["원래값 유지 (1000×900)", "예외 적용 (900×1000)"],
            key=EXC_KEY,
            horizontal=False,
        )
        if "예외 적용" in choice:
            shw_eff, shl_eff = 900, 1000
            exception_applied = True

st.sidebar.subheader("계산 옵션")
mgmt_rate_pct = st.sidebar.number_input(
    "생산관리비율 (%)", min_value=0.0, step=0.5, value=25.0
)
mgmt_rate = mgmt_rate_pct / 100.0

pve_kind = st.sidebar.radio(
    "PVE 유형", ["일반형 (+380mm)", "주거약자 (+480mm)"], index=0
)

st.sidebar.write("---")
do_calc = st.sidebar.button("계산하기", type="primary")


# ---------------------------
# 데이터 로딩 및 정규화
# ---------------------------
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "소재",
        "중앙배수",
        "형태",
        "유형",
        "욕실폭",
        "욕실길이",
        "세면부폭",
        "세면부길이",
        "샤워부폭",
        "샤워부길이",
        "소계",
    ]
    extra = ["부재료", "수량", "단가1", "노무비", "단가2"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df["형태"] = df["형태"].replace({"샤각형": "사각형"}).fillna("")
    df["유형"] = df["유형"].replace({"샤워": "샤워형"}).fillna("")
    df["중앙배수"] = df["중앙배수"].astype(str).str.strip().str.title()
    df["중앙배수"] = df["중앙배수"].replace(
        {"Y": "Yes", "N": "No", "Yes": "Yes", "No": "No"}
    )

    num_cols = [
        "욕실폭",
        "욕실길이",
        "세면부폭",
        "세면부길이",
        "샤워부폭",
        "샤워부길이",
        "소계",
    ] + [c for c in extra if c in df.columns]
    for c in num_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.replace(",", "", regex=False)
            .replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
        )
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def is_nan(x) -> bool:
    return pd.isna(x)


# ----- 정확 일치 비교 도우미 -----
def exact_eq(a: Optional[float], b: Optional[float]) -> bool:
    if is_nan(a) or is_nan(b):
        return False
    try:
        return float(a) == float(b)
    except Exception:
        return False


def exact_eq_series(s: pd.Series, value: Optional[float]) -> pd.Series:
    if value is None:
        return pd.Series(True, index=s.index)  # 입력이 None이면 조건 생략
    return (~s.isna()) & (s.astype(float) == float(value))


# 입력이 None이면 조건 생략 (정확 일치)
def optional_eq_series(s: pd.Series, value: Optional[float]) -> pd.Series:
    if value is None:
        return pd.Series(True, index=s.index)
    return exact_eq_series(s, value)


# ---------------------------
# PVE 계산
# ---------------------------
def pve_quote(
    width_mm: int, length_mm: int, mgmt_rate: float, kind: str = "일반형"
) -> Dict[str, Any]:
    add = 380 if "일반" in kind else 480
    w_m = (width_mm + add) / 1000.0
    l_m = (length_mm + add) / 1000.0
    area = w_m * l_m
    raw = round(area * 12000)  # 원재료비
    process = 24331  # 가공비
    subtotal = raw + process  # 소계
    subtotal_mgmt = round(subtotal * (1.0 + mgmt_rate))
    return {
        "소재": "PVE",
        "원재료비": int(raw),
        "가공비": int(process),
        "소계": int(subtotal),
        "관리비율": mgmt_rate,
        "관리비포함소계": int(subtotal_mgmt),
        "설명": f"PVE({kind}) 계산: (W+{add})*(L+{add}), 면적×12000 + 24331 후 관리비율 적용",
    }


# ---------------------------
# 매칭 함수들 (모두 dict 또는 None 반환)
# ---------------------------
def match_center_drain(
    df: pd.DataFrame, shape: str, btype: str, bw: int, bl: int
) -> Optional[Dict[str, Any]]:
    """중앙배수 Yes: GRP(중앙배수 계열)만 매칭"""
    C = (df["중앙배수"] == "Yes") & (df["형태"] == shape) & (df["유형"] == btype)
    sub = df[C & df["소재"].str.startswith("GRP", na=False)]
    cond = exact_eq_series(sub["욕실폭"], bw) & exact_eq_series(sub["욕실길이"], bl)
    hit = sub[cond]
    if hit.empty:
        return None
    row = hit.sort_values("소계", ascending=True).iloc[0]
    return {"row": row, "소재": "GRP(중앙배수)", "단차없음": False}


def match_non_center_rectangle(
    df: pd.DataFrame,
    btype: str,
    bw: int,
    bl: int,
    sw: Optional[int],
    sl: Optional[int],
    shw: Optional[int],
    shl: Optional[int],
) -> Optional[Dict[str, Any]]:
    """중앙배수 No & 사각형 정책"""
    base = df[(df["중앙배수"] == "No") & (df["형태"] == "사각형")]

    # A) 구분없음: GRP만 W/L 매칭
    if btype == "구분없음":
        grp = base[
            (base["유형"] == "구분없음")
            & (base["소재"].str.startswith("GRP", na=False))
        ]
        cond = exact_eq_series(grp["욕실폭"], bw) & exact_eq_series(grp["욕실길이"], bl)
        hit = grp[cond]
        if hit.empty:
            return None
        row = hit.sort_values("소계").iloc[0]
        return {"row": row, "소재": "GRP", "단차없음": False}

    # B) 샤워형: GRP는 세면/샤워 치수 없음 → FRP만
    if btype == "샤워형":
        frp = base[(base["유형"] == "샤워형") & (base["소재"] == "FRP")]
        # 특수규격(단차없음)
        special = {(1200, 1900), (1400, 1900)}
        if (bw, bl) in special:
            cond = exact_eq_series(frp["욕실폭"], bw) & exact_eq_series(
                frp["욕실길이"], bl
            )
            hit = frp[cond]
            if hit.empty:
                return None
            row = hit.sort_values("소계").iloc[0]
            return {"row": row, "소재": "FRP", "단차없음": True}

        cond = (
            exact_eq_series(frp["욕실폭"], bw)
            & exact_eq_series(frp["욕실길이"], bl)
            & optional_eq_series(frp["세면부폭"], sw)
            & optional_eq_series(frp["세면부길이"], sl)
            & optional_eq_series(frp["샤워부폭"], shw)
            & optional_eq_series(frp["샤워부길이"], shl)
        )
        hit = frp[cond]
        if hit.empty:
            return None
        row = hit.sort_values("소계").iloc[0]
        return {"row": row, "소재": "FRP", "단차없음": False}

    # C) 욕조형: FRP만
    if btype == "욕조형":
        frp = base[(base["유형"] == "욕조형") & (base["소재"] == "FRP")]
        cond = (
            exact_eq_series(frp["욕실폭"], bw)
            & exact_eq_series(frp["욕실길이"], bl)
            & optional_eq_series(frp["세면부폭"], sw)
            & optional_eq_series(frp["세면부길이"], sl)
            & optional_eq_series(frp["샤워부폭"], shw)
            & optional_eq_series(frp["샤워부길이"], shl)
        )
        hit = frp[cond]
        if hit.empty:
            return None
        row = hit.sort_values("소계").iloc[0]
        return {"row": row, "소재": "FRP", "단차없음": False}

    return None


def match_corner_shower(
    df: pd.DataFrame,
    bw: int,
    bl: int,
    sw: Optional[int],
    sl: Optional[int],
    shw: Optional[int],
    shl: Optional[int],
) -> Optional[Dict[str, Any]]:
    """중앙배수 No & 코너형 & 샤워형: GRP→FRP"""
    C = (df["형태"] == "코너형") & (df["유형"] == "샤워형") & (df["중앙배수"] == "No")

    # 1) GRP
    grp = df[C & df["소재"].str.startswith("GRP", na=False)]
    cond_grp = (
        exact_eq_series(grp["욕실폭"], bw)
        & exact_eq_series(grp["욕실길이"], bl)
        & optional_eq_series(grp["세면부폭"], sw)
        & optional_eq_series(grp["세면부길이"], sl)
        & optional_eq_series(grp["샤워부폭"], shw)
        & optional_eq_series(grp["샤워부길이"], shl)
    )
    hit = grp[cond_grp]
    if not hit.empty:
        row = hit.sort_values("소계").iloc[0]
        return {"row": row, "소재": "GRP", "단차없음": False}

    # 2) FRP
    frp = df[C & (df["소재"] == "FRP")]
    cond_frp = (
        exact_eq_series(frp["욕실폭"], bw)
        & exact_eq_series(frp["욕실길이"], bl)
        & optional_eq_series(frp["세면부폭"], sw)
        & optional_eq_series(frp["세면부길이"], sl)
        & optional_eq_series(frp["샤워부폭"], shw)
        & optional_eq_series(frp["샤워부길이"], shl)
    )
    hit = frp[cond_frp]
    if not hit.empty:
        row = hit.sort_values("소계").iloc[0]
        return {"row": row, "소재": "FRP", "단차없음": False}

    return None


# ---------------------------
# 도형 렌더링 (PIL, 약 1/3 화면 크기)
# ---------------------------
def draw_bathroom(
    shape: str,
    bw_mm: int,
    bl_mm: int,  # 욕실 폭(세로), 욕실 길이(가로)
    sw_mm: int | None,
    sl_mm: int | None,  # 세면부 폭/길이
    shw_mm: int | None,
    shl_mm: int | None,  # 샤워부 폭/길이
    central: str | None = None,  # "Yes"/"No"
    btype: str | None = None,  # "샤워형"/"욕조형"/"구분없음"
) -> Image.Image:
    """
    렌더 규칙
    - 중앙배수=Yes 또는 유형=구분없음 → 외곽 사각형만 그림(내부 구획 생략)
    - 사각형 → 세면부(좌하), 샤워부(우하) '폭×길이' 그대로
    - 코너형 → 좌측 전고는 세면부(파랑), 우측은 샤워부(빨강, 90° 회전: 가로=샤워부 길이, 세로=샤워부 폭),
               두 영역 사이 빨간 세로 경계선 표시
    출력 크기: 약 1/3 화면(540×360 px)
    """
    # ── 캔버스 설정
    W, H = 540, 360
    PAD, BORDER, GAP = 14, 6, 4

    img = Image.new("RGB", (W, H), "white")
    drw = ImageDraw.Draw(img)

    def safe_rect(x0, y0, x1, y1, color, width=3):
        """좌표가 유효할 때만 사각형 그림(예외 방지)."""
        if x1 <= x0 or y1 <= y0:
            return False
        drw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return True

    def text_center(x, y, txt, fill="black"):
        """Pillow 버전 호환용 중앙 정렬 텍스트."""
        try:
            drw.text((x, y), txt, fill=fill, anchor="mm")
        except TypeError:
            drw.text((x - 20, y - 8), txt, fill=fill)

    # None 방어
    sw = 0 if sw_mm is None else int(sw_mm)
    sl = 0 if sl_mm is None else int(sl_mm)
    shw = 0 if shw_mm is None else int(shw_mm)
    shl = 0 if shl_mm is None else int(shl_mm)

    # 스케일(mm→px) : 가로=욕실길이, 세로=욕실폭
    sx = (W - 2 * PAD) / float(max(1, bl_mm))
    sy = (H - 2 * PAD) / float(max(1, bw_mm))
    s = min(sx, sy)

    # 욕실 외곽
    BW = int(round(bl_mm * s))  # 가로 px
    BH = int(round(bw_mm * s))  # 세로 px
    x0 = (W - BW) // 2
    y0 = (H - BH) // 2
    x1 = x0 + BW
    y1 = y0 + BH
    safe_rect(x0, y0, x1, y1, "black", 3)

    # 치수 라벨(간단)
    try:
        drw.text(((x0 + x1) / 2, y0 - 8), "욕실길이", fill="black", anchor="mb")
        drw.text((x0 - 8, (y0 + y1) / 2), "욕실폭", fill="black", anchor="rm")
    except Exception:
        pass

    # ── 중앙배수 Yes 또는 유형 구분없음 → 외곽만
    if (central == "Yes") or (btype == "구분없음"):
        return img

    # ── 사각형
    if shape == "사각형":
        # 세면부(좌하)
        if sw > 0 and sl > 0:
            sink_w = int(round(min(sw, bl_mm) * s))
            sink_h = int(round(min(sl, bw_mm) * s))
            sx0 = x0 + BORDER
            sy1 = y1 - BORDER
            sx1 = min(x1 - BORDER, sx0 + sink_w)
            sy0 = max(y0 + BORDER, sy1 - sink_h)
            if safe_rect(sx0, sy0, sx1, sy1, "blue", 3):
                text_center((sx0 + sx1) / 2, (sy0 + sy1) / 2, "세면부", "blue")

        # 샤워부(우하)
        if shw > 0 and shl > 0:
            sh_w = int(round(min(shw, bl_mm) * s))
            sh_h = int(round(min(shl, bw_mm) * s))
            tx1 = x1 - BORDER
            ty1 = y1 - BORDER
            tx0 = max(x0 + BORDER, tx1 - sh_w)
            ty0 = max(y0 + BORDER, ty1 - sh_h)
            # 세면부와 겹치면 우측으로 한 칸 밀어줌
            if "sx1" in locals() and tx0 < (sx1 + GAP):
                tx0 = min(tx1 - 1, sx1 + GAP)
            if safe_rect(tx0, ty0, tx1, ty1, "red", 3):
                text_center((tx0 + tx1) / 2, (ty0 + ty1) / 2, "샤워부", "red")

        return img

    # ── 코너형
    # 좌측: 세면부(전고). 우측: 샤워부(90° 회전). 두 영역 사이 빨간 경계선.
    total_w = sw + shw
    ratio = (sw / total_w) if total_w > 0 else 0.5  # 세면부 비율
    boundary_x = x0 + int(round(BW * ratio))

    # 세면부(전고)
    left_x0 = x0 + BORDER
    left_x1 = max(left_x0 + 1, boundary_x - GAP)
    if left_x1 > left_x0:
        if safe_rect(left_x0, y0 + BORDER, left_x1, y1 - BORDER, "blue", 3):
            text_center((left_x0 + left_x1) / 2, (y0 + y1) / 2, "세면부", "blue")

    # 경계선(전고)
    ImageDraw.Draw(img).line(
        [boundary_x, y0 + BORDER // 2, boundary_x, y1 - BORDER // 2],
        fill="red",
        width=3,
    )

    # 샤워부(우측, 90° 회전: 가로=샤워부 '길이', 세로=샤워부 '폭')
    if shw > 0 and shl > 0:
        usable_w = (x1 - boundary_x) - BORDER
        rot_w = int(round(min(shl, bl_mm) * s))  # 회전 후 가로
        rot_h = int(round(min(shw, bw_mm) * s))  # 회전 후 세로
        rx1 = x1 - BORDER
        ry1 = y1 - BORDER
        rx0 = max(boundary_x + BORDER, rx1 - min(rot_w, usable_w))
        ry0 = max(y0 + BORDER, ry1 - rot_h)
        if safe_rect(rx0, ry0, rx1, ry1, "red", 3):
            text_center((rx0 + rx1) / 2, (ry0 + ry1) / 2, "샤워부", "red")

    return img


# ---------------------------
# 실행
# ---------------------------
st.title("바닥판 규격/옵션 산출")

if not uploaded:
    st.info(
        "왼쪽에서 엑셀 파일(시트명: **바닥판**)을 업로드한 뒤, **계산하기**를 눌러주세요."
    )
    st.stop()

# 엑셀 로딩
try:
    raw = pd.read_excel(uploaded, sheet_name="바닥판")
except Exception as e:
    st.error(f"엑셀 로딩 오류: {e}")
    st.stop()

df = normalize_df(raw)

if do_calc:
    decision_log = []

    if exception_applied:
        decision_log.append("샤워부 1000×900 → 예외규칙 적용으로 900×1000 간주")
    elif (
        (not disable_sink_shower)
        and (shw is not None)
        and (shl is not None)
        and shw == 1000
        and shl == 900
    ):
        decision_log.append(
            "샤워부 1000×900 감지됨(예외 규격) → 사이드바에서 적용 여부 선택 가능"
        )

    # (선택) 샤워부 1000×900 → 900×1000으로 정확 일치 교정이 필요하다면 아래 주석 해제
    # if (not disable_sink_shower) and (shw is not None) and (shl is not None):
    #     if exact_eq(shw, 1000) and exact_eq(shl, 900):
    #         decision_log.append("샤워부(1000×900) → 예외규칙(정확일치)으로 900×1000 교정")
    #         shw, shl = 900, 1000

    # 세대수 우선 규칙
    if units < 100:
        decision_log.append(f"세대수={units} (<100) → PVE 강제 선택")
        q = pve_quote(bw, bl, mgmt_rate, pve_kind)
        material = q["소재"]
        base_subtotal = q["소계"]
        mgmt_total = q["관리비포함소계"]
        result_kind = "PVE"

    else:
        # 1) 중앙배수 Yes → GRP(중앙배수) 시도 → 실패 시 PVE
        if central == "Yes":
            decision_log.append("중앙배수=Yes → GRP(중앙배수) 매칭 시도")
            matched = match_center_drain(df, shape, btype, bw, bl)
            if matched is None:
                decision_log.append("GRP(중앙배수) 매칭 실패 → PVE 계산")
                q = pve_quote(bw, bl, mgmt_rate, pve_kind)
                material = q["소재"]
                base_subtotal = q["소계"]
                mgmt_total = q["관리비포함소계"]
                result_kind = "PVE"
            else:
                row = matched["row"]
                material = matched["소재"]
                base_subtotal = int(row["소계"])
                result_kind = material
                decision_log.append("GRP(중앙배수) 매칭 성공 → 최소 소계 선택")

        # 2) 중앙배수 No
        else:
            if shape == "사각형":
                decision_log.append("중앙배수=No & 형태=사각형")
                matched = match_non_center_rectangle(
                    df, btype, bw, bl, sw, sl, shw_eff, shl_eff
                )
                if matched is None:
                    decision_log.append("사각형 매칭 실패 → PVE 계산")
                    q = pve_quote(bw, bl, mgmt_rate, pve_kind)
                    material = q["소재"]
                    base_subtotal = q["소계"]
                    mgmt_total = q["관리비포함소계"]
                    result_kind = "PVE"
                else:
                    row = matched["row"]
                    material = matched["소재"]
                    base_subtotal = int(row["소계"])
                    result_kind = f"{material}" + (
                        " (단차없음)" if matched.get("단차없음") else ""
                    )
                    decision_log.append(f"{result_kind} 매칭 성공 → 최소 소계 선택")
            else:
                decision_log.append(
                    "중앙배수=No & 형태=코너형 & 유형=샤워형 → GRP→FRP 순서"
                )
                matched = match_corner_shower(
                    df,
                    bw,
                    bl,
                    sw,
                    sl,
                    shw_eff,
                    shl_eff,
                )
                if matched is None:
                    decision_log.append("코너형/샤워형 매칭 실패 → PVE 계산")
                    q = pve_quote(bw, bl, mgmt_rate, pve_kind)
                    material = q["소재"]
                    base_subtotal = q["소계"]
                    mgmt_total = q["관리비포함소계"]
                    result_kind = "PVE"
                else:
                    row = matched["row"]
                    material = matched["소재"]
                    base_subtotal = int(row["소계"])
                    result_kind = material
                    decision_log.append(f"{result_kind} 매칭 성공 → 최소 소계 선택")

        # 공통: 관리비 포함 소계(매칭 케이스에도 적용)
        mgmt_total = int(round(base_subtotal * (1.0 + mgmt_rate)))

    # ---------------------------
    # 출력
    # ---------------------------
    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        img = draw_bathroom(shape, bw, bl, sw, sl, shw_eff, shl_eff, central, btype)
        st.image(img, caption="욕실 도형(약 1/3 크기)", width=480)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with right:
        st.subheader("선택된 바닥판")
        st.write(f"**재질**: {result_kind}")
        st.write(f"**소계(원)**: {base_subtotal:,}")
        st.write(
            f"**관리비 포함 소계(원)**: {mgmt_total:,}  (관리비율 {mgmt_rate_pct:.1f}%)"
        )

        st.info("결정 과정", icon="ℹ️")
        st.write("\n".join([f"- {x}" for x in decision_log]))

        st.markdown("---")
        b1, b2, b3 = st.columns([1, 1, 2])

        def _save_done():
            # 바닥 결과 요약 저장 (벽 페이지에서 활용할 수 있도록 최소 정보만)
            st.session_state[FLOOR_RESULT_KEY] = {
                "material": result_kind,
                "subtotal": int(base_subtotal),
                "subtotal_with_mgmt": int(mgmt_total),
                "inputs": {
                    "central": central,
                    "shape": shape,
                    "btype": btype,
                    "bw": int(bw),
                    "bl": int(bl),
                    "sw": (None if sw is None else int(sw)),
                    "sl": (None if sl is None else int(sl)),
                    "shw": (None if shw_eff is None else int(shw_eff)),
                    "shl": (None if shl_eff is None else int(shl_eff)),
                    "mgmt_rate_pct": float(mgmt_rate_pct),
                    "pve_kind": pve_kind,
                    "units": int(units),
                },
            }
            st.session_state[FLOOR_DONE_KEY] = True
            st.success("바닥 계산 결과를 저장했습니다. (다음 단계로 이동 가능)")

        def _reset_done():
            st.session_state[FLOOR_DONE_KEY] = False
            st.session_state[FLOOR_RESULT_KEY] = None
            st.info("저장 상태를 초기화했습니다.")

        with b1:
            st.button("✅ 완료 저장", on_click=_save_done, type="primary")
        with b2:
            st.button("↩️ 초기화", on_click=_reset_done)

        with b3:
            # 프로그램적으로 벽 페이지로 이동 (Streamlit 1.25+)
            go_wall = st.button("➡️ 벽 계산기로 이동", help="저장 후 이동을 권장")
            if go_wall:
                try:
                    st.switch_page("pages/original_wall.py")
                except Exception:
                    st.info("좌측 네비게이션에서 ‘벽판 계산’ 페이지로 이동해주세요.")

    st.success("계산 완료 ✅")
