# streamlit run app.py
import io
import os, glob, json
from typing import Optional, Dict, Any, Tuple

# --- Common Styles ---
from common_styles import apply_common_styles, set_page_config

# --- Streamlit ---
import streamlit as st

set_page_config(page_title="바닥판 규격/옵션 산출", layout="wide")
apply_common_styles()

# --- Authentication ---
import auth

auth.require_auth()

# --- Pillow / Image ---
from PIL import Image, ImageDraw, ImageFont

# --- Numpy / Pandas ---
import numpy as np
import pandas as pd

# --- Floor/Walls 연동용 상태키 ---
FLOOR_DONE_KEY = "floor_done"
FLOOR_RESULT_KEY = "floor_result"

# 공유 카탈로그 세션 키 (모든 페이지에서 공통 사용)
SHARED_EXCEL_KEY = "shared_excel_file"
SHARED_EXCEL_NAME_KEY = "shared_excel_filename"

# 공유 욕실 정보 세션 키
SHARED_BATH_SHAPE_KEY = "shared_bath_shape"  # 욕실 형태: "사각형" or "코너형"
SHARED_BATH_WIDTH_KEY = "shared_bath_width"  # 욕실 폭 (bw)
SHARED_BATH_LENGTH_KEY = "shared_bath_length"  # 욕실 길이 (bl)
SHARED_SINK_WIDTH_KEY = "shared_sink_width"  # 세면부 폭 (sw) - 경계선 정보
SHARED_MATERIAL_KEY = "shared_floor_material"  # 바닥판 재료 (결과에서 추출)

# ===== 경로 =====
EXPORT_DIR = "exports"  # 섹션 JSON 저장 폴더
os.makedirs(EXPORT_DIR, exist_ok=True)


# ===== 유틸 =====
def _get_font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("NotoSansKR-Regular.ttf", size)
    except:
        return ImageFont.load_default()


def _map_floor_material_label(result_kind: str) -> str:
    rk = (result_kind or "").upper()
    if "PVE" in rk:
        return "PP/PE 바닥판"
    if "FRP" in rk:
        return "SMC/FRP바닥판"
    return "GRP바닥판"


def _extract_prices_from_row(row) -> Dict[str, int]:
    prices = {"단가1": 0, "노무비": 0, "단가2": 0}
    if row is None:
        return prices
    for k in prices.keys():
        if k in row and pd.notna(row[k]):
            try:
                prices[k] = int(row[k])
            except:
                pass
    return prices


def _pve_prices_from_quote(q: Dict[str, Any]) -> Dict[str, int]:
    return {
        "단가1": int(q.get("원재료비", 0)),
        "노무비": int(q.get("가공비", 0)),
        "단가2": int(q.get("소계", 0)),
    }


def save_json(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# def _init_state():
#     st.session_state.setdefault(FLOOR_DONE_KEY, False)
#     st.session_state.setdefault(FLOOR_RESULT_KEY, None)


# _init_state()


# --- Pillow font loader (CJK 안전) ---
def _get_font(size: int = 16) -> ImageFont.ImageFont:
    # 1) 프로젝트 루트에 폰트 파일 있으면 최우선 사용 (재현성↑)
    for name in [
        "NotoSansKR-Regular.ttf",
        "NanumGothic.ttf",
        "Pretendard-Regular.otf",
        "NotoSans-Regular.ttf",
        "Malgun.ttf",
    ]:
        p = os.path.join(os.getcwd(), name)
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass

    # 2) 시스템 경로 탐색(리눅스/맥/윈도 공통 후보)
    candidates = []
    for pat in [
        "/usr/share/fonts/**/NotoSans*.*",
        "/usr/share/fonts/**/Nanum*.*",
        "/Library/Fonts/**/AppleSDGothicNeo*.*",
        "C:/Windows/Fonts/*malgun*.*",
        "C:/Windows/Fonts/*nanum*.*",
        "C:/Windows/Fonts/*noto*.*",
    ]:
        candidates.extend(glob.glob(pat, recursive=True))
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue

    # 3) 최후: 기본 비트맵 폰트(한글은 각질 수 있음)
    return ImageFont.load_default()


# ===== 시공비 시트에서 PVE 가공비 추출 =====
def get_pve_process_cost(df_cost: pd.DataFrame) -> Optional[int]:
    """
    '시공비' 시트에서 항목=바닥판 이고 공정에 'PVE'가 포함된 행의 '시공비'를 반환.
    없으면 None.
    """
    df = df_cost.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 컬럼 매핑(유연 대응)
    col_map = {}
    for c in df.columns:
        cs = str(c).strip()
        if cs in ["항목", "Item"]:
            col_map["항목"] = c
        elif cs in ["공정", "공사", "Process"]:
            col_map["공정"] = c
        elif cs in ["시공비", "금액", "Cost"]:
            col_map["시공비"] = c

    if not {"항목","공정","시공비"}.issubset(col_map.keys()):
        return None

    def _clean_num(x):
        if pd.isna(x): return None
        s = str(x).replace(",", "").strip()
        try: return int(float(s))
        except: return None

    df["__항목"] = df[col_map["항목"]].astype(str).str.strip()
    df["__공정"] = df[col_map["공정"]].astype(str).str.strip()
    df["__시공비"] = df[col_map["시공비"]].apply(_clean_num)

    hit = df[
        (df["__항목"] == "바닥판") &
        (df["__공정"].str.contains("PVE", case=False, na=False))
    ]

    vals = hit["__시공비"].dropna()
    return int(vals.iloc[0]) if not vals.empty else None


# ---------------------------
# UI: Sidebar (왼쪽 입력 인터페이스)
# ---------------------------
st.sidebar.header("입력값 (왼쪽 인터페이스)")
uploaded = st.sidebar.file_uploader(
    "엑셀 업로드 (필수 시트: '바닥판', '시공비')", type=["xlsx", "xls"]
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
prod_rate_pct = st.sidebar.number_input(
    "생산관리비율 rₚ (%)", min_value=0.0, max_value=99.9, step=0.5, value=25.0
)
sales_rate_pct = st.sidebar.number_input(
    "영업관리비율 rₛ (%)", min_value=0.0, max_value=30.0, step=0.5, value=20.0
)
r_p = prod_rate_pct / 100.0
r_s = sales_rate_pct / 100.0

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
# 관리비 2단계 계산 (생산관리비 + 영업관리비)
# ---------------------------
def price_blocks_pve(subtotal: int, r_p: float, r_s: float) -> Dict[str, int]:
    """PVE: 생산관리비 비포함(단순 곱), 영업관리비 포함역산."""
    prod_fee = int(round(subtotal * r_p))
    prod_incl = int(round(subtotal + prod_fee))
    sales_fee = int(round(prod_incl / (1.0 - r_s) - prod_incl)) if r_s > 0 else 0
    sales_incl = int(round(prod_incl + sales_fee))
    return {
        "생산관리비": prod_fee,
        "생산관리비포함": prod_incl,
        "영업관리비": sales_fee,
        "영업관리비포함": sales_incl,
    }


def price_blocks_grp_frp(subtotal: int, r_p: float, r_s: float) -> Dict[str, int]:
    """GRP/FRP: 생산관리비 포함역산, 영업관리비 포함역산."""
    if r_p >= 1.0:
        raise ZeroDivisionError("생산관리비율은 100% 미만이어야 합니다.")
    prod_incl = int(round(subtotal / (1.0 - r_p))) if r_p > 0 else int(subtotal)
    prod_fee = int(round(prod_incl - subtotal))
    sales_fee = int(round(prod_incl / (1.0 - r_s) - prod_incl)) if r_s > 0 else 0
    sales_incl = int(round(prod_incl + sales_fee))
    return {
        "생산관리비": prod_fee,
        "생산관리비포함": prod_incl,
        "영업관리비": sales_fee,
        "영업관리비포함": sales_incl,
    }


# ---------------------------
# PVE 계산
# ---------------------------
def pve_quote(
    width_mm: int,
    length_mm: int,
    r_p: float,
    r_s: float,
    pve_process_cost: Optional[int],
    kind: str = "일반형",
) -> Dict[str, Any]:
    """PVE 원가 산정 + 관리비 블록."""
    add = 380 if "일반" in kind else 480
    w_m = (width_mm + add) / 1000.0
    l_m = (length_mm + add) / 1000.0
    area = w_m * l_m
    raw = int(round(area * 12000))  # ㎡당 12,000원
    process = int(pve_process_cost) if pve_process_cost is not None else 24331
    subtotal = raw + process
    blocks = price_blocks_pve(subtotal, r_p, r_s)
    return {
        "소재": "PVE",
        "원재료비": raw,
        "가공비": process,
        "소계": subtotal,
        **blocks,
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
# 도형 렌더링 (PIL, 고해상도 렌더링 후 축소)
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
    출력: 1080×720(2x)로 그리고 화면엔 540px로 축소 표시
    """
    # ── 캔버스 설정 (고해상도 렌더링)
    BASE_W, BASE_H = 540, 360
    SCALE = 2  # 2배로 그리고 축소 표시
    W, H = BASE_W * SCALE, BASE_H * SCALE

    # ✅ 방향별 패딩: 위/왼쪽을 크게 잡아 라벨 공간 확보
    PAD_L = 48 * SCALE  # 왼쪽 (라벨 "욕실폭"이 바깥으로 나갈 공간)
    PAD_R = 16 * SCALE
    PAD_T = 48 * SCALE  # 위쪽 (라벨 "욕실길이"가 바깥으로 나갈 공간)
    PAD_B = 16 * SCALE

    BORDER, GAP = 6 * SCALE, 4 * SCALE

    img = Image.new("RGB", (W, H), "white")
    drw = ImageDraw.Draw(img)

    # 폰트(라벨/작은 글자)
    font_label = _get_font(18 * SCALE)
    font_small = _get_font(14 * SCALE)

    def safe_rect(x0, y0, x1, y1, color, width=3 * SCALE):
        """좌표가 유효할 때만 사각형 그림(예외 방지)."""
        if x1 <= x0 or y1 <= y0:
            return False
        drw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        return True

    def text_center(x, y, txt, fill="black", font=None):
        if font is None:
            font = font_label
        try:
            drw.text((x, y), txt, fill=fill, anchor="mm", font=font)
        except TypeError:
            drw.text((x - 20 * SCALE, y - 8 * SCALE), txt, fill=fill, font=font)

    # None 방어
    sw = 0 if sw_mm is None else int(sw_mm)
    sl = 0 if sl_mm is None else int(sl_mm)
    shw = 0 if shw_mm is None else int(shw_mm)
    shl = 0 if shl_mm is None else int(shl_mm)

    # ✅ 방향별 패딩을 반영한 가용 너비/높이
    avail_w = W - (PAD_L + PAD_R)
    avail_h = H - (PAD_T + PAD_B)

    # 스케일(mm→px) : 가로=욕실길이, 세로=욕실폭
    sx = avail_w / float(max(1, bl_mm))
    sy = avail_h / float(max(1, bw_mm))
    s = min(sx, sy)

    # 욕실 외곽
    BW = int(round(bl_mm * s))  # 가로 px
    BH = int(round(bw_mm * s))  # 세로 px
    x0 = (W - BW) // 2
    y0 = (H - BH) // 2
    x1 = x0 + BW
    y1 = y0 + BH
    safe_rect(x0, y0, x1, y1, "black", 3 * SCALE)

    # ✅ 라벨을 그릴 좌표 계산 + 화면 밖 방지(최소값 클램프)
    # 텍스트 크기 파악(혹시 anchor 미지원 Pillow 대비)
    try:
        # getbbox → (x0, y0, x1, y1)
        bx1 = font_small.getbbox("욕실길이")
        w1, h1 = (bx1[2] - bx1[0], bx1[3] - bx1[1])
        bx2 = font_small.getbbox("욕실폭")
        w2, h2 = (bx2[2] - bx2[0], bx2[3] - bx2[1])
    except Exception:
        # getbbox 미지원일 경우 대략값
        w1 = 80 * SCALE
        h1 = 20 * SCALE
        w2 = 60 * SCALE
        h2 = 20 * SCALE

    # 위쪽 중앙 바깥(아래로 붙이는 'mb' 기준): y가 너무 작아지지 않게 클램프
    top_x = (x0 + x1) / 2
    top_y = max(4 * SCALE + h1, y0 - 8 * SCALE)
    try:
        drw.text((top_x, top_y), "욕실길이", fill="black", anchor="mb", font=font_small)
    except Exception:
        # anchor 미지원일 때 대략 중앙 정렬
        drw.text(
            (top_x - w1 / 2, top_y - h1), "욕실길이", fill="black", font=font_small
        )

    # 왼쪽 중앙 바깥(오른쪽으로 붙이는 'rm' 기준): x가 너무 작아지지 않게 클램프
    left_x = max(4 * SCALE + w2, x0 - 8 * SCALE)
    left_y = (y0 + y1) / 2
    try:
        drw.text((left_x, left_y), "욕실폭", fill="black", anchor="rm", font=font_small)
    except Exception:
        drw.text(
            (left_x - w2, left_y - h2 / 2), "욕실폭", fill="black", font=font_small
        )

    # 치수 라벨(간단)
    # try:
    #     drw.text(((x0 + x1) / 2, y0 - 8 * SCALE), "욕실길이",
    #              fill="black", anchor="mb", font=font_small)
    #     drw.text((x0 - 8 * SCALE, (y0 + y1) / 2), "욕실폭",
    #              fill="black", anchor="rm", font=font_small)
    # except Exception:
    #     pass

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
            if safe_rect(sx0, sy0, sx1, sy1, "blue", 3 * SCALE):
                text_center(
                    (sx0 + sx1) / 2, (sy0 + sy1) / 2, "세면부", "blue", font=font_label
                )

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
            if safe_rect(tx0, ty0, tx1, ty1, "red", 3 * SCALE):
                text_center(
                    (tx0 + tx1) / 2, (ty0 + ty1) / 2, "샤워부", "red", font=font_label
                )

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
        if safe_rect(left_x0, y0 + BORDER, left_x1, y1 - BORDER, "blue", 3 * SCALE):
            text_center(
                (left_x0 + left_x1) / 2,
                (y0 + y1) / 2,
                "세면부",
                "blue",
                font=font_label,
            )

    # 경계선(전고)
    ImageDraw.Draw(img).line(
        [boundary_x, y0 + BORDER // 2, boundary_x, y1 - BORDER // 2],
        fill="red",
        width=3 * SCALE,
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
        if safe_rect(rx0, ry0, rx1, ry1, "red", 3 * SCALE):
            text_center(
                (rx0 + rx1) / 2, (ry0 + ry1) / 2, "샤워부", "red", font=font_label
            )

    return img


# ---------------------------
# 실행
# ---------------------------
st.title("바닥판 규격/옵션 산출")

# 업로드된 파일이 있으면 세션에 저장
if uploaded:
    st.session_state[SHARED_EXCEL_KEY] = uploaded
    st.session_state[SHARED_EXCEL_NAME_KEY] = uploaded.name

# 세션에 저장된 파일 사용 (현재 페이지 업로드 or 다른 페이지에서 업로드)
excel_file = st.session_state.get(SHARED_EXCEL_KEY)
excel_filename = st.session_state.get(SHARED_EXCEL_NAME_KEY, "알 수 없음")

if not excel_file:
    st.info(
        "왼쪽에서 엑셀 파일(필수 시트: **바닥판**, **시공비**)을 업로드한 뒤, **계산하기**를 눌러주세요. (천장판/벽판 페이지에서 업로드한 파일도 사용 가능)"
    )
    st.stop()

# 공유 카탈로그 표시
if uploaded:
    st.success(f"✅ 엑셀 로드 완료 (현재 페이지): {excel_filename}")
else:
    st.info(f"📂 공유 카탈로그 사용 중: {excel_filename}")

# 엑셀 로딩
try:
    xls = pd.ExcelFile(excel_file)
except Exception as e:
    st.error(f"엑셀 로딩 실패: {e}")
    st.stop()

missing_sheets = [s for s in ["바닥판", "시공비"] if s not in xls.sheet_names]
if missing_sheets:
    st.error(f"필수 시트 누락: {missing_sheets} — 엑셀을 확인하세요.")
    st.stop()

try:
    raw = pd.read_excel(xls, sheet_name="바닥판")
    df = normalize_df(raw)
except Exception as e:
    st.error(f"'바닥판' 시트 파싱 실패: {e}")
    st.stop()

try:
    df_cost = pd.read_excel(xls, sheet_name="시공비")
    pve_process_cost = get_pve_process_cost(df_cost)  # 못 찾으면 None
except Exception:
    pve_process_cost = None

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

    # 이 변수들 반드시 모든 분기에서 채워지게 기본값 준비
    result_kind = None
    base_subtotal = 0
    mgmt_total = 0
    prices = {"단가1": 0, "노무비": 0, "단가2": 0}
    material_label = ""
    floor_spec = f"{int(bw)}×{int(bl)}"  # 기본 규격 문자열

    # ---------------------------
    # 결정 로직
    # ---------------------------
    if units < 100:
        # PVE 강제
        decision_log.append(f"세대수={units} (<100) → PVE 강제 선택")
        q = pve_quote(bw, bl, r_p, r_s, pve_process_cost, pve_kind)
        result_kind = "PVE"
        base_subtotal = q["소계"]
        mgmt_total = q["영업관리비포함"]
        prices = _pve_prices_from_quote(q)
    else:
        if central == "Yes":
            decision_log.append("중앙배수=Yes → GRP(중앙배수) 매칭 시도")
            matched = match_center_drain(df, shape, btype, bw, bl)
            if matched is None:
                decision_log.append("GRP(중앙배수) 매칭 실패 → PVE 계산")
                q = pve_quote(bw, bl, r_p, r_s, pve_process_cost, pve_kind)
                result_kind = "PVE"
                base_subtotal = q["소계"]
                mgmt_total = q["영업관리비포함"]
                prices = _pve_prices_from_quote(q)
            else:
                row = matched["row"]
                result_kind = "GRP"  # 표준화
                base_subtotal = int(row["소계"])
                prices = _extract_prices_from_row(row)
                decision_log.append("GRP(중앙배수) 매칭 성공 → 최소 소계 선택")
        else:
            if shape == "사각형":
                decision_log.append("중앙배수=No & 형태=사각형")
                matched = match_non_center_rectangle(
                    df, btype, bw, bl, sw, sl, shw_eff, shl_eff
                )
                if matched is None:
                    decision_log.append("사각형 매칭 실패 → PVE 계산")
                    q = pve_quote(bw, bl, r_p, r_s, pve_process_cost, pve_kind)
                    result_kind = "PVE"
                    base_subtotal = q["소계"]
                    mgmt_total = q["영업관리비포함"]
                    prices = _pve_prices_from_quote(q)
                else:
                    row = matched["row"]
                    result_kind = "FRP" if matched["소재"] == "FRP" else "GRP"
                    base_subtotal = int(row["소계"])
                    prices = _extract_prices_from_row(row)
                    if matched.get("단차없음"):
                        result_kind += " (단차없음)"
                    decision_log.append(f"{result_kind} 매칭 성공 → 최소 소계 선택")
            else:
                decision_log.append(
                    "중앙배수=No & 형태=코너형 & 유형=샤워형 → GRP→FRP 순서"
                )
                matched = match_corner_shower(df, bw, bl, sw, sl, shw_eff, shl_eff)
                if matched is None:
                    decision_log.append("코너형/샤워형 매칭 실패 → PVE 계산")
                    q = pve_quote(bw, bl, r_p, r_s, pve_process_cost, pve_kind)
                    result_kind = "PVE"
                    base_subtotal = q["소계"]
                    mgmt_total = q["영업관리비포함"]
                    prices = _pve_prices_from_quote(q)
                else:
                    row = matched["row"]
                    result_kind = "FRP" if matched["소재"] == "FRP" else "GRP"
                    base_subtotal = int(row["소계"])
                    prices = _extract_prices_from_row(row)
                    decision_log.append(f"{result_kind} 매칭 성공 → 최소 소계 선택")

        # 매칭 케이스에도 관리비 적용 (GRP/FRP는 역산 방식)
        if mgmt_total == 0:
            blocks = price_blocks_grp_frp(base_subtotal, r_p, r_s)
            mgmt_total = blocks["영업관리비포함"]

    # 공통: 재질 라벨 및 규격(문자열) 정규화
    material_label = _map_floor_material_label(result_kind or "")
    floor_spec = f"{int(bw)}×{int(bl)}"  # 필요시 행(row)에서 규격 필드가 있으면 치환

    # 관리비 계산 (세션 상태용)
    if result_kind and "PVE" in result_kind.upper():
        # PVE는 이미 계산된 q 사용
        prod_fee = q.get("생산관리비", 0)
        prod_incl = q.get("생산관리비포함", 0)
        sales_fee = q.get("영업관리비", 0)
        sales_incl = mgmt_total
    else:
        # GRP/FRP는 역산
        blocks_session = price_blocks_grp_frp(base_subtotal, r_p, r_s)
        prod_fee = blocks_session["생산관리비"]
        prod_incl = blocks_session["생산관리비포함"]
        sales_fee = blocks_session["영업관리비"]
        sales_incl = blocks_session["영업관리비포함"]

    floor_result_payload = {
        "section": "floor",
        "material": material_label,
        "spec": floor_spec,
        "prices": {
            "단가1": int(prices.get("단가1", 0)),
            "노무비": int(prices.get("노무비", 0)),
            "단가2": int(prices.get("단가2", 0)),
        },
        "qty": 1,
        "meta": {
            "result_kind": result_kind,
            "subtotal": int(base_subtotal),
            "생산관리비": int(prod_fee),
            "생산관리비포함": int(prod_incl),
            "영업관리비": int(sales_fee),
            "영업관리비포함": int(sales_incl),
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
                "prod_rate_pct": float(prod_rate_pct),
                "sales_rate_pct": float(sales_rate_pct),
                "pve_kind": pve_kind,
                "units": int(units),
            },
        },
    }

    # 세션 상태에 자동 저장
    st.session_state[FLOOR_RESULT_KEY] = floor_result_payload
    st.session_state[FLOOR_DONE_KEY] = True

    # 공유 욕실 정보 저장 (벽판/천장판에서 사용)
    st.session_state[SHARED_BATH_SHAPE_KEY] = shape  # "사각형" or "코너형"
    st.session_state[SHARED_BATH_WIDTH_KEY] = int(bw)  # 욕실 폭
    st.session_state[SHARED_BATH_LENGTH_KEY] = int(bl)  # 욕실 길이
    st.session_state[SHARED_SINK_WIDTH_KEY] = int(sw)  # 세면부 폭 (경계선 정보)
    st.session_state[SHARED_MATERIAL_KEY] = result_kind  # 바닥판 재료 (예: "PVE", "FRP", "GRP")

    st.toast("바닥 계산 결과가 자동 저장되었습니다. (벽판/천장판에서 자동 사용)", icon="✅")

    # ---------------------------
    # 출력(UI) — 단 한 번만!
    # ---------------------------
    # 성공 메시지 및 다음 단계 안내
    st.success("✅ 바닥판 계산이 완료되었습니다!")

    # 다음 단계 안내 카드
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
            <span style="font-size: 24px;">🔄</span>
            <h3 style="margin: 0; color: #0f172a; font-weight: 700;">다음 단계</h3>
        </div>
        <p style="margin: 0 0 12px 36px; color: #475569; line-height: 1.6;">
            바닥판 계산이 완료되었습니다. 이제 <strong>벽판 계산</strong>으로 진행해주세요.
        </p>
        <div style="margin-left: 36px; padding: 12px; background: white; border-radius: 8px; border: 1px solid #cbd5e1;">
            <p style="margin: 0; color: #64748b; font-size: 0.9rem; line-height: 1.5;">
                📍 <strong>권장 순서:</strong> 바닥판 ✅ → 벽판 → 천장판 → 견적서 생성
            </p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 벽판 계산 페이지로 이동 버튼
    col_spacer, col_btn, col_spacer2 = st.columns([1, 2, 1])
    with col_btn:
        st.page_link("pages/2_벽판_계산.py", label="🟩 벽판 계산 시작하기", icon=None)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 2], vertical_alignment="top")

    with left:
        img = draw_bathroom(shape, bw, bl, sw, sl, shw_eff, shl_eff, central, btype)
        st.image(img, caption="욕실 도형(약 1/3 크기)", width=540, output_format="PNG")
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with right:
        st.subheader("선택된 바닥판")
        st.write(f"**재질**: {material_label}")
        st.write(f"**규격**: {floor_spec}")
        st.write(
            f"**단가1/노무비/단가2**: {prices['단가1']:,} / {prices['노무비']:,} / {prices['단가2']:,}"
        )
        st.write(f"**소계(원)**: {base_subtotal:,}")
        st.write(f"**생산관리비({prod_rate_pct:.1f}%)**: 포함")
        st.write(
            f"**영업관리비 포함 소계(원)**: {mgmt_total:,}  (영업관리비율 {sales_rate_pct:.1f}%)"
        )

        st.info("결정 과정", icon="ℹ️")
        st.write("\n".join([f"- {x}" for x in decision_log]))

        st.markdown("---")

        # ====== floor.json 저장 + 다운로드 버튼 ======
        # 완전한 결과 payload (관리비 2단계 포함)
        if result_kind and result_kind.upper() == "PVE":
            # PVE는 직접 계산값 사용
            floor_json_payload = {
                "소재": result_kind,
                "형태": shape,
                "욕실폭": int(bw),
                "욕실길이": int(bl),
                "세면부폭": (None if sw is None else int(sw)),
                "세면부길이": (None if sl is None else int(sl)),
                "샤워부폭": (None if shw_eff is None else int(shw_eff)),
                "샤워부길이": (None if shl_eff is None else int(shl_eff)),
                "원재료비": int(prices.get("단가1", 0)),
                "가공비": int(prices.get("노무비", 0)),
                "소계": int(base_subtotal),
                "생산관리비": int(q.get("생산관리비", 0)),
                "생산관리비포함단가": int(q.get("생산관리비포함", 0)),
                "영업관리비": int(q.get("영업관리비", 0)),
                "영업관리비포함단가": int(mgmt_total),
            }
        else:
            # GRP/FRP 매칭인 경우
            blocks = price_blocks_grp_frp(base_subtotal, r_p, r_s)
            floor_json_payload = {
                "소재": result_kind,
                "형태": shape,
                "욕실폭": int(bw),
                "욕실길이": int(bl),
                "세면부폭": (None if sw is None else int(sw)),
                "세면부길이": (None if sl is None else int(sl)),
                "샤워부폭": (None if shw_eff is None else int(shw_eff)),
                "샤워부길이": (None if shl_eff is None else int(shl_eff)),
                "단가1": int(prices.get("단가1", 0)),
                "노무비": int(prices.get("노무비", 0)),
                "단가2": int(prices.get("단가2", 0)),
                "소계": int(base_subtotal),
                "생산관리비": int(blocks["생산관리비"]),
                "생산관리비포함단가": int(blocks["생산관리비포함"]),
                "영업관리비": int(blocks["영업관리비"]),
                "영업관리비포함단가": int(blocks["영업관리비포함"]),
            }

        # 로컬 파일 저장
        try:
            with open("floor.json", "w", encoding="utf-8") as f:
                json.dump(floor_json_payload, f, ensure_ascii=False, indent=2)
            st.success("floor.json 파일이 저장되었습니다.")
        except Exception as e:
            st.error(f"floor.json 저장 실패: {e}")

        # 다운로드 버튼
        json_bytes = json.dumps(floor_json_payload, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button(
            label="📥 floor.json 다운로드",
            data=json_bytes,
            file_name="floor.json",
            mime="application/json",
            type="secondary",
        )

        # JSON 미리보기
        with st.expander("저장된 JSON 미리보기"):
            st.json(floor_json_payload)
