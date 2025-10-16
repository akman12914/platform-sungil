# 욕실 견적서 생성기
# session_state 연동 버전 - 바닥/벽/천장 계산 결과를 자동으로 가져옵니다.

import json
import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd
import streamlit as st

# Session state keys
FLOOR_RESULT_KEY = "floor_result"
WALL_RESULT_KEY = "wall_result"
CEIL_RESULT_KEY = "ceil_result"

# ----------------------------
# Dark Sidebar Styling
# ----------------------------
def _design_refresh():
    st.markdown(
        """
    <style>
      :root{
        --sb-bg:#0b1220;
        --sb-fg:#e2e8f0;
        --sb-muted:#cbd5e1;
        --sb-line:#1f2a44;
        --accent:#f1f5f9;
        --accent-2:#cbd5e1;
        --ink:#0f172a;
        --muted:#475569;
        --line:#e2e8f0;
      }
      section[data-testid="stSidebar"]{
        background:var(--sb-bg)!important;
        color:var(--sb-fg)!important;
        border-right:1px solid var(--sb-line);
      }
      section[data-testid="stSidebar"] *{ color:var(--sb-fg)!important; }
      section[data-testid="stSidebar"] .stMarkdown p,
      section[data-testid="stSidebar"] label{
        color:var(--sb-muted)!important;
        font-weight:600!important;
      }
      [data-testid="stAppViewContainer"] .stButton>button{
        background:linear-gradient(180deg,var(--accent),var(--accent-2))!important;
        color:#001018!important;
        border:0!important;
        font-weight:800!important;
        letter-spacing:.2px;
      }
      [data-testid="stAppViewContainer"] .stButton>button:hover{
        filter:brightness(1.05);
      }
    </style>
    """,
        unsafe_allow_html=True,
    )

_design_refresh()

st.set_page_config(page_title="욕실 견적서 생성기", layout="wide")

# ----------------------------
# Helper Functions
# ----------------------------
REQ_COLUMNS = ["품목", "분류", "사양 및 규격", "단가", "수량"]

@st.cache_data(show_spinner=False)
def load_pricebook_from_excel(file_bytes: bytes, sheet_name: str = "자재단가내역") -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    # Normalize columns
    colmap = {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2 in ["품목","폼목"]: colmap[c] = "품목"
        elif c2 in ["분류"]: colmap[c] = "분류"
        elif c2 in ["사양 및 규격", "사양","규격"]: colmap[c] = "사양 및 규격"
        elif c2 in ["단가"]: colmap[c] = "단가"
        elif c2 in ["수량"]: colmap[c] = "수량"
        elif c2 in ["금액"]: colmap[c] = "금액"
    df = df.rename(columns=colmap)
    # Ensure required columns exist
    for c in ["품목","분류","사양 및 규격","단가","수량"]:
        if c not in df.columns:
            df[c] = None
    # Clean values
    for c in ["품목","분류","사양 및 규격"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["단가","수량"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "금액" not in df.columns:
        df["금액"] = df["단가"].fillna(0) * df["수량"].fillna(0)
    return df

def find_item(df: pd.DataFrame, 품목: str, 분류: Optional[str]=None, spec_contains: Optional[str]=None) -> Optional[pd.Series]:
    q = (df["품목"] == 품목)
    if 분류 is not None:
        q &= (df["분류"] == 분류)
    if spec_contains:
        q &= df["사양 및 규격"].str.contains(str(spec_contains), case=False, na=False)
    candidates = df[q]
    if len(candidates) == 0:
        return None
    # If multiple, prefer exact spec match first
    if spec_contains:
        exact = candidates[candidates["사양 및 규격"].str.strip() == spec_contains]
        if len(exact) == 1:
            return exact.iloc[0]
    return candidates.iloc[0]

def add_row(rows: List[Dict[str,Any]], 품목: str, spec: str, qty: float, unit_price: Optional[float]) -> None:
    unit_price = unit_price if unit_price is not None else 0
    amount = (qty or 0) * (unit_price or 0)
    rows.append({"품목": 품목, "사양 및 규격": spec, "수량": qty, "단가": unit_price, "금액": amount})

def add_all_by_category(rows: List[Dict[str,Any]], df: pd.DataFrame, 품목: str, 분류: str):
    sub = df[(df["품목"]==품목) & (df["분류"]==분류)]
    for _, r in sub.iterrows():
        add_row(rows, 품목, str(r["사양 및 규격"]), r["수량"] if pd.notna(r["수량"]) else 1, r["단가"] if pd.notna(r["단가"]) else 0)

# ----------------------------
# Convert session_state to quotation format
# ----------------------------
def convert_floor_data(floor_result: dict) -> dict:
    """Convert floor_result to quotation format"""
    if not floor_result:
        return {}

    material = floor_result.get("material", "")
    # "PP/PE 바닥판" -> "PP/PE" 변환
    material_clean = material.replace(" 바닥판", "").replace("바닥판", "").strip()

    # 가격 정보 추출
    prices = floor_result.get("prices", {})
    단가 = prices.get("단가2", 0) or prices.get("단가1", 0)

    return {
        "재질": material_clean,
        "규격": floor_result.get("spec", ""),
        "수량": floor_result.get("qty", 1),
        "단가": 단가,
        "주거약자": floor_result.get("meta", {}).get("inputs", {}).get("pve_kind", "") == "주거약자 (+480mm)"
    }

def convert_wall_data(wall_result: dict) -> dict:
    """Convert wall_result to quotation format"""
    if not wall_result:
        return {}

    result = wall_result.get("result", {})
    counts = result.get("counts", {})
    inputs = wall_result.get("inputs", {})

    return {
        "총개수": counts.get("n_panels", 0),
        "단가": 0,  # 단가표에서 찾을 예정
        "벽타일": inputs.get("tile", "300×600")
    }

def convert_ceiling_data(ceil_result: dict) -> dict:
    """Convert ceil_result to quotation format"""
    if not ceil_result:
        return {}

    result = ceil_result.get("result", {})
    detail_best = result.get("detail_best", {})
    oriented = detail_best.get("oriented", [])

    # 바디판넬과 사이드판넬 분리
    body_panels = [p for p in oriented if p.get("kind") == "B"]
    side_panels = [p for p in oriented if p.get("kind") == "S"]

    body_info = {}
    if body_panels:
        body_info = {
            "개수": len(body_panels),
            "종류": body_panels[0].get("name", "")
        }

    side_info = {}
    if side_panels:
        side_info = {
            "개수": len(side_panels),
            "종류": side_panels[0].get("name", "")
        }

    return {
        "재질": "ABS",  # 기본값, 실제로는 panel 종류에서 판단해야 함
        "총개수": len(oriented),
        "바디판넬": body_info,
        "사이드판넬": side_info,
        "천공구": 1,  # 기본값
        "단가": detail_best.get("material_cost", 0) / max(len(oriented), 1)
    }

# ----------------------------
# UI
# ----------------------------
st.title("🛁 욕실 견적서 생성기")

# Check for calculation results
floor_result = st.session_state.get(FLOOR_RESULT_KEY)
wall_result = st.session_state.get(WALL_RESULT_KEY)
ceil_result = st.session_state.get(CEIL_RESULT_KEY)

has_floor = bool(floor_result)
has_wall = bool(wall_result)
has_ceil = bool(ceil_result)

# Status display
st.markdown("### 계산 결과 상태")
col1, col2, col3 = st.columns(3)
with col1:
    status = "✅ 완료" if has_floor else "❌ 미완료"
    st.metric("바닥판", status)
with col2:
    status = "✅ 완료" if has_wall else "❌ 미완료"
    st.metric("벽판", status)
with col3:
    status = "✅ 완료" if has_ceil else "❌ 미완료"
    st.metric("천장판", status)

if not (has_floor and has_wall and has_ceil):
    st.warning("⚠️ 바닥판, 벽판, 천장판 계산을 모두 완료한 후 견적서를 생성할 수 있습니다.")
    st.info("왼쪽 사이드바에서 각 계산 페이지로 이동하여 계산을 완료하세요.")
    st.stop()

# Convert session_state data
floor_data = convert_floor_data(floor_result)
wall_data = convert_wall_data(wall_result)
ceiling_data = convert_ceiling_data(ceil_result)

# Sidebar: Pricebook upload
with st.sidebar:
    st.markdown("### ① 단가표 업로드")
    pricebook_file = st.file_uploader("Sungil_DB2_new.xlsx (시트명: 자재단가내역)", type=["xlsx"])

    st.markdown("---")
    st.markdown("### ② 계산 결과 (자동 연동)")
    st.success(f"✅ 바닥판: {floor_data.get('재질', 'N/A')}")
    st.success(f"✅ 벽판: {wall_data.get('총개수', 0)}장")
    st.success(f"✅ 천장판: {ceiling_data.get('총개수', 0)}장")

    st.markdown("---")
    st.markdown("### ③ 옵션 선택")

# Load pricebook
price_df: Optional[pd.DataFrame] = None
if pricebook_file is not None:
    try:
        price_df = load_pricebook_from_excel(pricebook_file.read())
        st.sidebar.success(f"단가표 로드 완료: {len(price_df)}행")
    except Exception as e:
        st.sidebar.error(f"단가표 로드 실패: {e}")

# ----------------------------
# UI: 단일/다중 선택 그룹
# ----------------------------
single_choice_specs = {
    "냉온수배관": ["PB 독립배관","PB 세대 세트 배관","PB+이중관(오픈수전함)"],
    "문틀규격": ["110m/m","130m/m","140m/m","155m/m","175m/m","195m/m","210m/m","230m/m"],
    "도기류(세면기/수전)": ["긴다리 세면기 수전(원홀)","긴다리 세면샤워 겸용수전(원홀)","반다리 세면기 수전(원홀)","반다리 세면샤워 겸용수전(원홀)"],
    "도기류(변기)": ["양변기 투피스","양변기 준피스"],
    "은경": ["있음","없음"],
    "욕실장": ["PS장(600*900)","슬라이딩 욕실장"],
    "칸막이": ["샤워부스","샤워파티션"],
    "욕조": ["SQ욕조","세라믹 욕조"],
    "환기류": ["환풍기","후렉시블 호스, 서스밴드"],
}

multi_choice_specs = {
    "문세트": ["PVC 4방틀 (130 ~ 230바)","ABS 문짝","도어락","경첩","도어스토퍼"],
    "액세서리": ["수건걸이","휴지걸이","매립형 휴지걸이","코너선반","일자 유리선반","청소솔","2단 수건선반"],
    "수전": ["샤워수전","슬라이드바","레인 샤워수전","선반형 레인 샤워수전","청소건","세탁기 수전"],
    "욕실등": ["천장 매립등(사각)","천장 매립등(원형)","벽부등"],
}

with st.expander("단일 선택 (Radio)", expanded=True):
    single_selections = {}
    for group, options in single_choice_specs.items():
        single_selections[group] = st.radio(group, options, horizontal=True, index=0)

with st.expander("다중 선택 (Checkbox)", expanded=True):
    multi_selections = {}
    for group, options in multi_choice_specs.items():
        picked = []
        cols = st.columns(min(4, len(options)))
        for i, opt in enumerate(options):
            with cols[i % len(cols)]:
                if st.checkbox(f"{group}: {opt}"):
                    picked.append(opt)
        multi_selections[group] = picked

# ----------------------------
# 견적서 생성
# ----------------------------
rows: List[Dict[str,Any]] = []
warnings: List[str] = []

if price_df is None:
    st.warning("단가표(엑셀)를 먼저 업로드하세요.")
else:
    # 1) 바닥판
    if floor_data:
        material = str(floor_data.get("재질","")).upper()
        spec_text = str(floor_data.get("규격","")).strip()
        qty = float(floor_data.get("수량", 1))
        unit_price = float(floor_data.get("단가", 0))
        senior = bool(floor_data.get("주거약자", False))

        # 품목 '바닥판' 본체
        add_row(rows, "바닥판", material, qty, unit_price)

        # 부재료 자동 포함
        if material in ["GRP","SMC/FRP","PP/PE","PVE"]:
            if material == "PVE":
                분류 = "PP/PE 부재료"
            elif material == "SMC/FRP":
                분류 = "SMC/FRP 부재료"
            elif material == "PP/PE":
                분류 = "PP/PE 부재료"
            else:
                분류 = "GRP부재료"
            add_all_by_category(rows, price_df, "바닥판", 분류)
        else:
            warnings.append(f"바닥판 재질 '{material}'에 대한 분류 매핑을 찾을 수 없습니다.")

        # 주거약자 추가
        if senior:
            for spec in ["매립형 휴지걸이(비상폰)","L형 손잡이","ㅡ형 손잡이","접의식 의자"]:
                rec = find_item(price_df, "액세서리", "주거약자", spec_contains=spec)
                if rec is not None:
                    add_row(rows, "액세서리", spec, rec.get("수량",1) or 1, rec.get("단가",0))
                else:
                    add_row(rows, "액세서리", spec, 1, 0)
                    warnings.append(f"주거약자 '{spec}' 단가 미발견 → 0 처리")

    # 2) 벽판 & 타일
    if wall_data:
        # PU벽판
        wall_spec = "PU벽판"
        rec = find_item(price_df, "벽판", "PU타일 벽체", spec_contains="PU벽판")
        qty = float(wall_data.get("총개수", 0))
        unit_price = None
        if rec is not None:
            unit_price = rec.get("단가", None)
        else:
            unit_price = float(wall_data.get("단가", 0))
            warnings.append("벽판(PU벽판) 단가를 엑셀에서 찾지 못해 기본값 0으로 설정했습니다.")
        add_row(rows, "벽판", wall_spec, qty, unit_price)

        # 벽타일 & 바닥타일 규격 연동
        tile_str = str(wall_data.get("벽타일","")).replace("×","x").replace(" ", "")
        wall_tile_spec = None
        if tile_str in ["250x400","250*400"]:
            wall_tile_spec = "벽타일 250*400"
            floor_tile_spec = "바닥타일 200*200"
        else:
            wall_tile_spec = "벽타일 300*600"
            floor_tile_spec = "바닥타일 300*300"

        # 벽타일
        rec = find_item(price_df, "타일", "PU타일 벽체 타일", spec_contains=wall_tile_spec)
        if rec is not None:
            add_row(rows, "타일", wall_tile_spec, rec.get("수량",1) or 1, rec.get("단가",0))
        else:
            add_row(rows, "타일", wall_tile_spec, 1, 0)
            warnings.append(f"'{wall_tile_spec}' 단가 미발견 → 0 처리")

        # 바닥타일
        rec = find_item(price_df, "타일", "바닥타일", spec_contains=floor_tile_spec.split()[-1])
        if rec is None:
            rec = find_item(price_df, "타일", "바닥타일", spec_contains=floor_tile_spec)
        if rec is not None:
            add_row(rows, "타일", floor_tile_spec, rec.get("수량",1) or 1, rec.get("단가",0))
        else:
            add_row(rows, "타일", floor_tile_spec, 1, 0)
            warnings.append(f"'{floor_tile_spec}' 단가 미발견 → 0 처리")

    # 3) 천장판
    if ceiling_data:
        material = str(ceiling_data.get("재질","")).upper()
        body = ceiling_data.get("바디판넬", {}) or {}
        side = ceiling_data.get("사이드판넬", {}) or {}
        total_cnt = float(ceiling_data.get("총개수", 0))
        hole_cnt = float(ceiling_data.get("천공구", 0))

        # 메인 판
        if material == "ABS":
            rec = find_item(price_df, "천장판", None, spec_contains="ABS천장판")
            add_row(rows, "천장판", "ABS천장판", total_cnt or (body.get("개수",0)+side.get("개수",0)), rec.get("단가",0) if rec is not None else 0)
            if rec is None:
                warnings.append("ABS천장판 단가 미발견 → 0 처리")
        elif material == "GRP":
            rec = find_item(price_df, "천장판", None, spec_contains="GRP천장판")
            add_row(rows, "천장판", "GRP천장판", total_cnt or (body.get("개수",0)+side.get("개수",0)), rec.get("단가",0) if rec is not None else 0)
            if rec is None:
                warnings.append("GRP천장판 단가 미발견 → 0 처리")
        else:
            add_row(rows, "천장판", material, total_cnt, 0)
            warnings.append(f"천장판 재질 '{material}' 단가 미발견 → 0 처리")

        # 세부 수량 표기 (정보용)
        if body.get("개수",0):
            add_row(rows, "천장판", f"바디판넬 ({body.get('종류','')})", float(body.get("개수",0)), float(ceiling_data.get("단가",0)))
        if side.get("개수",0):
            add_row(rows, "천장판", f"사이드판넬 ({side.get('종류','')})", float(side.get("개수",0)), float(ceiling_data.get("단가",0)))
        if hole_cnt:
            add_row(rows, "천장판", "천공구", hole_cnt, 0)

    # 4) 단일 선택 그룹 반영
    for group, spec in single_selections.items():
        if group == "은경" and spec == "없음":
            continue
        품목 = group.split("(")[0]
        rec = find_item(price_df, 품목, None, spec_contains=spec)
        if rec is None:
            alt_map = {
                "도기류(세면기/수전)": ("도기류", None),
                "도기류(변기)": ("도기류", None),
            }
            if group in alt_map:
                품목2, 분류2 = alt_map[group]
                rec = find_item(price_df, 품목2, 분류2, spec_contains=spec)
                품목 = 품목2
        if rec is not None:
            add_row(rows, 품목, spec, rec.get("수량",1) or 1, rec.get("단가",0))
        else:
            add_row(rows, 품목, spec, 1, 0)
            warnings.append(f"[단일선택] '{group} - {spec}' 단가 미발견 → 0 처리")

    # 5) 다중 선택 그룹 반영
    for group, specs in multi_selections.items():
        for spec in specs:
            rec = find_item(price_df, group, None, spec_contains=spec)
            if rec is None:
                alt_map = {
                    "문세트": "문세트",
                    "액세서리": "액세서리",
                    "수전": "수전",
                    "욕실등": "욕실등",
                }
                품목2 = alt_map.get(group, group)
                rec = find_item(price_df, 품목2, None, spec_contains=spec)
                if rec is None:
                    add_row(rows, 품목2, spec, 1, 0)
                    warnings.append(f"[다중선택] '{group} - {spec}' 단가 미발견 → 0 처리")
                    continue
                add_row(rows, 품목2, spec, rec.get("수량",1) or 1, rec.get("단가",0))
            else:
                add_row(rows, group, spec, rec.get("수량",1) or 1, rec.get("단가",0))

    # 6) 공통자재 전부 포함
    commons = price_df[price_df["품목"]=="공통자재"]
    for _, r in commons.iterrows():
        add_row(rows, "공통자재", str(r["사양 및 규격"]), r["수량"] if pd.notna(r["수량"]) else 1, r["단가"] if pd.notna(r["단가"]) else 0)

# ----------------------------
# 결과 표
# ----------------------------
if rows:
    est_df = pd.DataFrame(rows, columns=["품목","사양 및 규격","수량","단가","금액"])
    est_df["수량"] = pd.to_numeric(est_df["수량"], errors="coerce").fillna(0).astype(float)
    est_df["단가"] = pd.to_numeric(est_df["단가"], errors="coerce").fillna(0).astype(float)
    est_df["금액"] = (est_df["수량"] * est_df["단가"]).round(0)

    st.subheader("견적서 미리보기")
    st.dataframe(est_df, use_container_width=True)

    totals = est_df.groupby("품목", dropna=False)["금액"].sum().reset_index().sort_values("금액", ascending=False)
    st.markdown("#### 품목별 합계")
    st.dataframe(totals, use_container_width=True)

    grand_total = est_df["금액"].sum()
    st.metric("총 금액", f"{grand_total:,.0f} 원")

    # Excel 다운로드
    @st.cache_data(show_spinner=False)
    def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="견적서")
        return output.getvalue()

    xlsx_bytes = df_to_excel_bytes(est_df)
    st.download_button(
        "📥 견적서 Excel 다운로드",
        data=xlsx_bytes,
        file_name=f"estimate_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if warnings:
    with st.expander("⚠️ 경고/참고", expanded=False):
        for w in warnings:
            st.warning(w)
