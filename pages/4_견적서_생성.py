# 욕실 견적서 생성기
# session_state 연동 버전 - 바닥/벽/천장 계산 결과를 자동으로 가져옵니다.

from common_styles import apply_common_styles, set_page_config
import auth

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
SAVED_QUOTATIONS_KEY = "saved_quotations"  # 저장된 세대 타입별 견적 목록 (최대 10개)

set_page_config(page_title="욕실 견적서 생성기", layout="wide")
apply_common_styles()

auth.require_auth()

# ----------------------------
# Helper Functions
# ----------------------------
REQ_COLUMNS = ["품목", "분류", "사양 및 규격", "단가", "수량"]


@st.cache_data(show_spinner=False)
def load_pricebook_from_excel(
    file_bytes: bytes, sheet_name: str = "자재단가내역"
) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    # Normalize columns
    colmap = {}
    for c in df.columns:
        c2 = str(c).strip()
        if c2 in ["품목", "폼목"]:
            colmap[c] = "품목"
        elif c2 in ["분류"]:
            colmap[c] = "분류"
        elif c2 in ["사양 및 규격", "사양", "규격"]:
            colmap[c] = "사양 및 규격"
        elif c2 in ["단가"]:
            colmap[c] = "단가"
        elif c2 in ["수량"]:
            colmap[c] = "수량"
        elif c2 in ["금액"]:
            colmap[c] = "금액"
    df = df.rename(columns=colmap)
    # Ensure required columns exist
    for c in ["품목", "분류", "사양 및 규격", "단가", "수량"]:
        if c not in df.columns:
            df[c] = None
    # Clean values
    for c in ["품목", "분류", "사양 및 규격"]:
        df[c] = df[c].astype(str).str.strip()
    for c in ["단가", "수량"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "금액" not in df.columns:
        df["금액"] = df["단가"].fillna(0) * df["수량"].fillna(0)
    return df


def find_item(
    df: pd.DataFrame,
    품목: str,
    분류: Optional[str] = None,
    spec_contains: Optional[str] = None,
) -> Optional[pd.Series]:
    q = df["품목"] == 품목
    if 분류 is not None:
        q &= df["분류"] == 분류
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


def add_row(
    rows: List[Dict[str, Any]],
    품목: str,
    spec: str,
    qty: float,
    unit_price: Optional[float],
) -> None:
    unit_price = unit_price if unit_price is not None else 0
    amount = (qty or 0) * (unit_price or 0)
    rows.append(
        {
            "품목": 품목,
            "사양 및 규격": spec,
            "수량": qty,
            "단가": unit_price,
            "금액": amount,
        }
    )


def add_all_by_category(
    rows: List[Dict[str, Any]], df: pd.DataFrame, 품목: str, 분류: str
):
    sub = df[(df["품목"] == 품목) & (df["분류"] == 분류)]
    for _, r in sub.iterrows():
        add_row(
            rows,
            품목,
            str(r["사양 및 규격"]),
            r["수량"] if pd.notna(r["수량"]) else 1,
            r["단가"] if pd.notna(r["단가"]) else 0,
        )


# ----------------------------
# Convert session_state to quotation format
# ----------------------------
def convert_floor_data(floor_result: dict) -> dict:
    """Convert floor_result to quotation format"""
    if not floor_result:
        return {}

    # session_state 구조: {"section", "inputs", "result", "decision_log"}
    inputs = floor_result.get("inputs", {})
    result = floor_result.get("result", {})

    # 소재 정보 추출 (result에서)
    material = result.get("소재", "")
    # "PP/PE 바닥판" -> "PP/PE" 변환
    material_clean = material.replace(" 바닥판", "").replace("바닥판", "").strip()

    # 가격 정보 추출 (result에서)
    단가 = result.get("영업관리비포함단가", 0) or result.get("소계", 0)

    # 세대수 정보 (inputs에서)
    units = inputs.get("units", 1)

    # 규격 문자열 생성
    W = inputs.get("W", 0)
    L = inputs.get("L", 0)
    spec = f"{W}×{L}" if W and L else ""

    return {
        "재질": material_clean,
        "규격": spec,
        "수량": units,  # 세대수 = 수량
        "단가": 단가,
        "주거약자": inputs.get("user_type", "") == "주거약자",
        "inputs": inputs,  # inputs 정보 유지 (세대수 등)
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
        "벽타일": inputs.get("tile", "300×600"),
    }


def convert_ceiling_data(ceil_result: dict) -> dict:
    """Convert ceil_result to quotation format"""
    if not ceil_result:
        return {}

    # ceil_panel_final.py의 session_state 구조에 맞춰 파싱
    inputs = ceil_result.get("inputs", {})
    result = ceil_result.get("result", {})

    # 재질 정보 추출 (inputs에서)
    material = inputs.get("material", "GRP")  # GRP/FRP/기타

    # JSON export 데이터 사용 (이미 변환된 포맷)
    json_export = result.get("json_export", {})
    if json_export:
        # 점검구가 딕셔너리 형태인 경우 개수만 추출
        jgm = json_export.get("점검구", 1)
        hole_count = jgm.get("개수", 1) if isinstance(jgm, dict) else jgm

        return {
            "재질": json_export.get("재질", material),
            "총개수": json_export.get("총개수", 0),
            "바디판넬": json_export.get("바디판넬", {}),
            "사이드판넬": json_export.get("사이드판넬", {}),
            "천공구": hole_count,
            "단가": json_export.get("단가", 0),
        }

    # Fallback: summary 데이터에서 추출
    summary = result.get("summary", {})
    elements = result.get("elements", [])

    # 바디/사이드 개수 카운트
    body_cnt = sum(1 for e in elements if e.get("kind") == "BODY")
    side_cnt = sum(1 for e in elements if e.get("kind") == "SIDE")

    # 대표 모델명 추출
    body_models = [e.get("model", "") for e in elements if e.get("kind") == "BODY"]
    side_models = [e.get("model", "") for e in elements if e.get("kind") == "SIDE"]

    body_info = {}
    if body_models:
        # 가장 많이 나온 모델
        from collections import Counter
        body_top = Counter(body_models).most_common(1)
        if body_top:
            body_info = {"종류": body_top[0][0].replace("(rot)", ""), "개수": body_cnt}

    side_info = {}
    if side_models:
        from collections import Counter
        side_top = Counter(side_models).most_common(1)
        if side_top:
            side_info = {"종류": side_top[0][0].replace("(rot)", ""), "개수": side_cnt}

    total_cnt = summary.get("총판넬수", body_cnt + side_cnt)
    total_price = summary.get("총단가합계", 0)

    return {
        "재질": material,
        "총개수": int(total_cnt),
        "바디판넬": body_info,
        "사이드판넬": side_info,
        "천공구": 1,  # 기본값, json_export 없으면 1로 가정
        "단가": int(total_price),
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
col1, col2, col3, col4 = st.columns(4)
with col1:
    status = "✅ 완료" if has_floor else "❌ 미완료"
    st.metric("바닥판", status)
with col2:
    status = "✅ 완료" if has_wall else "❌ 미완료"
    st.metric("벽판", status)
with col3:
    status = "✅ 완료" if has_ceil else "❌ 미완료"
    st.metric("천장판", status)
with col4:
    # 바닥판 세대수 표시
    units_display = 1
    if floor_result:
        inputs = floor_result.get("inputs", {})
        units_display = int(inputs.get("units", 1))
    st.metric("공사 세대수", f"{units_display}세대")

# ========== 바닥판, 벽판, 천장판 계산 의존성 체크 ==========
missing_steps = []
if not has_floor:
    missing_steps.append("🟦 바닥판 계산")
if not has_wall:
    missing_steps.append("🟩 벽판 계산")
if not has_ceil:
    missing_steps.append("🟨 천장판 계산")

if missing_steps:
    st.warning(
        f"⚠️ 견적서를 생성하려면 먼저 **{', '.join(missing_steps)}**을(를) 완료해야 합니다."
    )

    # 안내 카드
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
            <span style="font-size: 24px;">📋</span>
            <h3 style="margin: 0; color: #0f172a; font-weight: 700;">계산 순서 안내</h3>
        </div>
        <p style="margin: 0 0 12px 36px; color: #78350f; line-height: 1.6;">
            견적서 생성은 모든 계산이 완료된 후 진행할 수 있습니다:
        </p>
        <div style="margin-left: 36px; padding: 12px; background: white; border-radius: 8px; border: 1px solid #f59e0b;">
            <p style="margin: 0; color: #92400e; font-size: 0.95rem; line-height: 1.6;">
                <strong>1단계:</strong> 🟦 바닥판 계산"""
        + (" ← <em style='color:#dc2626;'>미완료</em>" if not has_floor else " ✅")
        + """<br>
                <strong>2단계:</strong> 🟩 벽판 계산"""
        + (" ← <em style='color:#dc2626;'>미완료</em>" if not has_wall else " ✅")
        + """<br>
                <strong>3단계:</strong> 🟨 천장판 계산"""
        + (" ← <em style='color:#dc2626;'>미완료</em>" if not has_ceil else " ✅")
        + """<br>
                <strong>4단계:</strong> 📋 견적서 생성 ← <em>현재 페이지</em>
            </p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 미완료 단계로 이동하는 버튼
    col_spacer, col_btn, col_spacer2 = st.columns([1, 2, 1])
    with col_btn:
        if not has_floor:
            st.page_link(
                "pages/1_바닥판_계산.py", label="🟦 바닥판 계산 시작하기", icon=None
            )
        elif not has_wall:
            st.page_link(
                "pages/2_벽판_계산.py", label="🟩 벽판 계산 시작하기", icon=None
            )
        elif not has_ceil:
            st.page_link(
                "pages/3_천장판_계산.py", label="🟨 천장판 계산 시작하기", icon=None
            )

    st.stop()  # 이전 단계 미완료 시 이후 UI 차단

# 모든 단계 완료 시 성공 메시지
st.success("✅ 모든 계산이 완료되었습니다. 견적서를 생성할 수 있습니다.")

# Convert session_state data
floor_data = convert_floor_data(floor_result)
wall_data = convert_wall_data(wall_result)
ceiling_data = convert_ceiling_data(ceil_result)

# Sidebar: Pricebook upload
with st.sidebar:
    st.markdown("### ① 단가표 업로드")
    pricebook_file = st.file_uploader(
        "Sungil_DB2_new.xlsx (시트명: 자재단가내역)", type=["xlsx"]
    )

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
    "냉온수배관": ["선택안함", "PB 독립배관", "PB 세대 세트 배관", "PB+이중관(오픈수전함)"],
    "문틀규격": [
        "선택안함",
        "110m/m",
        "130m/m",
        "140m/m",
        "155m/m",
        "175m/m",
        "195m/m",
        "210m/m",
        "230m/m",
    ],
    "도기류(세면기/수전)": [
        "선택안함",
        "긴다리 세면기 수전(원홀)",
        "긴다리 세면샤워 겸용수전(원홀)",
        "반다리 세면기 수전(원홀)",
        "반다리 세면샤워 겸용수전(원홀)",
    ],
    "도기류(변기)": ["선택안함", "양변기 투피스", "양변기 준피스"],
    "은경": ["선택안함", "있음", "없음"],
    "욕실장": ["선택안함", "PS장(600*900)", "슬라이딩 욕실장"],
    "칸막이": ["선택안함", "샤워부스", "샤워파티션"],
    "욕조": ["선택안함", "SQ욕조", "세라믹 욕조"],
    "환기류": ["선택안함", "환풍기", "후렉시블 호스, 서스밴드"],
}

multi_choice_specs = {
    "문세트": ["PVC 4방틀 (130 ~ 230바)", "ABS 문짝", "도어락", "경첩", "도어스토퍼"],
    "액세서리": [
        "수건걸이",
        "휴지걸이",
        "매립형 휴지걸이",
        "코너선반",
        "일자 유리선반",
        "청소솔",
        "2단 수건선반",
    ],
    "수전": [
        "샤워수전",
        "슬라이드바",
        "레인 샤워수전",
        "선반형 레인 샤워수전",
        "청소건",
        "세탁기 수전",
    ],
    "욕실등": ["천장 매립등(사각)", "천장 매립등(원형)", "벽부등"],
    "공통자재": [
        "욕실등, 콘센트 내함",
        "휴지걸이 내함",
        "실리콘(내항균성)",
        "실리콘(외장용)",
        "코너비드",
        "코너마감재",
        "도어실(문지방)",
        "젠다이상판",
        "타일 평탄클립",
        "우레탄폼",
        "PVC보온재",
        "바닥타일 보양",
        "바닥타일 보양테이프",
        "이면지지클립",
        "슬리브 방수액",
        "재료분리대(SUS)",
    ],
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
rows: List[Dict[str, Any]] = []
warnings: List[str] = []

if price_df is None:
    st.warning("단가표(엑셀)를 먼저 업로드하세요.")
else:
    # 1) 바닥판
    if floor_data:
        material = str(floor_data.get("재질", "")).upper()
        spec_text = str(floor_data.get("규격", "")).strip()
        qty = float(floor_data.get("수량", 1))
        unit_price = float(floor_data.get("단가", 0))
        senior = bool(floor_data.get("주거약자", False))

        # 품목 '바닥판' 본체
        add_row(rows, "바닥판", material, qty, unit_price)

        # 부재료 자동 포함
        if material in ["GRP", "SMC/FRP", "PP/PE", "PVE"]:
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
            warnings.append(
                f"바닥판 재질 '{material}'에 대한 분류 매핑을 찾을 수 없습니다."
            )

        # 주거약자 추가
        if senior:
            for spec in [
                "매립형 휴지걸이(비상폰)",
                "L형 손잡이",
                "ㅡ형 손잡이",
                "접의식 의자",
            ]:
                rec = find_item(price_df, "액세서리", "주거약자", spec_contains=spec)
                if rec is not None:
                    add_row(
                        rows,
                        "액세서리",
                        spec,
                        rec.get("수량", 1) or 1,
                        rec.get("단가", 0),
                    )
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
            warnings.append(
                "벽판(PU벽판) 단가를 엑셀에서 찾지 못해 기본값 0으로 설정했습니다."
            )
        add_row(rows, "벽판", wall_spec, qty, unit_price)

        # 벽타일 & 바닥타일 규격 연동
        tile_str = str(wall_data.get("벽타일", "")).replace("×", "x").replace(" ", "")
        wall_tile_spec = None
        if tile_str in ["250x400", "250*400"]:
            wall_tile_spec = "벽타일 250*400"
            floor_tile_spec = "바닥타일 200*200"
        else:
            wall_tile_spec = "벽타일 300*600"
            floor_tile_spec = "바닥타일 300*300"

        # 벽타일
        rec = find_item(
            price_df, "타일", "PU타일 벽체 타일", spec_contains=wall_tile_spec
        )
        if rec is not None:
            add_row(
                rows,
                "타일",
                wall_tile_spec,
                rec.get("수량", 1) or 1,
                rec.get("단가", 0),
            )
        else:
            add_row(rows, "타일", wall_tile_spec, 1, 0)
            warnings.append(f"'{wall_tile_spec}' 단가 미발견 → 0 처리")

        # 바닥타일
        rec = find_item(
            price_df, "타일", "바닥타일", spec_contains=floor_tile_spec.split()[-1]
        )
        if rec is None:
            rec = find_item(price_df, "타일", "바닥타일", spec_contains=floor_tile_spec)
        if rec is not None:
            add_row(
                rows,
                "타일",
                floor_tile_spec,
                rec.get("수량", 1) or 1,
                rec.get("단가", 0),
            )
        else:
            add_row(rows, "타일", floor_tile_spec, 1, 0)
            warnings.append(f"'{floor_tile_spec}' 단가 미발견 → 0 처리")

    # 3) 천장판
    if ceiling_data:
        material = str(ceiling_data.get("재질", "")).upper()
        body = ceiling_data.get("바디판넬", {}) or {}
        side = ceiling_data.get("사이드판넬", {}) or {}
        total_cnt = float(ceiling_data.get("총개수", 0))

        # 천공구: 딕셔너리인 경우 개수 필드 추출
        hole_data = ceiling_data.get("천공구", 0)
        hole_cnt = float(hole_data.get("개수", 0) if isinstance(hole_data, dict) else hole_data)

        # 메인 판
        if material == "ABS":
            rec = find_item(price_df, "천장판", None, spec_contains="ABS천장판")
            add_row(
                rows,
                "천장판",
                "ABS천장판",
                total_cnt or (body.get("개수", 0) + side.get("개수", 0)),
                rec.get("단가", 0) if rec is not None else 0,
            )
            if rec is None:
                warnings.append("ABS천장판 단가 미발견 → 0 처리")
        elif material == "GRP":
            rec = find_item(price_df, "천장판", None, spec_contains="GRP천장판")
            add_row(
                rows,
                "천장판",
                "GRP천장판",
                total_cnt or (body.get("개수", 0) + side.get("개수", 0)),
                rec.get("단가", 0) if rec is not None else 0,
            )
            if rec is None:
                warnings.append("GRP천장판 단가 미발견 → 0 처리")
        else:
            add_row(rows, "천장판", material, total_cnt, 0)
            warnings.append(f"천장판 재질 '{material}' 단가 미발견 → 0 처리")

        # 세부 수량 표기 (정보용)
        if body.get("개수", 0):
            add_row(
                rows,
                "천장판",
                f"바디판넬 ({body.get('종류','')})",
                float(body.get("개수", 0)),
                float(ceiling_data.get("단가", 0)),
            )
        if side.get("개수", 0):
            add_row(
                rows,
                "천장판",
                f"사이드판넬 ({side.get('종류','')})",
                float(side.get("개수", 0)),
                float(ceiling_data.get("단가", 0)),
            )
        if hole_cnt:
            add_row(rows, "천장판", "천공구", hole_cnt, 0)

    # 4) 단일 선택 그룹 반영
    for group, spec in single_selections.items():
        if spec == "선택안함":
            continue
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
            add_row(rows, 품목, spec, rec.get("수량", 1) or 1, rec.get("단가", 0))
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
                    warnings.append(
                        f"[다중선택] '{group} - {spec}' 단가 미발견 → 0 처리"
                    )
                    continue
                add_row(rows, 품목2, spec, rec.get("수량", 1) or 1, rec.get("단가", 0))
            else:
                add_row(rows, group, spec, rec.get("수량", 1) or 1, rec.get("단가", 0))

    # 6) 공통자재는 다중 선택에서 처리됨 (위 5번에서 multi_selections["공통자재"]로 처리)

# ----------------------------
# 결과 표
# ----------------------------
if rows:
    est_df = pd.DataFrame(
        rows, columns=["품목", "사양 및 규격", "수량", "단가", "금액"]
    )
    est_df["수량"] = (
        pd.to_numeric(est_df["수량"], errors="coerce").fillna(0).astype(float)
    )
    est_df["단가"] = (
        pd.to_numeric(est_df["단가"], errors="coerce").fillna(0).astype(float)
    )
    est_df["금액"] = (est_df["수량"] * est_df["단가"]).round(0)

    st.subheader("견적서 미리보기")
    st.dataframe(est_df, use_container_width=True)

    totals = (
        est_df.groupby("품목", dropna=False)["금액"]
        .sum()
        .reset_index()
        .sort_values("금액", ascending=False)
    )
    st.markdown("#### 품목별 합계")
    st.dataframe(totals, use_container_width=True)

    grand_total = est_df["금액"].sum()
    st.metric("총 금액", f"{grand_total:,.0f} 원")

    # ----------------------------
    # 세대 타입 저장 기능
    # ----------------------------
    st.markdown("---")
    st.subheader("세대 타입 저장")

    # 저장된 견적 목록 초기화
    if SAVED_QUOTATIONS_KEY not in st.session_state:
        st.session_state[SAVED_QUOTATIONS_KEY] = []

    # 현재 세대 정보
    current_spec = floor_data.get("규격", "N/A") if floor_data else "N/A"
    current_units = floor_data.get("inputs", {}).get("units", 1) if floor_data else 1

    col_name, col_save = st.columns([3, 1])
    with col_name:
        type_name = st.text_input(
            "세대 타입 이름",
            value=f"타입{len(st.session_state[SAVED_QUOTATIONS_KEY]) + 1}",
            help="예: 21A,B,E/22C,F"
        )
    with col_save:
        st.write("")  # 공백으로 높이 맞춤
        st.write("")
        save_disabled = len(st.session_state[SAVED_QUOTATIONS_KEY]) >= 10
        if st.button(
            "💾 현재 견적 저장",
            disabled=save_disabled,
            help="최대 10개까지 저장 가능"
        ):
            # 현재 견적 데이터 저장
            quotation_data = {
                "name": type_name,
                "spec": current_spec,
                "units": current_units,
                "rows": rows.copy(),  # 견적 항목 목록
                "total": grand_total,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            st.session_state[SAVED_QUOTATIONS_KEY].append(quotation_data)
            st.success(f"✅ '{type_name}' 저장 완료! (규격: {current_spec}, {current_units}세대)")
            st.rerun()

    # 저장된 세대 타입 목록 표시
    saved_list = st.session_state.get(SAVED_QUOTATIONS_KEY, [])
    if saved_list:
        st.markdown("#### 저장된 세대 타입 목록")

        # 테이블 형식으로 표시
        saved_df = pd.DataFrame([
            {
                "번호": i + 1,
                "타입명": q["name"],
                "규격": q["spec"],
                "세대수": q["units"],
                "세대당 단가": f"{q['total']:,.0f}",
                "총 금액": f"{q['total'] * q['units']:,.0f}",
            }
            for i, q in enumerate(saved_list)
        ])
        st.dataframe(saved_df, use_container_width=True, hide_index=True)

        # 삭제 기능
        col_del, col_clear = st.columns([2, 1])
        with col_del:
            if len(saved_list) > 0:
                del_idx = st.selectbox(
                    "삭제할 타입 선택",
                    options=range(len(saved_list)),
                    format_func=lambda x: f"{x+1}. {saved_list[x]['name']} ({saved_list[x]['spec']})"
                )
                if st.button("🗑️ 선택 항목 삭제"):
                    del st.session_state[SAVED_QUOTATIONS_KEY][del_idx]
                    st.success("삭제 완료!")
                    st.rerun()
        with col_clear:
            st.write("")
            if st.button("🗑️ 전체 삭제", type="secondary"):
                st.session_state[SAVED_QUOTATIONS_KEY] = []
                st.success("전체 삭제 완료!")
                st.rerun()

        # 총 세대수 및 총 금액 합계
        total_all_units = sum(q["units"] for q in saved_list)
        total_all_amount = sum(q["total"] * q["units"] for q in saved_list)
        st.markdown(f"**총 세대수: {total_all_units}세대 | 총 금액: {total_all_amount:,.0f}원**")

    st.markdown("---")

    # Excel 다운로드 (LGE 창원 스마트파크 형식)
    def df_to_excel_bytes(df: pd.DataFrame, total_units: int = 1) -> bytes:
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

        wb = Workbook()
        ws = wb.active
        ws.title = "원자재 세대당 단가내역"

        # A4 가로 형식 설정
        ws.page_setup.orientation = ws.ORIENTATION_LANDSCAPE
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.page_setup.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0  # 높이는 자동

        # 가운데 정렬을 위해 왼쪽 여백 컬럼 추가
        LEFT_MARGIN = 3  # 왼쪽 여백 컬럼 수 (더 넓게)

        # 스타일 정의
        title_font = Font(name="맑은 고딕", size=18, bold=True)
        subtitle_font = Font(name="맑은 고딕", size=11, bold=True)
        header_font = Font(name="맑은 고딕", size=10, bold=True)
        data_font = Font(name="맑은 고딕", size=9)
        small_font = Font(name="맑은 고딕", size=8)

        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")

        # 투명 배경 (fill 제거)
        no_fill = PatternFill(fill_type=None)

        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        # 여백 컬럼 설정
        for i in range(1, LEFT_MARGIN + 1):
            ws.column_dimensions[chr(64 + i)].width = 2

        # 실제 시작 컬럼 (C부터)
        START_COL = LEFT_MARGIN + 1

        # 1행: 타이틀 - 가로로 넓게
        title_range = f"{chr(64+START_COL)}1:{chr(64+START_COL+7)}1"
        ws.merge_cells(title_range)
        title_cell = ws.cell(1, START_COL)
        title_cell.value = "욕실 원자재 세대당 단가 내역"
        title_cell.font = title_font
        title_cell.alignment = center_align
        ws.row_dimensions[1].height = 30

        # 2-3행: 빈 행
        ws.row_dimensions[2].height = 10
        ws.row_dimensions[3].height = 10

        # 4행: 세대 정보 및 날짜
        info_range = f"{chr(64+START_COL)}4:{chr(64+START_COL+2)}4"
        ws.merge_cells(info_range)
        info_cell = ws.cell(4, START_COL)
        info_cell.value = f"총 세대수: {total_units}세대"
        info_cell.font = subtitle_font
        info_cell.alignment = left_align

        date_range = f"{chr(64+START_COL+5)}4:{chr(64+START_COL+7)}4"
        ws.merge_cells(date_range)
        date_cell = ws.cell(4, START_COL + 5)
        date_cell.value = f"작성일: {datetime.now():%Y. %m. %d}"
        date_cell.font = subtitle_font
        date_cell.alignment = right_align

        # 5행: 컬럼 헤더 (단일 세대 타입) - 테두리 추가, 배경 투명
        # 품목 (C5:D5)
        품목_range = f"{chr(64+START_COL)}5:{chr(64+START_COL+1)}5"
        ws.merge_cells(품목_range)
        ws.cell(5, START_COL).value = "품목"
        ws.cell(5, START_COL).font = header_font
        ws.cell(5, START_COL).alignment = center_align
        for i in range(START_COL, START_COL + 2):
            ws.cell(5, i).border = thin_border

        # 세대당 단가 (E5:G5)
        세대당_range = f"{chr(64+START_COL+2)}5:{chr(64+START_COL+4)}5"
        ws.merge_cells(세대당_range)
        ws.cell(5, START_COL + 2).value = "세대당 단가"
        ws.cell(5, START_COL + 2).font = header_font
        ws.cell(5, START_COL + 2).alignment = center_align
        for i in range(START_COL + 2, START_COL + 5):
            ws.cell(5, i).border = thin_border

        # 총 금액 (H5:J5)
        총금액_range = f"{chr(64+START_COL+5)}5:{chr(64+START_COL+7)}5"
        ws.merge_cells(총금액_range)
        ws.cell(5, START_COL + 5).value = f"총 금액 ({total_units}세대)"
        ws.cell(5, START_COL + 5).font = header_font
        ws.cell(5, START_COL + 5).alignment = center_align
        for i in range(START_COL + 5, START_COL + 8):
            ws.cell(5, i).border = thin_border

        # 6행: 세부 컬럼 헤더 (배경 투명)
        headers_6 = [
            "대분류",
            "사양 및 규격",
            "수량",
            "단가",
            "금액",
            "수량",
            "단가",
            "금액",
        ]
        for idx, header_text in enumerate(headers_6):
            cell = ws.cell(6, START_COL + idx)
            cell.value = header_text
            cell.font = header_font
            cell.alignment = center_align
            cell.border = thin_border

        # 컬럼 너비 설정 (가로로 넓게)
        ws.column_dimensions[chr(64 + START_COL)].width = 12  # 대분류
        ws.column_dimensions[chr(64 + START_COL + 1)].width = 38  # 사양 및 규격
        ws.column_dimensions[chr(64 + START_COL + 2)].width = 9  # 수량
        ws.column_dimensions[chr(64 + START_COL + 3)].width = 13  # 단가
        ws.column_dimensions[chr(64 + START_COL + 4)].width = 15  # 금액
        ws.column_dimensions[chr(64 + START_COL + 5)].width = 9  # 수량(총)
        ws.column_dimensions[chr(64 + START_COL + 6)].width = 13  # 단가(총)
        ws.column_dimensions[chr(64 + START_COL + 7)].width = 17  # 금액(총)

        # 데이터 행 작성
        row_num = 7
        current_category = None

        for idx, row_data in df.iterrows():
            품목 = str(row_data["품목"])
            사양 = str(row_data["사양 및 규격"])
            수량 = float(row_data["수량"])
            단가 = float(row_data["단가"])
            금액 = float(row_data["금액"])

            # 대분류 (품목이 바뀔 때만 표시)
            cell_a = ws.cell(row=row_num, column=START_COL)
            if 품목 != current_category:
                cell_a.value = 품목
                current_category = 품목
            else:
                cell_a.value = ""
            cell_a.font = data_font
            cell_a.alignment = left_align
            cell_a.border = thin_border

            # 사양 및 규격
            ws.cell(row=row_num, column=START_COL + 1).value = 사양
            ws.cell(row=row_num, column=START_COL + 1).font = data_font
            ws.cell(row=row_num, column=START_COL + 1).alignment = left_align
            ws.cell(row=row_num, column=START_COL + 1).border = thin_border

            # 세대당 단가 (C-E)
            ws.cell(row=row_num, column=START_COL + 2).value = 수량
            ws.cell(row=row_num, column=START_COL + 2).font = data_font
            ws.cell(row=row_num, column=START_COL + 2).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 2).border = thin_border
            ws.cell(row=row_num, column=START_COL + 2).number_format = "#,##0.##"

            ws.cell(row=row_num, column=START_COL + 3).value = 단가
            ws.cell(row=row_num, column=START_COL + 3).font = data_font
            ws.cell(row=row_num, column=START_COL + 3).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 3).border = thin_border
            ws.cell(row=row_num, column=START_COL + 3).number_format = "#,##0"

            ws.cell(row=row_num, column=START_COL + 4).value = 금액
            ws.cell(row=row_num, column=START_COL + 4).font = data_font
            ws.cell(row=row_num, column=START_COL + 4).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 4).border = thin_border
            ws.cell(row=row_num, column=START_COL + 4).number_format = "#,##0"

            # 총 금액 (F-H) - 세대수 곱하기
            ws.cell(row=row_num, column=START_COL + 5).value = 수량 * total_units
            ws.cell(row=row_num, column=START_COL + 5).font = data_font
            ws.cell(row=row_num, column=START_COL + 5).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 5).border = thin_border
            ws.cell(row=row_num, column=START_COL + 5).number_format = "#,##0.##"

            ws.cell(row=row_num, column=START_COL + 6).value = 단가
            ws.cell(row=row_num, column=START_COL + 6).font = data_font
            ws.cell(row=row_num, column=START_COL + 6).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 6).border = thin_border
            ws.cell(row=row_num, column=START_COL + 6).number_format = "#,##0"

            ws.cell(row=row_num, column=START_COL + 7).value = 금액 * total_units
            ws.cell(row=row_num, column=START_COL + 7).font = data_font
            ws.cell(row=row_num, column=START_COL + 7).alignment = right_align
            ws.cell(row=row_num, column=START_COL + 7).border = thin_border
            ws.cell(row=row_num, column=START_COL + 7).number_format = "#,##0"

            row_num += 1

        # 합계 행 (배경 투명)
        ws.cell(row=row_num, column=START_COL).value = "합계"
        ws.cell(row=row_num, column=START_COL).font = header_font
        ws.cell(row=row_num, column=START_COL).alignment = center_align
        ws.cell(row=row_num, column=START_COL).border = thin_border

        ws.cell(row=row_num, column=START_COL + 1).value = "(V.A.T 별도)"
        ws.cell(row=row_num, column=START_COL + 1).font = header_font
        ws.cell(row=row_num, column=START_COL + 1).alignment = center_align
        ws.cell(row=row_num, column=START_COL + 1).border = thin_border

        # 세대당 합계
        for col in [START_COL + 2, START_COL + 3]:
            ws.cell(row=row_num, column=col).value = ""
            ws.cell(row=row_num, column=col).border = thin_border

        ws.cell(row=row_num, column=START_COL + 4).value = df["금액"].sum()
        ws.cell(row=row_num, column=START_COL + 4).font = header_font
        ws.cell(row=row_num, column=START_COL + 4).alignment = right_align
        ws.cell(row=row_num, column=START_COL + 4).border = thin_border
        ws.cell(row=row_num, column=START_COL + 4).number_format = "#,##0"

        # 총 합계
        for col in [START_COL + 5, START_COL + 6]:
            ws.cell(row=row_num, column=col).value = ""
            ws.cell(row=row_num, column=col).border = thin_border

        ws.cell(row=row_num, column=START_COL + 7).value = (
            df["금액"].sum() * total_units
        )
        ws.cell(row=row_num, column=START_COL + 7).font = header_font
        ws.cell(row=row_num, column=START_COL + 7).alignment = right_align
        ws.cell(row=row_num, column=START_COL + 7).border = thin_border
        ws.cell(row=row_num, column=START_COL + 7).number_format = "#,##0"

        # BytesIO로 저장
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return output.getvalue()

    # 바닥판 세대수 추출
    total_units = 1  # 기본값
    if floor_data:
        # floor_data 구조: {"inputs": {"units": N}, ...}
        inputs = floor_data.get("inputs", {})
        total_units = int(inputs.get("units", 1))

    xlsx_bytes = df_to_excel_bytes(est_df, total_units)
    st.download_button(
        "📥 현재 세대 견적서 다운로드",
        data=xlsx_bytes,
        file_name=f"욕실_원자재_세대당_단가내역_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ----------------------------
    # 통합 엑셀 출력 (다중 세대 타입)
    # ----------------------------
    def create_integrated_excel(saved_quotations: List[Dict]) -> bytes:
        """LGE 창원 스마트파크 형식의 통합 엑셀 생성"""
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "세대당 원자재 단가내역"

        # 스타일 정의
        title_font = Font(name="맑은 고딕", size=16, bold=True)
        header_font = Font(name="맑은 고딕", size=9, bold=True)
        data_font = Font(name="맑은 고딕", size=9)
        small_font = Font(name="맑은 고딕", size=8)

        center_align = Alignment(horizontal="center", vertical="center", wrap_text=True)
        left_align = Alignment(horizontal="left", vertical="center")
        right_align = Alignment(horizontal="right", vertical="center")

        thin_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        num_types = len(saved_quotations)
        if num_types == 0:
            return b""

        # 모든 품목/사양 조합 수집 (순서 유지)
        all_items = []
        seen = set()
        for q in saved_quotations:
            for row in q["rows"]:
                key = (row["품목"], row["사양 및 규격"])
                if key not in seen:
                    seen.add(key)
                    all_items.append(key)

        # 컬럼 구조 계산
        # 품목(1) + 사양(1) + [수량,단가,금액] × num_types + 비고(1)
        START_COL = 1
        SPEC_COL = 2
        DATA_START_COL = 3  # 첫 번째 세대 타입의 수량 컬럼

        # 1행: 타이틀
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=2 + num_types * 3 + 1)
        ws.cell(1, 1).value = "욕실 원자재 세대당 단가 내역"
        ws.cell(1, 1).font = title_font
        ws.cell(1, 1).alignment = center_align
        ws.row_dimensions[1].height = 25

        # 4행: 총수량 및 작성일
        total_all_units = sum(q["units"] for q in saved_quotations)
        ws.merge_cells(start_row=4, start_column=1, end_row=4, end_column=2)
        ws.cell(4, 1).font = header_font
        ws.merge_cells(start_row=4, start_column=DATA_START_COL + num_types * 3 - 2,
                       end_row=4, end_column=DATA_START_COL + num_types * 3)
        date_col = DATA_START_COL + num_types * 3 - 2
        ws.cell(4, date_col).value = f"총수량: {total_all_units}개"
        ws.cell(4, date_col).font = header_font
        ws.cell(4, date_col).alignment = right_align

        # 5행: 세대 타입 헤더 (◎ 타입명 ◎ 형태)
        ws.cell(5, START_COL).value = "품목"
        ws.cell(5, START_COL).font = header_font
        ws.cell(5, START_COL).alignment = center_align
        ws.cell(5, START_COL).border = thin_border

        ws.cell(5, SPEC_COL).value = "사양 및 규격"
        ws.cell(5, SPEC_COL).font = header_font
        ws.cell(5, SPEC_COL).alignment = center_align
        ws.cell(5, SPEC_COL).border = thin_border

        for i, q in enumerate(saved_quotations):
            col_start = DATA_START_COL + i * 3
            # 3컬럼 병합
            ws.merge_cells(start_row=5, start_column=col_start, end_row=5, end_column=col_start + 2)
            ws.cell(5, col_start).value = f"◎ {q['name']}"
            ws.cell(5, col_start).font = header_font
            ws.cell(5, col_start).alignment = center_align
            for c in range(col_start, col_start + 3):
                ws.cell(5, c).border = thin_border

        # 비고 컬럼
        remark_col = DATA_START_COL + num_types * 3
        ws.cell(5, remark_col).value = "(V.A.T 제외)"
        ws.cell(5, remark_col).font = small_font
        ws.cell(5, remark_col).alignment = center_align
        ws.cell(5, remark_col).border = thin_border

        # 6행: 규격 및 세대수
        ws.cell(6, START_COL).value = ""
        ws.cell(6, START_COL).border = thin_border
        ws.cell(6, SPEC_COL).value = ""
        ws.cell(6, SPEC_COL).border = thin_border

        for i, q in enumerate(saved_quotations):
            col_start = DATA_START_COL + i * 3
            ws.merge_cells(start_row=6, start_column=col_start, end_row=6, end_column=col_start + 2)
            ws.cell(6, col_start).value = f"◎ 규격({q['spec']})  ◎ {q['units']}세대"
            ws.cell(6, col_start).font = small_font
            ws.cell(6, col_start).alignment = center_align
            for c in range(col_start, col_start + 3):
                ws.cell(6, c).border = thin_border

        ws.cell(6, remark_col).value = "비고"
        ws.cell(6, remark_col).font = header_font
        ws.cell(6, remark_col).alignment = center_align
        ws.cell(6, remark_col).border = thin_border

        # 7행: 수량/단가/금액 헤더
        ws.cell(7, START_COL).value = ""
        ws.cell(7, START_COL).border = thin_border
        ws.cell(7, SPEC_COL).value = ""
        ws.cell(7, SPEC_COL).border = thin_border

        for i in range(num_types):
            col_start = DATA_START_COL + i * 3
            ws.cell(7, col_start).value = "수량"
            ws.cell(7, col_start).font = header_font
            ws.cell(7, col_start).alignment = center_align
            ws.cell(7, col_start).border = thin_border

            ws.cell(7, col_start + 1).value = "단가"
            ws.cell(7, col_start + 1).font = header_font
            ws.cell(7, col_start + 1).alignment = center_align
            ws.cell(7, col_start + 1).border = thin_border

            ws.cell(7, col_start + 2).value = "금액"
            ws.cell(7, col_start + 2).font = header_font
            ws.cell(7, col_start + 2).alignment = center_align
            ws.cell(7, col_start + 2).border = thin_border

        ws.cell(7, remark_col).value = ""
        ws.cell(7, remark_col).border = thin_border

        # 데이터 행 작성
        row_num = 8
        current_category = None

        # 각 세대별 데이터를 딕셔너리로 변환 (빠른 조회용)
        type_data = []
        for q in saved_quotations:
            item_dict = {}
            for r in q["rows"]:
                key = (r["품목"], r["사양 및 규격"])
                item_dict[key] = r
            type_data.append(item_dict)

        for 품목, 사양 in all_items:
            # 품목 (카테고리 변경시만 표시)
            cell_cat = ws.cell(row=row_num, column=START_COL)
            if 품목 != current_category:
                cell_cat.value = 품목
                current_category = 품목
            else:
                cell_cat.value = ""
            cell_cat.font = data_font
            cell_cat.alignment = left_align
            cell_cat.border = thin_border

            # 사양 및 규격
            ws.cell(row=row_num, column=SPEC_COL).value = 사양
            ws.cell(row=row_num, column=SPEC_COL).font = data_font
            ws.cell(row=row_num, column=SPEC_COL).alignment = left_align
            ws.cell(row=row_num, column=SPEC_COL).border = thin_border

            # 각 세대 타입별 수량/단가/금액
            for i, td in enumerate(type_data):
                col_start = DATA_START_COL + i * 3
                key = (품목, 사양)
                if key in td:
                    r = td[key]
                    qty = r.get("수량", 0) or 0
                    price = r.get("단가", 0) or 0
                    amount = r.get("금액", 0) or 0
                else:
                    qty, price, amount = 0, 0, 0

                ws.cell(row=row_num, column=col_start).value = qty if qty else 0
                ws.cell(row=row_num, column=col_start).font = data_font
                ws.cell(row=row_num, column=col_start).alignment = right_align
                ws.cell(row=row_num, column=col_start).border = thin_border
                ws.cell(row=row_num, column=col_start).number_format = "#,##0.##"

                ws.cell(row=row_num, column=col_start + 1).value = price if price else 0
                ws.cell(row=row_num, column=col_start + 1).font = data_font
                ws.cell(row=row_num, column=col_start + 1).alignment = right_align
                ws.cell(row=row_num, column=col_start + 1).border = thin_border
                ws.cell(row=row_num, column=col_start + 1).number_format = "#,##0"

                ws.cell(row=row_num, column=col_start + 2).value = amount if amount else 0
                ws.cell(row=row_num, column=col_start + 2).font = data_font
                ws.cell(row=row_num, column=col_start + 2).alignment = right_align
                ws.cell(row=row_num, column=col_start + 2).border = thin_border
                ws.cell(row=row_num, column=col_start + 2).number_format = "#,##0"

            # 비고
            ws.cell(row=row_num, column=remark_col).value = ""
            ws.cell(row=row_num, column=remark_col).border = thin_border

            row_num += 1

        # 합계 행: 세트당 단가
        ws.cell(row=row_num, column=START_COL).value = "세트당 단가"
        ws.cell(row=row_num, column=START_COL).font = header_font
        ws.cell(row=row_num, column=START_COL).alignment = center_align
        ws.cell(row=row_num, column=START_COL).border = thin_border
        ws.cell(row=row_num, column=SPEC_COL).value = ""
        ws.cell(row=row_num, column=SPEC_COL).border = thin_border

        for i, q in enumerate(saved_quotations):
            col_start = DATA_START_COL + i * 3
            ws.cell(row=row_num, column=col_start).value = 1
            ws.cell(row=row_num, column=col_start).font = header_font
            ws.cell(row=row_num, column=col_start).alignment = right_align
            ws.cell(row=row_num, column=col_start).border = thin_border

            ws.cell(row=row_num, column=col_start + 1).value = ""
            ws.cell(row=row_num, column=col_start + 1).border = thin_border

            ws.cell(row=row_num, column=col_start + 2).value = q["total"]
            ws.cell(row=row_num, column=col_start + 2).font = header_font
            ws.cell(row=row_num, column=col_start + 2).alignment = right_align
            ws.cell(row=row_num, column=col_start + 2).border = thin_border
            ws.cell(row=row_num, column=col_start + 2).number_format = "#,##0"

        ws.cell(row=row_num, column=remark_col).value = ""
        ws.cell(row=row_num, column=remark_col).border = thin_border
        row_num += 1

        # 세대 총 합계 행
        ws.cell(row=row_num, column=START_COL).value = "세대 총 합계"
        ws.cell(row=row_num, column=START_COL).font = header_font
        ws.cell(row=row_num, column=START_COL).alignment = center_align
        ws.cell(row=row_num, column=START_COL).border = thin_border
        ws.cell(row=row_num, column=SPEC_COL).value = ""
        ws.cell(row=row_num, column=SPEC_COL).border = thin_border

        grand_total = 0
        for i, q in enumerate(saved_quotations):
            col_start = DATA_START_COL + i * 3
            type_total = q["total"] * q["units"]
            grand_total += type_total

            ws.cell(row=row_num, column=col_start).value = q["units"]
            ws.cell(row=row_num, column=col_start).font = header_font
            ws.cell(row=row_num, column=col_start).alignment = right_align
            ws.cell(row=row_num, column=col_start).border = thin_border

            ws.cell(row=row_num, column=col_start + 1).value = ""
            ws.cell(row=row_num, column=col_start + 1).border = thin_border

            ws.cell(row=row_num, column=col_start + 2).value = type_total
            ws.cell(row=row_num, column=col_start + 2).font = header_font
            ws.cell(row=row_num, column=col_start + 2).alignment = right_align
            ws.cell(row=row_num, column=col_start + 2).border = thin_border
            ws.cell(row=row_num, column=col_start + 2).number_format = "#,##0"

        # 총 합계 표시
        ws.cell(row=row_num, column=remark_col).value = f"{grand_total:,.0f}"
        ws.cell(row=row_num, column=remark_col).font = header_font
        ws.cell(row=row_num, column=remark_col).alignment = right_align
        ws.cell(row=row_num, column=remark_col).border = thin_border

        # 컬럼 너비 설정
        ws.column_dimensions[get_column_letter(START_COL)].width = 12
        ws.column_dimensions[get_column_letter(SPEC_COL)].width = 30
        for i in range(num_types):
            col_start = DATA_START_COL + i * 3
            ws.column_dimensions[get_column_letter(col_start)].width = 7      # 수량
            ws.column_dimensions[get_column_letter(col_start + 1)].width = 10  # 단가
            ws.column_dimensions[get_column_letter(col_start + 2)].width = 12  # 금액
        ws.column_dimensions[get_column_letter(remark_col)].width = 15

        # BytesIO로 저장
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        return output.getvalue()

    # 통합 엑셀 다운로드 버튼
    saved_list = st.session_state.get(SAVED_QUOTATIONS_KEY, [])
    if saved_list and len(saved_list) >= 1:
        st.markdown("### 통합 견적서 다운로드")
        integrated_bytes = create_integrated_excel(saved_list)
        if integrated_bytes:
            st.download_button(
                "📥 통합 견적서 Excel 다운로드 (LGE 형식)",
                data=integrated_bytes,
                file_name=f"욕실_원자재_통합_단가내역_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
            st.info(f"총 {len(saved_list)}개 세대 타입 포함")

if warnings:
    with st.expander("⚠️ 경고/참고", expanded=False):
        for w in warnings:
            st.warning(w)
