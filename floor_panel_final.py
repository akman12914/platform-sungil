# floor_panel_final.py (수정됨: 순수 로직 모듈)
import io
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


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
# 최종 계산 및 매칭 함수 (UI 페이지에서 호출하여 사용)
# ---------------------------
def calculate_floor_panel(
    df: pd.DataFrame,
    units: int,
    central: str,
    shape: str,
    btype: str,
    bw: int,
    bl: int,
    sw: Optional[int],
    sl: Optional[int],
    shw: Optional[int],
    shl: Optional[int],
    mgmt_rate: float,
    pve_kind: str,
) -> Dict[str, Any]:
    """바닥판 규격/옵션 산출 메인 로직"""
    decision_log = []

    # 세면/샤워 치수 None 처리
    disable_sink_shower = (
        (central == "Yes")
        or (btype == "구분없음")
        or (shape == "코너형" and btype != "샤워형")
    )
    if disable_sink_shower:
        sw, sl, shw, shl = None, None, None, None

    # 세대수 우선 규칙
    if units < 100:
        decision_log.append(f"세대수={units} (<100) → PVE 강제 선택")
        q = pve_quote(bw, bl, mgmt_rate, pve_kind)
        base_subtotal = q["소계"]
        material = q["소재"]
        result_kind = "PVE"

    else:
        matched = None
        # 1) 중앙배수 Yes → GRP(중앙배수) 시도 → 실패 시 PVE
        if central == "Yes":
            decision_log.append("중앙배수=Yes → GRP(중앙배수) 매칭 시도")
            matched = match_center_drain(df, shape, btype, bw, bl)

        # 2) 중앙배수 No & 사각형
        elif shape == "사각형":
            decision_log.append("중앙배수=No & 형태=사각형 → 매칭 시도")
            matched = match_non_center_rectangle(df, btype, bw, bl, sw, sl, shw, shl)

        # 3) 중앙배수 No & 코너형 (샤워형만 고려)
        elif shape == "코너형" and btype == "샤워형":
            decision_log.append(
                "중앙배수=No & 형태=코너형 & 유형=샤워형 → GRP→FRP 순서"
            )
            matched = match_corner_shower(df, bw, bl, sw, sl, shw, shl)

        if matched is None:
            decision_log.append("규격표 매칭 실패 → PVE 계산")
            q = pve_quote(bw, bl, mgmt_rate, pve_kind)
            base_subtotal = q["소계"]
            material = q["소재"]
            result_kind = "PVE"
        else:
            row = matched["row"]
            material = matched["소재"]
            base_subtotal = int(row["소계"])
            result_kind = f"{material}" + (
                " (단차없음)" if matched.get("단차없음") else ""
            )
            decision_log.append(f"{result_kind} 매칭 성공 → 최소 소계 선택")

    # 공통: 관리비 포함 소계
    mgmt_total = int(round(base_subtotal * (1.0 + mgmt_rate)))

    return {
        "material": material,
        "base_subtotal": base_subtotal,
        "mgmt_total": mgmt_total,
        "result_kind": result_kind,
        "decision_log": decision_log,
    }


# ---------------------------
# 도형 렌더링 (PIL)
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
            # anchor='mm' 지원 시 (최신 Pillow)
            drw.text((x, y), txt, fill=fill, anchor="mm")
        except TypeError:
            # anchor 미지원 시 (구 버전)
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
        # anchor 미지원 시 기본 텍스트
        drw.text((x0, y0 - 8), "욕실길이", fill="black")
        drw.text((x0 - 40, (y0 + y1) / 2), "욕실폭", fill="black")

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
    if shape == "코너형" and btype == "샤워형":
        # 코너형은 UI에서 세면/샤워부 폭/길이를 입력받지 않아 sw, sl, shw, shl이 None으로 올 수 있습니다.
        # 그러나 코너형 샤워부 매칭 함수는 이 치수들을 사용하므로, UI에서 입력 받은 값을 사용합니다.
        # 다만, 코너형 도식 규칙에 따라 여기서 sw, shw는 0이 아닐 경우 사용합니다.

        # 코너형 도식 로직은 중앙배수 No & 샤워형일 때만 실행되어야 합니다.

        total_w = sw + shw
        ratio = (sw / total_w) if total_w > 0 else 0.5  # 세면부 비율
        boundary_x = x0 + int(round(BW * ratio))

        # 세면부(전고)
        left_x0 = x0 + BORDER
        left_x1 = max(left_x0 + 1, boundary_x - GAP)
        if left_x1 > left_x0 and sw > 0:
            if safe_rect(left_x0, y0 + BORDER, left_x1, y1 - BORDER, "blue", 3):
                text_center((left_x0 + left_x1) / 2, (y0 + y1) / 2, "세면부", "blue")

        # 경계선(전고)
        if sw > 0 and shw > 0:
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
