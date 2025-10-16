# wall_panel.py  (streamlit 앱)
# 새 Layout 계산 엔진(layout_report) 완전 통합 버전

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# --- Common Styles ---
from common_styles import apply_common_styles, set_page_config

# --- Streamlit ---
import streamlit as st
set_page_config(page_title="벽판 계산기", layout="wide")
apply_common_styles()

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os, json
from datetime import datetime

EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def _save_json(path:str, data:dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

try:
    FONT = ImageFont.truetype("NanumGothic.ttf", 16)
    FONT_SMALL = ImageFont.truetype("NanumGothic.ttf", 14)
except Exception:
    FONT = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()

# =========================================================
# 0) 공통 유틸
# =========================================================

FLOOR_DONE_KEY = "floor_done"
FLOOR_RESULT_KEY = "floor_result"

WALL_DONE_KEY  = "wall_done"
WALL_RESULT_KEY = "wall_result"

CEIL_DONE_KEY  = "ceil_done"
CEIL_RESULT_KEY = "ceil_result"

def parse_tile(tile_str: str) -> Tuple[int, int]:
    """'300×600' 또는 '250×400' → (300, 600)"""
    a, b = tile_str.replace("x", "×").split("×")
    return int(a), int(b)

def effective_height(H: int, floor_type: str) -> int:
    """바닥판 유형이 PVE면 +50."""
    return int(H) + 50 if floor_type.upper() == "PVE" else int(H)

MIN_EDGE = 80

# =========================================================
# 1) 새 Layout 계산 엔진 (Self-contained)
# =========================================================
def iround(x: float) -> int:
    """Half-up 반올림(ROUND)."""
    return int(math.floor(x + 0.5))

class RuleError(Exception):
    pass

@dataclass
class PanelCell:
    col: int  # 1-based
    row: int  # 1-based (bottom=1)
    w: int
    h: int
    col_tags: Tuple[str, ...] = ()
    row_tags: Tuple[str, ...] = ()
    col_note: str = ""
    row_note: str = ""
    def as_dict(self) -> Dict[str, Any]:
        return {
            "col": self.col, "row": self.row,
            "panel_w": self.w, "panel_h": self.h,
            "col_tags": ",".join(self.col_tags) if self.col_tags else "",
            "row_tags": ",".join(self.row_tags) if self.row_tags else "",
            "col_note": self.col_note,
            "row_note": self.row_note,
        }

def ensure_producible_new(panels: List[PanelCell]):
    for p in panels:
        if p.w <= MIN_EDGE or p.h <= MIN_EDGE:
            raise RuleError(f"PANEL_TOO_SMALL: {p.w}x{p.h} at C{p.col}-R{p.row}")

# --------- 가로(HB/VSTRIP) 분해 ----------
def hb2_split(width: int, TW: int) -> Tuple[int, int, str]:
    """가로발란스(2분할, ROUND)."""
    if width < 2 * TW:
        raise RuleError(f"WIDTH_TOO_SMALL_FOR_BALANCE: {width} < {2*TW}")
    n = iround(width / (2 * TW))
    left = n * TW
    right = width - left
    if left <= MIN_EDGE or right <= MIN_EDGE:
        raise RuleError(f"HB2_BAD_SPLIT: {left}, {right}")
    note = f"HB2: n={n}, left={left}, right={right} (TW={TW})"
    return left, right, note

def split_columns_verbose(W: int, TW: int) -> Tuple[List[Dict], str]:
    """
    columns: [{width, tags(HB/VSTRIP), col_note}, ...]
    horiz_branch: 가로 분해 설명
    규칙: 2400 모듈, 오른쪽 2칸만 HB 허용, 잔여 dW<=80이면 (2400→2400−TW, 세로판→dW+TW)
    """
    cols: List[Dict] = []
    # 80 < W <= 1000 : VSTRIP 1열
    if 80 < W <= 1000:
        cols.append({"width": W, "tags": ("VSTRIP",), "col_note": "W<=1000 → VSTRIP 1열"})
        return cols, "VERTICAL STRIP ONLY (80<W<=1000)"
    # 1000 < W <= 2400 : 단일 열
    if W <= 2400:
        cols.append({"width": W, "tags": tuple(), "col_note": "SINGLE COLUMN (1000<W<=2400)"})
        return cols, "SINGLE COLUMN (1000<W<=2400)"

    # 2400 < W <= 3400 : 2400 + VSTRIP(dW) with ≤80mm correction
    if W <= 3400:
        dW = W - 2400
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400 + VSTRIP(dW), dW={dW}"

    # 3400 < W <= 4800 : HB 2열
    if W <= 4800:
        L, R, note = hb2_split(W, TW)
        cols.append({"width": L, "tags": ("HB",), "col_note": note})
        cols.append({"width": R, "tags": ("HB",), "col_note": note})
        return cols, "HB2 (3400<W<=4800)"

    # 4800 < W <= 5800 : 2400 + 2400 + VSTRIP(dW) with ≤80mm correction
    if W <= 5800:
        dW = W - 4800
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400 + 2400 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400 + 2400 + VSTRIP(dW), dW={dW}"

    # 5800 < W <= 7200 : 2400 + HB2(W-2400)
    if W <= 7200:
        rest = W - 2400
        L, R, note = hb2_split(rest, TW)
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": L, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        cols.append({"width": R, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        return cols, f"2400 + HB2(W-2400), rest={rest}"

    # 7200 < W <= 8200 : 2400×3 + VSTRIP(dW) with ≤80mm correction
    if W <= 8200:
        dW = W - 7200
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        if dW <= 80:
            cols.append({"width": 2400 - TW, "tags": tuple(), "col_note": f"2400→{2400-TW} (80mm 보정)"})
            cols.append({"width": dW + TW, "tags": ("VSTRIP",), "col_note": f"VSTRIP {dW}+{TW} (80mm 보정)"})
            return cols, f"2400×3 + VSTRIP(dW), dW={dW} ≤ 80 → 보정"
        else:
            cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
            cols.append({"width": dW, "tags": ("VSTRIP",), "col_note": f"VSTRIP dW={dW}"})
            return cols, f"2400×3 + VSTRIP(dW), dW={dW}"

    # 8200 < W <= 9600 : 2400×2 + HB2(W-4800)
    if W <= 9600:
        rest = W - 4800
        L, R, note = hb2_split(rest, TW)
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": 2400, "tags": tuple(), "col_note": "MODULE 2400"})
        cols.append({"width": L, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        cols.append({"width": R, "tags": ("HB",), "col_note": f"HB2 on rest: {note}"})
        return cols, f"2400×2 + HB2(W-4800), rest={rest}"

    raise RuleError("WIDTH_OUT_OF_RANGE")

# --------- 세로(행) 분해 ----------
def vb_round(H: int, TH: int) -> Tuple[int, int]:
    """세로발란스(ROUND): top=m*TH, bot=H-top."""
    m = iround(H / (2 * TH))
    top = m * TH
    bot = H - top
    return top, bot

def split_heights_general(H: int, TH: int) -> Tuple[List[Tuple[int, Tuple[str, ...]]], str, str]:
    """
    TH=300 → 상부 1200 모듈 반복 후 잔여(newH)에 VB(1200~2400) 또는 1장(≤1200)
    TH=250 → 상부 1000 모듈 반복, 최하부 ≤1200, 잔여(newH∈(1200,2200])는 VB
    """
    heights: List[Tuple[int, Tuple[str, ...]]] = []
    note = ""
    if TH == 300:  # 300x600
        k = max(0, math.ceil((H - 2400) / 1200))
        for _ in range(k):
            heights.append((1200, tuple()))
        newH = H - 1200 * k
        branch = f"GEN 300x600: k_upper1200={k}, newH={newH}"
        if newH > 0:
            if newH <= 1200:
                heights.append((newH, tuple()))
                note = "newH<=1200 → bottom 1장"
            else:
                top, bot = vb_round(newH, 300)
                heights.append((top, ('VB',)))
                heights.append((bot, ('VB',)))
                note = f"newH in (1200,2400] → VB ROUND (top={top}, bot={bot})"
        return heights, branch, note

    elif TH == 250:  # 250x400
        k = max(0, (H - 1200) // 1000)
        for _ in range(k):
            heights.append((1000, tuple()))
        newH = H - 1000 * k
        branch = f"GEN 250x400: k_upper1000={k}, newH={newH}"
        if newH <= 1200:
            heights.append((newH, tuple()))
            note = "newH<=1200 → bottom 1장"
        else:
            if newH > 2200:
                heights.append((1000, tuple()))
                newH -= 1000
            top, bot = vb_round(newH, 250)
            heights.append((top, ('VB',)))
            heights.append((bot, ('VB',)))
            note = f"newH in (1200,2200] → VB ROUND (top={top}, bot={bot})"
        return heights, branch, note

    else:
        raise RuleError("UNSUPPORTED_TILE_HEIGHT")

def split_heights_vstrip(H: int, TH: int) -> Tuple[List[Tuple[int, Tuple[str, ...]]], str]:
    """세로판(VSTRIP) 열의 세로 분해 (타일별 1~4판 규칙 + 아래 2판에 VB)"""
    if TH == 300:
        if H <= 2400:
            return [(H, tuple())], "VSTRIP 300x600: H<=2400 → 1판"
        if H <= 4800:
            top, bot = vb_round(H, 300)
            return [(top, ('VB',)), (bot, ('VB',))], f"VSTRIP 300x600: 2판 VB (top={top}, bot={bot})"
        if H <= 7200:
            rem = H - 1200
            top2, bot2 = vb_round(rem, 300)
            return [(1200, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 300x600: 3판 (1200 + VB on {rem} → {top2},{bot2})"
        rem = H - 2400
        top2, bot2 = vb_round(rem, 300)
        return [(1200, tuple()), (1200, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 300x600: 4판 (1200×2 + VB on {rem} → {top2},{bot2})"

    elif TH == 250:
        if H <= 2200:
            return [(H, tuple())], "VSTRIP 250x400: H<=2200 → 1판"
        if H <= 4200:
            top, bot = vb_round(H, 250)
            return [(top, ('VB',)), (bot, ('VB',))], f"VSTRIP 250x400: 2판 VB (top={top}, bot={bot})"
        if H <= 6200:
            rem = H - 1000
            top2, bot2 = vb_round(rem, 250)
            return [(1000, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 250x400: 3판 (1000 + VB on {rem} → {top2},{bot2})"
        rem = H - 2000
        top2, bot2 = vb_round(rem, 250)
        return [(1000, tuple()), (1000, tuple()), (top2, ('VB',)), (bot2, ('VB',))], f"VSTRIP 250x400: 4판 (1000×2 + VB on {rem} → {top2},{bot2})"

    else:
        raise RuleError("UNSUPPORTED_TILE_HEIGHT")

def layout_report(W: int, H: int, TH: int, TW: int) -> Dict[str, Any]:
    """
    입력: W,H,TH,TW
    출력: {
      inputs, constraints, horiz_branch,
      columns: [{col, col_w, col_tags, col_note, vertical_rule, vertical_note}],
      panels:  [{col,row,panel_w,panel_h,col_tags,row_tags,col_note,row_note}],
      counts:  {n_cols, n_cols_vstrip, n_cols_hb, n_panels}
    }
    """
    if W <= MIN_EDGE or H <= MIN_EDGE:
        raise RuleError("W_AND_H_MUST_BE_GREATER_THAN_80")
    if (TH, TW) == (300, 600):
        if W > 9600 or H > 9600:
            raise RuleError("OUT_OF_RANGE_300x600_MAX_9600x9600")
    elif (TH, TW) == (250, 400):
        if W > 9600 or H > 8200:
            raise RuleError("OUT_OF_RANGE_250x400_MAX_9600x8200")
    else:
        raise RuleError("UNSUPPORTED_TILE_SIZE")

    cols, horiz_branch = split_columns_verbose(W, TW)

    panels: List[PanelCell] = []
    col_meta: List[Dict[str, Any]] = []
    for ci, c in enumerate(cols, start=1):
        cw, ctags, cnote = int(c["width"]), tuple(c["tags"]), c["col_note"]
        if 'VSTRIP' in ctags:
            rows, vbranch = split_heights_vstrip(H, TH)
            vbranch_general, vnote_general = "", ""
        else:
            rows, vbranch_general, vnote_general = split_heights_general(H, TH)
            vbranch = ""
        col_meta.append({
            "col": ci, "col_w": cw, "col_tags": ",".join(ctags) if ctags else "",
            "col_note": cnote,
            "vertical_rule": vbranch or vbranch_general, "vertical_note": vnote_general
        })
        for rj, (rh, rtags) in enumerate(rows, start=1):
            panels.append(PanelCell(
                col=ci, row=rj, w=cw, h=int(rh),
                col_tags=ctags, row_tags=rtags,
                col_note=cnote, row_note=(vbranch or vnote_general)
            ))

    ensure_producible_new(panels)

    n_cols = len(cols)
    n_vstrip = sum(1 for c in cols if 'VSTRIP' in c["tags"])
    n_hbcols = sum(1 for c in cols if 'HB' in c["tags"])
    out = {
        "inputs": {"W": W, "H": H, "TH": TH, "TW": TW},
        "constraints": {
            "min_edge_mm": MIN_EDGE, "range_ok": True,
            "tile_limit": "300x600: 9600×9600 / 250x400: 9600×8200"
        },
        "horiz_branch": horiz_branch,
        "columns": [{
            "col": cm["col"], "col_w": cm["col_w"],
            "col_tags": cm["col_tags"], "col_note": cm["col_note"],
            "vertical_rule": cm["vertical_rule"], "vertical_note": cm["vertical_note"],
        } for cm in col_meta],
        "panels": [p.as_dict() for p in panels],
        "counts": {"n_cols": n_cols, "n_cols_vstrip": n_vstrip, "n_cols_hb": n_hbcols, "n_panels": len(panels)}
    }
    return out

# =========================================================
# 2) 벽/벽면(Face) 모델 & 생성 (기존 UI/분할 로직 유지)
# =========================================================
def validate_corner_dims(w: Dict[int, int]) -> List[str]:
    """코너형 합치 조건 검사: W1==W3+W5, W2==W4+W6, 모두>0."""
    err = []
    W1, W2, W3, W4, W5, W6 = (w[i] for i in range(1, 7))
    if any(v <= 0 for v in [W1, W2, W3, W4, W5, W6]):
        err.append("코너형 모든 벽폭(W1~W6)은 0보다 커야 합니다.")
    if W1 != W3 + W5:
        err.append(f"합치 조건 위반: W1(={W1})은 W3+W5(={W3+W5}) 이어야 합니다.")
    if W2 != W4 + W6:
        err.append(f"합치 조건 위반: W2(={W2})은 W4+W6(={W4+W6}) 이어야 합니다.")
    return err

def normalize_door(W: int, s: float, d: float) -> Tuple[float, float, float, float, int]:
    """도어 시작/폭 정규화: 반환 (s, e, L, R, n_faces)"""
    if d <= 0 or d > W:
        raise ValueError("문 폭(d)이 유효하지 않습니다. 0 < d ≤ 문벽 폭(W)을 만족해야 합니다.")
    s = max(0.0, min(float(s), float(W)))
    if s == W:
        s = float(W - d)
    e = s + d
    if e > W:
        raise ValueError("문 범위(s+d)가 문벽 폭(W)을 초과합니다. 시작점 또는 문폭을 줄이세요.")
    L = s
    R = W - e
    n_faces = (1 if L > 0 else 0) + (1 if R > 0 else 0)
    return s, e, L, R, n_faces

def rect_wall_width_of(wall_id: int, BW: int, BL: int) -> int:
    """사각형: 1=상(BL), 2=우(BW), 3=하(BL), 4=좌(BW)"""
    if wall_id == 1: return BL
    if wall_id == 2: return BW
    if wall_id == 3: return BL
    if wall_id == 4: return BW
    raise ValueError("사각형 문벽 번호는 1~4 범위여야 합니다.")

def corner_wall_width_of(wall_id: int, w: Dict[int, int]) -> int:
    """코너형: 입력 W1..W6 그대로 사용"""
    if wall_id not in w:
        raise ValueError("코너형 문/젠다이 벽 번호는 1~6 범위여야 합니다.")
    return w[wall_id]

@dataclass
class FaceSpec:
    wall_id: int
    wall_label: str  # "W1".."W6"
    face_idx: int
    face_label: str  # "W1F1".."W6F3"
    x0: int; x1: int
    y0: int; y1: int
    width_mm: int
    height_mm: int
    note: str        # "door-left"/"door-right"/"jendai-lower"/"jendai-upper"/"single"/"adj-1"/"adj-2"

def wall_label(shape: str, wall_id: int) -> str:
    return f"W{wall_id}"

def build_faces_for_wall(
    shape: str,
    wall_id: int,
    width_mm: int,
    height_mm: int,
    door_tuple: Optional[Tuple[float, float]] = None,   # (s,e) mm
    j_enabled: bool = False,
    j_wall: Optional[int] = None,
    j_has_step: bool = False,
    j_h: int = 1000,
    j_depth: int = 0,
    j_lower_segments: Optional[List[int]] = None,       # 단차 분할 폭 리스트 (사각형 3개, 코너형 2개)
    j_contact_walls: Optional[List[int]] = None,        # 접벽(최대 2개)
) -> List[FaceSpec]:
    """
    한 '벽'을 문/젠다이/접벽 설정에 따라 여러 FaceSpec으로 분해한다.
    """
    wl = wall_label(shape, wall_id)
    faces: List[FaceSpec] = []

    # 0) 도어 분할(우선 적용)
    if door_tuple is not None:
        s_mm = int(round(door_tuple[0]))
        e_mm = int(round(door_tuple[1]))
        L = max(0, s_mm)
        R = max(0, width_mm - e_mm)
        fi = 1
        if L > 0:
            faces.append(FaceSpec(wall_id, wl, fi, f"{wl}F{fi}", 0, L, 0, height_mm, L, height_mm, "door-left")); fi += 1
        if R > 0:
            faces.append(FaceSpec(wall_id, wl, fi, f"{wl}F{fi}", e_mm, e_mm + R, 0, height_mm, R, height_mm, "door-right"))
        return faces

    # 1) 접벽 분할(설치공간: 오버레이만, 면 제외)
    if j_enabled and j_contact_walls and (wall_id in j_contact_walls) and (j_depth > 0) and (j_h > 0):
        fi = 1
        depth = min(int(j_depth), int(width_mm))
        jh = min(int(j_h), int(height_mm))
        top_h = max(0, int(height_mm) - jh)
        if depth > 0 and top_h > 0:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                0, depth, jh, int(height_mm),
                depth, top_h, "adj-1"
            )); fi += 1
        rem_w = max(0, int(width_mm) - depth)
        if rem_w > 0:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                depth, int(width_mm), 0, int(height_mm),
                rem_w, int(height_mm), "adj-2"
            ))
        return faces

    # 2) 젠다이(해당 벽)
    if j_enabled and (j_wall is not None) and (int(j_wall) == int(wall_id)) and (j_h > 0):
        fi = 1
        band_h = min(int(j_h), int(height_mm))
        if j_has_step:
            segments = [int(v) for v in (j_lower_segments or []) if int(v) > 0]
            acc = 0
            for seg_w in segments:
                seg_w = min(seg_w, int(width_mm) - acc)
                if seg_w <= 0: continue
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    acc, acc + seg_w, 0, band_h,
                    seg_w, band_h, "jendai-lower"
                ))
                acc += seg_w; fi += 1
            upper_h = max(0, int(height_mm) - band_h)
            if upper_h > 0:
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    0, int(width_mm), band_h, band_h + upper_h,
                    int(width_mm), upper_h, "jendai-upper"
                ))
        else:
            faces.append(FaceSpec(
                wall_id, wl, fi, f"{wl}F{fi}",
                0, int(width_mm), 0, band_h,
                int(width_mm), band_h, "jendai-lower"
            ))
            fi += 1
            upper_h = max(0, int(height_mm) - band_h)
            if upper_h > 0:
                faces.append(FaceSpec(
                    wall_id, wl, fi, f"{wl}F{fi}",
                    0, int(width_mm), band_h, band_h + upper_h,
                    int(width_mm), upper_h, "jendai-upper"
                ))
        return faces

    # 3) 기본 면
    faces.append(FaceSpec(
        wall_id, wl, 1, f"{wl}F1",
        0, int(width_mm), 0, int(height_mm),
        int(width_mm), int(height_mm), "single"
    ))
    return faces

# =========================================================
# 3) 도면 렌더링 (평면도 + 정면도/벽면 라벨)
# =========================================================

def _text_size(font, text):
    try:
        x0, y0, x1, y1 = font.getbbox(text)  # (x0,y0,x1,y1)
        return (x1 - x0, y1 - y0)
    except Exception:
        # 대략값
        return (len(text) * 8, 16)


def draw_rect_preview(
    BL: int, BW: int,
    has_split: bool, X: Optional[int],
    door_info: Optional[Tuple[int, float, float, int]] = None,
) -> Image.Image:
    """사각형 평면도. 라벨: W1~W4"""
    if BW > BL:
        BL, BW = BW, BL
    CANVAS_W = 760
    MARGIN = 20

    # 라벨 텍스트 크기
    w_W1, h_W1 = _text_size(FONT, "W1")
    w_W2, h_W2 = _text_size(FONT, "W2")
    w_W3, h_W3 = _text_size(FONT, "W3")
    w_W4, h_W4 = _text_size(FONT, "W4")

    # 바깥 라벨용 가변 패딩(라벨 폭/높이 + 여유 6px)
    PAD_L = max(MARGIN, w_W4 + 6)
    PAD_R = max(MARGIN, w_W2 + 6)
    PAD_T = max(MARGIN, h_W3 + 6)
    PAD_B = max(MARGIN, h_W1 + 6)


    sx = (CANVAS_W - (PAD_L + PAD_R)) / max(1, float(BL))
    sy = sx
    CANVAS_H = int(BW * sy + 2*MARGIN)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    x0, y0 = int(PAD_L), int(PAD_T)
    x1, y1 = x0 + int(round(BL * sx)), y0 + int(round(BW * sy))

    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    if has_split and X is not None:
        gx = x0 + int(X * sx)
        drw.line([gx, y0, gx, y1], fill="blue", width=3)

    if door_info:
        wall_id, s, e, W_wall = door_info
        if wall_id == 1:
            xs = x0 + int(s * sx); xe = x0 + int(e * sx); y = y1
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 3:
            xs = x0 + int(s * sx); xe = x0 + int(e * sx); y = y0
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 2:
            ys = y0 + int(s * sy); ye = y0 + int(e * sy); x = x1
            drw.line([x, ys, x, ye], fill="red", width=5)
        elif wall_id == 4:
            ys = y0 + int(s * sy); ye = y0 + int(e * sy); x = x0
            drw.line([x, ys, x, ye], fill="red", width=5)

    # 5) 라벨(바깥) — anchor 사용(+ 폴백)
    pad = 4
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    try:
        drw.text((cx, y1 + pad), "W1", fill="black", font=FONT, anchor="mt")  # middle-top
        drw.text((x1 + pad, cy), "W2", fill="black", font=FONT, anchor="lm")  # left-middle
        drw.text((cx, y0 - pad), "W3", fill="black", font=FONT, anchor="mb")  # middle-bottom
        drw.text((x0 - pad, cy), "W4", fill="black", font=FONT, anchor="rm")  # right-middle
    except Exception:
        # anchor 미지원 폴백(폭/높이 반영)
        drw.text((cx - w_W1/2, y1 + pad), "W1", fill="black", font=FONT)
        drw.text((x1 + pad,   cy - h_W2/2), "W2", fill="black", font=FONT)
        drw.text((cx - w_W3/2, y0 - pad - h_W3), "W3", fill="black", font=FONT)
        drw.text((x0 - pad - w_W4, cy - h_W4/2), "W4", fill="black", font=FONT)
    return img

def draw_corner_preview(
    W: dict,
    has_split: bool,
    canvas_w: int = 760,
    margin: int = 20,
) -> Image.Image:
    """코너형 평면도. 라벨: W1~W6 (오목부는 기존 로직 유지)"""
    W1, W2, W3, W4, W5, W6 = (int(W[i]) for i in range(1, 7))

     # 라벨 크기
    w2,h2 = _text_size(FONT, "W2")
    w4,h4 = _text_size(FONT, "W4")
    w1,h1 = _text_size(FONT, "W1")
    w3,h3 = _text_size(FONT, "W3")
    w5,h5 = _text_size(FONT, "W5")
    w6,h6 = _text_size(FONT, "W6")



    # 라벨이 선에 붙지 않도록 추가 여백(라벨 폭/높이 반영)
    EXTRA_L = max(12, w2 + 8)      # 왼쪽(W2)
    EXTRA_R = max(12, w6 + 8)      # 오른쪽(W6)
    EXTRA_T = max(12, h3 + 8)      # 위쪽(W3)
    EXTRA_B = max(12, h1 + 8)      # 아래쪽(W1)

    CANVAS_W = int(canvas_w)
    MARGIN   = int(margin)

    # ── 2) 스케일 계산: 좌/우 여백을 뺀 가용 폭으로 sx 결정
    usable_w = CANVAS_W - (MARGIN + EXTRA_L) - (MARGIN + EXTRA_R)
    usable_w = max(1, usable_w)  # 안전

    sx = usable_w / max(1.0, float(W1))
    sy = sx

    # W6를 아래로 내릴 보정값(픽셀)
    nudge6 = int(round((W6 / 2.0) * sy));  

        # 최종 캔버스 높이: 실제 그림 높이 + 위/아래 여백
    H_px = int(round(W2 * sy))
    CANVAS_H = H_px + (MARGIN + EXTRA_T) + (MARGIN + EXTRA_B)


    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    
    x0 = MARGIN + EXTRA_L
    y0 = MARGIN + EXTRA_T 

    def X(mm): return int(round(x0 + mm * sx))
    def Y(mm): return int(round(y0 + mm * sy))

    drw.rectangle([X(0), Y(0), X(W1), Y(W2)], outline="black", width=3)

    notch_x0, notch_x1 = W1 - W5, W1
    notch_y0, notch_y1 = 0, W6
    drw.rectangle([X(notch_x0), Y(notch_y0), X(notch_x1), Y(notch_y1)], fill="white", outline="white")
    drw.line([X(notch_x0), Y(0), X(notch_x0), Y(W6)], fill="black", width=3)
    drw.line([X(notch_x0), Y(W6), X(W1),       Y(W6)], fill="black", width=3)

    if has_split:
        drw.line([X(W3), Y(0), X(W3), Y(W2)], fill="blue", width=3)
    
    # 라벨: 선과 6~8px 띄우고 anchor 사용
    pad = 6
    try:
        drw.text((X(W1/2),   Y(W2) + pad), "W1", fill="black", font=FONT, anchor="mt")
        drw.text((X(0) - pad, Y(W2/2)),    "W2", fill="black", font=FONT, anchor="rm")
        drw.text((X((W1 - W5)/2), Y(0) - pad), "W3", fill="black", font=FONT, anchor="mb")
        drw.text((X(notch_x0) - pad, Y(W6/2)),  "W4", fill="black", font=FONT, anchor="rm")
        drw.text((X(W1 - W5/2), Y(W6) + pad),   "W5", fill="black", font=FONT, anchor="mt")
        drw.text((X(W1) + pad,  Y(W2/2)+ nudge6),       "W6", fill="black", font=FONT, anchor="lm")
    except Exception:
        # anchor 미지원 폴백(폭/높이 반영)
        drw.text((X(W1/2) - w1/2,   Y(W2) + pad), "W1", fill="black", font=FONT)
        drw.text((X(0) - pad - w2,  Y(W2/2) - h2/2), "W2", fill="black", font=FONT)
        drw.text((X((W1 - W5)/2) - w3/2, Y(0) - pad - h3), "W3", fill="black", font=FONT)
        drw.text((X(notch_x0) - pad - w4, Y(W6/2) - h4/2), "W4", fill="black", font=FONT)
        drw.text((X(W1 - W5/2) - w5/2, Y(W6) + pad), "W5", fill="black", font=FONT)
        drw.text((X(W1) + pad,  Y(W2/2) - h6/2+ nudge6), "W6", fill="black", font=FONT)
    return img

def jendai_overlays_for_wall(
    wall_id: int,
    width_mm: int,
    height_mm: int,
    j_enabled: bool,
    j_depth: int,
    j_h: int,
    j_contact_walls: Optional[List[int]],
) -> List[Tuple[int,int,int,int]]:
    """접벽이면 젠다이 설치공간(검정 칠) 오버레이: (x0,x1,y0,y1) in mm"""
    if not (j_enabled and j_contact_walls and (wall_id in j_contact_walls)):
        return []
    depth = max(0, min(int(j_depth), int(width_mm)))
    jh    = max(0, min(int(j_h), int(height_mm)))
    if depth == 0 or jh == 0:
        return []
    return [(0, depth, 0, jh)]

def draw_wall_elevation_with_faces(
    wall_label_str: str,
    width_mm: int,
    height_mm: int,
    faces: List[FaceSpec],
    target_h_px: int = 280,
    margin: int = 16,
    overlays: Optional[List[Tuple[int,int,int,int]]] = None,  # (x0,x1,y0,y1) in mm
) -> Image.Image:
    usable_h = target_h_px - 2 * margin
    s = usable_h / max(1.0, float(height_mm))
    W = int(round(width_mm  * s))
    H = int(round(height_mm * s))
    CANVAS_W = int(W + 2 * margin)
    CANVAS_H = int(target_h_px + 28)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = margin, margin + 20
    x1, y1 = x0 + W, y0 + H

    drw.text((margin, 4), f"{wall_label_str} : {width_mm}×{height_mm} mm", fill="black")
    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    # 설치공간 오버레이
    if overlays:
        for (ox0, ox1, oy0, oy1) in overlays:
            fx0 = x0 + int(round(ox0 * s))
            fx1 = x0 + int(round(ox1 * s))
            fy0 = y1 - int(round(oy0 * s))
            fy1 = y1 - int(round(oy1 * s))
            drw.rectangle([fx0, fy1, fx1, fy0], outline="black", fill="black", width=2)

    # 실제 Face 칸
    for f in faces:
        fx0 = x0 + int(round(f.x0 * s))
        fx1 = x0 + int(round(f.x1 * s))
        fy0 = y1 - int(round(f.y0 * s))
        fy1 = y1 - int(round(f.y1 * s))
        drw.rectangle([fx0, fy1, fx1, fy0], outline="#666666", width=2)
        cx = (fx0 + fx1) // 2
        cy = (fy0 + fy1) // 2
        drw.text((cx - 14, cy - 7), f.face_label, fill="black")
    return img

# =========================================================
# 4) 통합 파이프라인 (Face → 새 엔진 → 집계)
# =========================================================
def collect_all_faces(
    shape: str,
    widths: Dict[int,int],
    H_eff: int,
    door_wall: Optional[int],
    door_s: Optional[float],
    door_e: Optional[float],
    j_enabled: bool,
    j_wall: Optional[int],
    j_has_step: bool,
    j_h: int,
    j_depth: int,
    j_lower_segments_map: Dict[int, List[int]],
    j_contact_walls: List[int],
) -> List[FaceSpec]:
    all_faces: List[FaceSpec] = []
    for wid, Wk in widths.items():
        door_tuple = None
        if (door_wall is not None) and (int(door_wall)==wid) and (door_s is not None) and (door_e is not None):
            door_tuple = (float(door_s), float(door_e))
        faces = build_faces_for_wall(
            shape=shape,
            wall_id=int(wid),
            width_mm=int(Wk),
            height_mm=int(H_eff),
            door_tuple=door_tuple,
            j_enabled=j_enabled,
            j_wall=(int(j_wall) if j_wall is not None else None),
            j_has_step=j_has_step,
            j_h=int(j_h),
            j_depth=int(j_depth),
            j_lower_segments=j_lower_segments_map.get(int(wid), None),
            j_contact_walls=j_contact_walls,
        )
        all_faces.extend(faces)
    return all_faces

def panels_for_faces_new_engine(faces: List[FaceSpec], TH: int, TW: int):
    """
    Face → (새 엔진 layout_report) → 패널/오류 수집
    """
    rows, errs = [], []

    for f in faces:
        if int(f.width_mm) <= 0 or int(f.height_mm) <= 0:
            errs.append({
                "벽": f.wall_label, "벽면": f.face_label,
                "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                "타일": f"{TH}×{TW}", "error": "INVALID_FACE_SIZE", "분할사유": getattr(f, "note", "")
            })
            continue

        try:
            rpt = layout_report(int(f.width_mm), int(f.height_mm), TH, TW)
            # face-level meta
            horiz = rpt.get("horiz_branch", "")
            columns = rpt.get("columns", [])
            # rows (=panels)
            for p in rpt.get("panels", []):
                rows.append({
                    "벽": f.wall_label,
                    "벽면": f.face_label,
                    "타일": f"{TH}×{TW}",
                    "가로분해": horiz,
                    "세로규칙": p.get("row_note","") or "",
                    "열": p["col"], "행": p["row"],
                    "panel_w": int(p["panel_w"]), "panel_h": int(p["panel_h"]),
                    "col_tags": p.get("col_tags",""), "row_tags": p.get("row_tags",""),
                    "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                })
        except Exception as ex:
            errs.append({
                "벽": f.wall_label, "벽면": f.face_label,
                "face_w": int(f.width_mm), "face_h": int(f.height_mm),
                "타일": f"{TH}×{TW}", "error": str(ex), "분할사유": getattr(f, "note", "")
            })

    return rows, errs

# =========================================================
# 5) UI
# =========================================================
st.title("벽판 규격/개수 산출 (통합 · New Layout Engine)")

with st.sidebar:
    st.header("기본 입력")
    shape = st.radio("욕실형태", ["사각형", "코너형"], horizontal=True)
    split_kind = st.radio("세면/샤워 구분", ["구분 없음", "구분 있음"], horizontal=True)
    H = st.number_input("벽 높이 H (mm)", min_value=300, value=2200, step=50)
    floor_type = st.radio("바닥판 유형", ["PVE", "그외(GRP/FRP)"], horizontal=True)
    tile = st.selectbox("벽타일 규격", ["300×600", "250×400"])
    # floor 연동: 바닥이 PVE면 자동으로 기본값 설정
    floor_res = st.session_state.get(FLOOR_RESULT_KEY)  # {'material': 'PVE' | 'GRP'...}
    if floor_res:
        mat = str(floor_res.get("material","")).upper()
        # 사이드바 라디오에 반영(이미 선언된 floor_type 변수를 덮어씀)
        floor_type = "PVE" if "PVE" in mat else "그외(GRP/FRP)"
        st.sidebar.info(f"바닥 재질 자동 반영: {floor_type}")
    H_eff = effective_height(H, floor_type)

    st.divider()
    st.subheader("문(도어) 설정")
    door_wall = st.number_input("문벽 번호", min_value=1, max_value=(4 if shape=="사각형" else 6), value=1, step=1)
    door_s = st.number_input("문 시작점 s (mm)", min_value=0.0, value=0.0, step=10.0)
    door_d = st.number_input("문 폭 d (mm)", min_value=0.0, value=800.0, step=10.0)

    st.divider()
    st.subheader("젠다이")
    j_enabled = st.checkbox("젠다이 있음")
    j_wall = None
    j_has_step = False
    j_h = 1000
    j_depth = 0
    j_lower_segments_map: Dict[int, List[int]] = {}
    j_contact_walls: List[int] = []

    if j_enabled:
        j_wall = st.number_input("젠다이 벽 번호", min_value=1, max_value=(4 if shape=="사각형" else 6), value=1, step=1)
        j_h = st.number_input("젠다이 높이 (mm)", min_value=50, value=1000, step=10)
        j_depth = st.number_input("젠다이 깊이 (mm)", min_value=0, value=300, step=10)

        j_has_step = st.radio("젠다이 단차", ["없음", "있음"], horizontal=True) == "있음"
        if j_has_step:
            if shape == "사각형":
                st.markdown("하부 분할(사각형): 왼쪽→오른쪽 순서 ①②③")
                w1 = st.number_input("하부 ① 폭 (mm)", min_value=0, value=600, step=10)
                w2 = st.number_input("하부 ② 폭 (mm)", min_value=0, value=600, step=10)
                w3 = st.number_input("하부 ③ 폭 (mm)", min_value=0, value=600, step=10)
                j_lower_segments_map[int(j_wall)] = [int(w1), int(w2), int(w3)]
            else:
                st.markdown("하부 분할(코너형): 왼쪽→오른쪽 순서 ①②")
                w1 = st.number_input("하부 ① 폭 (mm)", min_value=0, value=600, step=10, key="corner_step1")
                w2 = st.number_input("하부 ② 폭 (mm)", min_value=0, value=600, step=10, key="corner_step2")
                j_lower_segments_map[int(j_wall)] = [int(w1), int(w2)]

        st.markdown("**젠다이 접벽(옆벽) 지정** — 젠다이와 맞닿는 옆벽(최대 2개)")
        candidates = list(range(1, 5)) if shape == "사각형" else list(range(1, 7))
        j_contact_walls = st.multiselect(
            "접벽 벽 번호 선택", options=candidates, default=[], max_selections=2,
            help="선택된 벽은 폭=깊이, 높이=(벽높이-젠다이높이) 면과 나머지 면으로 분할됩니다. 설치공간은 검정 오버레이로만 표시됩니다."
        )
        if j_contact_walls and (int(j_wall) in j_contact_walls):
            st.error("젠다이 벽과 접벽을 동일 벽으로 지정할 수 없습니다.")
            j_contact_walls = [w for w in j_contact_walls if w != int(j_wall)]

    st.divider()
    calc = st.button("계산 & 미리보기", type="primary")

    

errors: List[str] = []
preview_img: Optional[Image.Image] = None

if shape == "사각형":
    st.subheader("사각형 입력")
    colA, colB = st.columns(2)
    with colA:
        BL = st.number_input("욕실 길이 BL (mm)", min_value=500, value=2000, step=50)
    with colB:
        BW = st.number_input("욕실 폭 BW (mm)", min_value=500, value=1600, step=50)

    X = None
    if split_kind == "구분 있음":
        X = st.slider("세면/샤워 경계 위치 X (mm)", min_value=100, max_value=int(BL), step=50, value=min(800, int(BL)))

    door_W = rect_wall_width_of(int(door_wall), int(BW), int(BL))

    # 젠다이 단차 검증
    if j_enabled and j_has_step and (j_wall is not None):
        target_w = rect_wall_width_of(int(j_wall), int(BW), int(BL))
        segs = [int(x) for x in (j_lower_segments_map.get(int(j_wall), []) or [])]
        need = 3
        if len(segs) < need:
            errors.append(f"사각형 단차: 하부 분할 폭은 {need}개가 필요합니다.")
        elif sum(segs) != target_w:
            errors.append(f"하부 분할 폭 합({sum(segs)}) ≠ 해당 벽폭({target_w})")

    if j_enabled and (int(door_wall) == int(j_wall or -999)):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
            door_draw_info = (int(door_wall), s, e, int(door_W))
        except Exception as ex:
            errors.append(str(ex))
            door_draw_info = None

        if errors:
            for msg in errors: st.error(msg)
        else:
            preview_img = draw_rect_preview(
                BL=int(BL), BW=int(BW),
                has_split=(split_kind=="구분 있음"),
                X=(int(X) if X is not None else None),
                door_info=door_draw_info
            )
            st.image(preview_img, caption="사각형 도면(평면) 미리보기", width=max(160, int(preview_img.width/2)))

            widths = {1:int(BL), 2:int(BW), 3:int(BL), 4:int(BW)}
            st.subheader("벽면(정면도) / 라벨: WnF#")

            cols = st.columns(2)
            TH, TW = parse_tile(tile)

            # 정면도 렌더 + Face 수집
            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1,2,3,4]):
                Wk = widths[wid]
                door_tuple = (float(s), float(e)) if (door_draw_info and int(door_wall)==wid) else None
                faces = collect_all_faces(
                    shape="사각형",
                    widths={wid: Wk},
                    H_eff=int(H_eff),
                    door_wall=(wid if door_tuple else None),
                    door_s=(door_tuple[0] if door_tuple else None),
                    door_e=(door_tuple[1] if door_tuple else None),
                    j_enabled=j_enabled,
                    j_wall=(int(j_wall) if j_enabled else None),
                    j_has_step=j_has_step,
                    j_h=int(j_h),
                    j_depth=int(j_depth),
                    j_lower_segments_map=j_lower_segments_map,
                    j_contact_walls=j_contact_walls,
                )
                faces = [f for f in faces if f.wall_id == wid]
                all_faces.extend(faces)

                overlays = jendai_overlays_for_wall(
                    wall_id=wid, width_mm=Wk, height_mm=int(H_eff),
                    j_enabled=j_enabled, j_depth=int(j_depth), j_h=int(j_h),
                    j_contact_walls=j_contact_walls,
                )
                img = draw_wall_elevation_with_faces(
                    wall_label("사각형", wid), Wk, int(H_eff), faces,
                    target_h_px=280, overlays=overlays
                )
                with cols[i%2]:
                    st.image(img, caption=f"{wall_label('사각형', wid)} (벽면 {len(faces)}개)", width="content")

            # 새 엔진으로 패널 산출
            st.subheader("벽면별 벽판 산출 (New Engine)")
            rows, errs = panels_for_faces_new_engine(all_faces, TH, TW)
            if rows:
                df = (pd.DataFrame(rows)
                      .rename(columns={
                          "face_w": "벽면폭", "face_h": "벽면높이",
                          "panel_w": "벽판폭", "panel_h": "벽판높이",
                          "가로분해": "가로분해(분기)", "세로규칙": "세로규칙(노트)"
                      }))
                show_cols = ["벽","벽면","타일","가로분해(분기)","세로규칙(노트)","열","행","벽판폭","벽판높이","벽면폭","벽면높이","col_tags","row_tags"]
                df = df[[c for c in show_cols if c in df.columns]]
                st.dataframe(df, width="stretch")

                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (df.groupby(["벽판폭","벽판높이"], as_index=False)
                           .size().rename(columns={"size":"qty"}))
                order["치수"] = order["벽판폭"].astype(int).astype(str) + "×" + order["벽판높이"].astype(int).astype(str)
                order = order[["치수","qty","벽판폭","벽판높이"]]
                st.dataframe(order, width="stretch")
                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                df_err = (pd.DataFrame(errs).rename(columns={"face_w":"벽면폭","face_h":"벽면높이"}))
                st.dataframe(df_err, width="stretch")

            # ====== 자동저장: 벽판 결과를 session_state에 기록 ======
            try:
                # rows, errs가 이미 계산되어 있다고 가정
                # 필수 입력 요약도 같이 저장합니다.
                st.session_state["wall_result"] = {
                    "section": "wall",
                    "inputs": {
                        "shape": shape,
                        "split_kind": split_kind,
                        "H": int(H),
                        "H_eff": int(H_eff),
                        "floor_type": floor_type,
                        "tile": tile,
                        "door_wall": (int(door_wall) if 'door_wall' in locals() else None),
                        "door_s": (float(door_s) if 'door_s' in locals() else None),
                        "door_d": (float(door_d) if 'door_d' in locals() else None),
                        "j_enabled": bool(j_enabled),
                        "j_wall": (int(j_wall) if j_enabled and (j_wall is not None) else None),
                        "j_has_step": bool(j_has_step),
                        "j_h": (int(j_h) if j_enabled else 0),
                        "j_depth": (int(j_depth) if j_enabled else 0),
                        "j_contact_walls": (j_contact_walls if j_enabled else []),
                    },
                    "result": {
                        "panels": rows,         # panels_for_faces_new_engine()에서 받아온 rows
                        "errors": errs,         # 같은 함수에서의 errs
                        # 필요하면 아래처럼 통계치도 추가
                        "counts": {
                            "n_panels": len(rows),
                            "n_errors": len(errs),
                        },
                    }
                }
                st.success("벽판 결과 자동저장 완료")
            except Exception as _e:
                st.warning(f"벽판 결과 자동저장 중 오류: {_e}")
                
else:
    # 코너형
    st.subheader("코너형 입력 (W1~W6)")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**가로(바닥) 방향**")
        W3 = st.number_input("W3 (mm)", min_value=100, value=800, step=50, key="corner_w3")
        W5 = st.number_input("W5 (mm)", min_value=100, value=1200, step=50, key="corner_w5")
        W1 = W3 + W5
        st.text_input("W1 = W3 + W5", value=str(W1), disabled=True)
    with cB:
        st.markdown("**세로(좌우) 방향**")
        W4 = st.number_input("W4 (mm)", min_value=100, value=600, step=50, key="corner_w4")
        W6 = st.number_input("W6 (mm)", min_value=100, value=1000, step=50, key="corner_w6")
        W2 = W4 + W6
        st.text_input("W2 = W4 + W6", value=str(W2), disabled=True)

    W = {1:int(W1), 2:int(W2), 3:int(W3), 4:int(W4), 5:int(W5), 6:int(W6)}
    door_W = corner_wall_width_of(int(door_wall), W)

    if j_enabled and j_has_step and (j_wall is not None):
        target_w = corner_wall_width_of(int(j_wall), W)
        segs = [int(x) for x in (j_lower_segments_map.get(int(j_wall), []) or [])]
        need = 2
        if len(segs) < need:
            errors.append(f"코너형 단차: 하부 분할 폭은 {need}개가 필요합니다.")
        elif sum(segs) != target_w:
            errors.append(f"하부 분할 폭 합({sum(segs)}) ≠ 해당 벽폭({target_w})")

    if j_enabled and 'j_wall' in locals() and int(door_wall) == int(j_wall):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        errors.extend(validate_corner_dims(W))
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
        except Exception as ex:
            errors.append(str(ex))

        if errors:
            for msg in errors: st.error(msg)
        else:
            preview_img = draw_corner_preview(W=W, has_split=(split_kind=="구분 있음"), canvas_w=480, margin = 30)
            st.image(preview_img, caption="코너형 도면(평면) 미리보기", width=preview_img.width)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            widths = {i:int(W[i]) for i in range(1,7)}
            st.subheader("벽면(정면도) / 라벨: WnF#")
            cols = st.columns(3)
            TH, TW = parse_tile(tile)

            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1,2,3,4,5,6]):
                Wk = widths[wid]
                door_tuple = (float(s), float(e)) if int(door_wall)==wid else None
                faces = collect_all_faces(
                    shape="코너형",
                    widths={wid: Wk},
                    H_eff=int(H_eff),
                    door_wall=(wid if door_tuple else None),
                    door_s=(door_tuple[0] if door_tuple else None),
                    door_e=(door_tuple[1] if door_tuple else None),
                    j_enabled=j_enabled,
                    j_wall=(int(j_wall) if j_enabled else None),
                    j_has_step=j_has_step,
                    j_h=int(j_h),
                    j_depth=int(j_depth),
                    j_lower_segments_map=j_lower_segments_map,
                    j_contact_walls=j_contact_walls,
                )
                faces = [f for f in faces if f.wall_id == wid]
                all_faces.extend(faces)

                overlays = jendai_overlays_for_wall(
                    wall_id=wid, width_mm=Wk, height_mm=int(H_eff),
                    j_enabled=j_enabled, j_depth=int(j_depth), j_h=int(j_h),
                    j_contact_walls=j_contact_walls,
                )
                img = draw_wall_elevation_with_faces(
                    wall_label("코너형", wid), Wk, int(H_eff), faces,
                    target_h_px=280, overlays=overlays
                )
                with cols[i%3]:
                    st.image(img, caption=f"{wall_label('코너형', wid)} (벽면 {len(faces)}개)", width="content")

            # 새 엔진으로 패널 산출
            st.subheader("벽면별 벽판 산출 (New Engine)")
            rows, errs = panels_for_faces_new_engine(all_faces, TH, TW)
            if rows:
                df = (pd.DataFrame(rows)
                      .rename(columns={
                          "face_w":"벽면폭","face_h":"벽면높이",
                          "panel_w":"벽판폭","panel_h":"벽판높이",
                          "가로분해":"가로분해(분기)","세로규칙":"세로규칙(노트)"
                      }))
                show_cols = ["벽","벽면","타일","가로분해(분기)","세로규칙(노트)","열","행","벽판폭","벽판높이","벽면폭","벽면높이","col_tags","row_tags"]
                df = df[[c for c in show_cols if c in df.columns]]
                st.dataframe(df, width="stretch")

                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (df.groupby(["벽판폭","벽판높이"], as_index=False)
                           .size().rename(columns={"size":"qty"}))
                order["치수"] = order["벽판폭"].astype(int).astype(str) + "×" + order["벽판높이"].astype(int).astype(str)
                order = order[["치수","qty","벽판폭","벽판높이"]]
                st.dataframe(order, width="stretch")
                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                st.dataframe(pd.DataFrame(errs).rename(columns={"face_w":"벽면폭","face_h":"벽면높이"}), width="stretch")
                # ====== 자동저장: 벽판 결과를 session_state에 기록 ======
            try:
                # rows, errs가 이미 계산되어 있다고 가정
                # 필수 입력 요약도 같이 저장합니다.
                st.session_state["wall_result"] = {
                    "section": "wall",
                    "inputs": {
                        "shape": shape,
                        "split_kind": split_kind,
                        "H": int(H),
                        "H_eff": int(H_eff),
                        "floor_type": floor_type,
                        "tile": tile,
                        "door_wall": (int(door_wall) if 'door_wall' in locals() else None),
                        "door_s": (float(door_s) if 'door_s' in locals() else None),
                        "door_d": (float(door_d) if 'door_d' in locals() else None),
                        "j_enabled": bool(j_enabled),
                        "j_wall": (int(j_wall) if j_enabled and (j_wall is not None) else None),
                        "j_has_step": bool(j_has_step),
                        "j_h": (int(j_h) if j_enabled else 0),
                        "j_depth": (int(j_depth) if j_enabled else 0),
                        "j_contact_walls": (j_contact_walls if j_enabled else []),
                    },
                    "result": {
                        "panels": rows,         # panels_for_faces_new_engine()에서 받아온 rows
                        "errors": errs,         # 같은 함수에서의 errs
                        # 필요하면 아래처럼 통계치도 추가
                        "counts": {
                            "n_panels": len(rows),
                            "n_errors": len(errs),
                        },
                    }
                }
                st.success("벽판 결과 자동저장 완료")
            except Exception as _e:
                st.warning(f"벽판 결과 자동저장 중 오류: {_e}")

st.caption("※ 새 엔진 적용: 2400 모듈 + 가로/세로 발란스 규칙 통합, 최대 9600까지 확장. 젠다이 높이/깊이·단차·접벽 로직 유지. 설치공간은 정면도 검정 오버레이로만 표시하며 집계에서 제외됩니다.")
