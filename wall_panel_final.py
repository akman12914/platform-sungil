# app.py
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# --- design refresh (prettier inline) ---
import streamlit as st


def _design_refresh(title: str, subtitle: str = ""):
    try:
        st.set_page_config(page_title=title, layout="wide")
    except Exception:
        pass
    st.markdown(
        """
    <style>
      :root {
        --brand: #2563eb;
        --brand-light: #3b82f6;
        --ink: #1e293b;
        --muted: #64748b;
        --panel: #f9fafb;
      }
      .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        padding: .55rem 1rem;
        font-weight: 600;
        border: none;
        background: var(--brand);
        color: white;
        transition: background .2s ease;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        background: var(--brand-light);
        color: #fff;
      }
      .app-card {
        background: var(--panel);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
      }
      .titlebar h1 {
        margin: 0 0 .3rem 0;
        color: var(--ink);
        font-size: 1.5rem;
      }
      .titlebar .sub {
        color: var(--muted);
        font-size: .95rem;
        margin-bottom: .5rem;
      }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='titlebar'><h1>{title}</h1>"
        + (f"<div class='sub'>{subtitle}</div>" if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


# --- end design refresh ---

_design_refresh("벽판 계산기", "UI 정리 · 사이드바 유지")


import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw

st.set_page_config(page_title="벽판 규격/개수 산출 (통합)", layout="wide")


# =========================================================
# 0) 공통 유틸
# =========================================================
def parse_tile(tile_str: str) -> Tuple[int, int]:
    """'300×600' 또는 '250×400' → (300, 600)"""
    a, b = tile_str.replace("x", "×").split("×")
    return int(a), int(b)


def effective_height(H: int, floor_type: str) -> int:
    """바닥판 유형이 PVE면 +50."""
    return int(H) + 50 if floor_type.upper() == "PVE" else int(H)


# =========================================================
# 1) 엔진 (벽면 폭/높이 + 타일 규격 → 벽판 분할)
#    (기존 코드2와 동일 로직; 그대로 통합)
# =========================================================
MIN_EDGE = 80


def iround(x: float) -> int:
    return int(math.floor(x + 0.5))


@dataclass
class Panel:
    kind: str
    pos: str
    w: int
    h: int

    def label(self) -> str:
        base = self.kind
        if self.pos:
            base += f"({self.pos})"
        return f"{base} {self.w}×{self.h}"


class RuleError(Exception):
    pass


def vertical_balance_round(H_target: int, TH: int) -> Tuple[int, int, int]:
    if H_target < 2 * TH:
        raise RuleError("HEIGHT_TOO_SMALL_FOR_BALANCE")
    m = iround(H_target / (2 * TH))
    top = m * TH
    bot = H_target - top
    return m, top, bot


def horizontal_balance_round(W_target: int, TW: int) -> Tuple[int, int, int]:
    if W_target < 2 * TW:
        raise RuleError("WIDTH_TOO_SMALL_FOR_BALANCE")
    n = iround(W_target / (2 * TW))
    left = n * TW
    right = W_target - left
    return n, left, right


def ensure_producible(panels: List[Panel]):
    for p in panels:
        if p.w <= MIN_EDGE or p.h <= MIN_EDGE:
            raise RuleError(f"PANEL_TOO_SMALL_TO_PRODUCE: {p.label()}")


def layout_300x600(W: int, H: int) -> Tuple[List[Panel], str]:
    TH, TW = 300, 600
    panels: List[Panel] = []
    label = ""

    if (W <= 1000) and (H <= 2400):
        label = "W<=1000 & H<=2400 : 세로판1"
        panels.append(Panel("세로판", "", W, H))

    elif (W <= 1000) and (2400 < H <= 4800):
        label = "W<=1000 & 2400<H<=4800 : 세로판2(m2,round)"
        _, top, bot = vertical_balance_round(H, TH)
        panels += [Panel("세로판", "u", W, top), Panel("세로판", "d", W, bot)]

    elif (1000 < W <= 2400) and (H <= 1200):
        label = "1000<W<=2400 & H<=1200 : 1장"
        panels.append(Panel("벽판", "", W, H))

    elif (1000 < W <= 2400) and (1200 < H <= 2400):
        label = "1000<W<=2400 & 1200<H<=2400 : 상1 하1"
        panels += [Panel("상부판", "", W, 1200), Panel("하부판", "", W, H - 1200)]

    elif (1000 < W <= 2400) and (2400 < H <= 3600):
        label = "1000<W<=2400 & 2400<H<=3600 : 상1 하2(round)"
        panels.append(Panel("상부판", "", W, 1200))
        newH = H - 1200
        _, top, bot = vertical_balance_round(newH, TH)
        panels += [Panel("하부판", "u", W, top), Panel("하부판", "d", W, bot)]

    elif (1000 < W <= 2400) and (3600 < H <= 4800):
        label = "1000<W<=2400 & 3600<H<=4800 : 상2 하2(round)"
        panels += [Panel("상부판", "u", W, 1200), Panel("상부판", "d", W, 1200)]
        newH = H - 2400
        _, top, bot = vertical_balance_round(newH, TH)
        panels += [Panel("하부판", "u", W, top), Panel("하부판", "d", W, bot)]

    elif (2400 < W <= 4800) and (H <= 1200):
        label = "2400<W<=4800 & H<=1200 : 가로발란스2(round)"
        _, Lw, Rw = horizontal_balance_round(W, TW)
        panels += [Panel("벽판", "l", Lw, H), Panel("벽판", "r", Rw, H)]

    elif (2400 < W <= 3400) and (1200 < H <= 2400):
        label = "2400<W<=3400 & 1200<H<=2400 : 상1 하1 + 세로1(ΔW)"
        dW = W - 2400
        if dW <= 80:
            panels += [
                Panel("상부판", "", 2400 - TW, 1200),
                Panel("하부판", "", 2400 - TW, H - 1200),
                Panel("세로판", "", W - (2400 - TW), H),
            ]
        else:
            panels += [
                Panel("상부판", "", 2400, 1200),
                Panel("하부판", "", 2400, H - 1200),
                Panel("세로판", "", W - 2400, H),
            ]

    elif (2400 < W <= 3400) and (2400 < H <= 3600):
        label = "2400<W<=3400 & 2400<H<=3600 : 상1 + 세로2(m2) + 하2(m1,round)"
        dW = W - 2400
        if dW <= 80:
            panels.append(Panel("상부판", "", 2400 - 600, 1200))
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - (2400 - 600), vtop),
                Panel("세로판", "d", W - (2400 - 600), vbot),
            ]
            newH = H - 1200
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400 - 600, btop),
                Panel("하부판", "d", 2400 - 600, bbot),
            ]
        else:
            panels.append(Panel("상부판", "", 2400, 1200))
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - 2400, vtop),
                Panel("세로판", "d", W - 2400, vbot),
            ]
            newH = H - 1200
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400, btop),
                Panel("하부판", "d", 2400, bbot),
            ]

    elif (2400 < W <= 3400) and (3600 < H <= 4800):
        label = "2400<W<=3400 & 3600<H<=4800 : 상2 + 세로2(m2) + 하2(m1,round)"
        dW = W - 2400
        if dW <= 80:
            panels += [
                Panel("상부판", "u", 2400 - 600, 1200),
                Panel("상부판", "d", 2400 - 600, 1200),
            ]
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - (2400 - 600), vtop),
                Panel("세로판", "d", W - (2400 - 600), vbot),
            ]
            newH = H - 2400
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400 - 600, btop),
                Panel("하부판", "d", 2400 - 600, bbot),
            ]
        else:
            panels += [
                Panel("상부판", "u", 2400, 1200),
                Panel("상부판", "d", 2400, 1200),
            ]
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - 2400, vtop),
                Panel("세로판", "d", W - 2400, vbot),
            ]
            newH = H - 2400
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400, btop),
                Panel("하부판", "d", 2400, bbot),
            ]

    elif (3400 < W <= 4800) and (1200 < H <= 2400):
        label = "3400<W<=4800 & 1200<H<=2400 : 좌우 가로발란스 + 상1하1"
        _, Lw, Rw = horizontal_balance_round(W, 600)
        panels += [
            Panel("상부판", "l", Lw, 1200),
            Panel("상부판", "r", Rw, 1200),
            Panel("하부판", "l", Lw, H - 1200),
            Panel("하부판", "r", Rw, H - 1200),
        ]

    elif (3400 < W <= 4800) and (2400 < H <= 3600):
        label = "3400<W<=4800 & 2400<H<=3600 : 좌우 가로발란스 + 상2하4(m1,round)"
        _, Lw, Rw = horizontal_balance_round(W, 600)
        panels += [Panel("상부판", "l", Lw, 1200), Panel("상부판", "r", Rw, 1200)]
        newH = H - 1200
        _, btop, bbot = vertical_balance_round(newH, TH)
        panels += [
            Panel("하부판", "l,u", Lw, btop),
            Panel("하부판", "r,u", Rw, btop),
            Panel("하부판", "l,d", Lw, bbot),
            Panel("하부판", "r,d", Rw, bbot),
        ]

    elif (3400 < W <= 4800) and (3600 < H <= 4800):
        label = "3400<W<=4800 & 3600<H<=4800 : 좌우 가로발란스 + 상4하4(m1,round)"
        _, Lw, Rw = horizontal_balance_round(W, 600)
        panels += [
            Panel("상부판", "l,u", Lw, 1200),
            Panel("상부판", "r,u", Rw, 1200),
            Panel("상부판", "l,d", Lw, 1200),
            Panel("상부판", "r,d", Rw, 1200),
        ]
        newH = H - 2400
        _, btop, bbot = vertical_balance_round(newH, TH)
        panels += [
            Panel("하부판", "l,u", Lw, btop),
            Panel("하부판", "r,u", Rw, btop),
            Panel("하부판", "l,d", Lw, bbot),
            Panel("하부판", "r,d", Rw, bbot),
        ]

    else:
        raise RuleError("NO_RULE_MATCHED_300x600")

    ensure_producible(panels)
    return panels, label


def layout_250x400(W: int, H: int) -> Tuple[List[Panel], str]:
    TH, TW = 250, 400
    panels: List[Panel] = []
    label = ""

    if (W <= 1000) and (H <= 2200):
        label = "W<=1000 & H<=2200 : 세로판1"
        panels.append(Panel("세로판", "", W, H))

    elif (W <= 1000) and (2200 < H <= 4200):
        label = "W<=1000 & 2200<H<=4200 : 세로판2(m2,round)"
        _, top, bot = vertical_balance_round(H, TH)
        panels += [Panel("세로판", "u", W, top), Panel("세로판", "d", W, bot)]

    elif (1000 < W <= 2400) and (H <= 1200):
        label = "1000<W<=2400 & H<=1200 : 1장"
        panels.append(Panel("벽판", "", W, H))

    elif (1000 < W <= 2400) and (1200 < H <= 2200):
        label = "1000<W<=2400 & 1200<H<=2200 : 상1 하1"
        panels += [Panel("상부판", "", W, 1000), Panel("하부판", "", W, H - 1000)]

    elif (1000 < W <= 2400) and (2200 < H <= 3200):
        label = "1000<W<=2400 & 2200<H<=3200 : 상1 하2(round)"
        panels.append(Panel("상부판", "", W, 1000))
        newH = H - 1000
        _, top, bot = vertical_balance_round(newH, TH)
        panels += [Panel("하부판", "u", W, top), Panel("하부판", "d", W, bot)]

    elif (1000 < W <= 2400) and (3200 < H <= 4200):
        label = "1000<W<=2400 & 3200<H<=4200 : 상2 하2(round)"
        panels += [Panel("상부판", "u", W, 1000), Panel("상부판", "d", W, 1000)]
        newH = H - 2000
        _, top, bot = vertical_balance_round(newH, TH)
        panels += [Panel("하부판", "u", W, top), Panel("하부판", "d", W, bot)]

    elif (2400 < W <= 4800) and (H <= 1200):
        label = "2400<W<=4800 & H<=1200 : 가로발란스2(round)"
        _, Lw, Rw = horizontal_balance_round(W, TW)
        panels += [Panel("벽판", "l", Lw, H), Panel("벽판", "r", Rw, H)]

    elif (2400 < W <= 3400) and (1200 < H <= 2200):
        label = "2400<W<=3400 & 1200<H<=2200 : 상1 하1 + 세로1(ΔW)"
        dW = W - 2400
        if dW <= 80:
            panels += [
                Panel("상부판", "", 2400 - TW, 1000),
                Panel("하부판", "", 2400 - TW, H - 1000),
                Panel("세로판", "", W - (2400 - TW), H),
            ]
        else:
            panels += [
                Panel("상부판", "", 2400, 1000),
                Panel("하부판", "", 2400, H - 1000),
                Panel("세로판", "", W - 2400, H),
            ]

    elif (2400 < W <= 3400) and (2200 < H <= 3200):
        label = "2400<W<=3400 & 2200<H<=3200 : 상1 + 세로2(m2) + 하2(m1,round)"
        dW = W - 2400
        if dW <= 80:
            panels.append(Panel("상부판", "", 2400 - 400, 1000))
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - (2400 - 400), vtop),
                Panel("세로판", "d", W - (2400 - 400), vbot),
            ]
            newH = H - 1000
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400 - 400, btop),
                Panel("하부판", "d", 2400 - 400, bbot),
            ]
        else:
            panels.append(Panel("상부판", "", 2400, 1000))
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - 2400, vtop),
                Panel("세로판", "d", W - 2400, vbot),
            ]
            newH = H - 1000
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400, btop),
                Panel("하부판", "d", 2400, bbot),
            ]

    elif (2400 < W <= 3400) and (3200 < H <= 4200):
        label = "2400<W<=3400 & 3200<H<=4200 : 상2 + 세로2(m2) + 하2(m1,round)"
        dW = W - 2400
        if dW <= 80:
            panels += [
                Panel("상부판", "u", 2400 - 400, 1000),
                Panel("상부판", "d", 2400 - 400, 1000),
            ]
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - (2400 - 400), vtop),
                Panel("세로판", "d", W - (2400 - 400), vbot),
            ]
            newH = H - 2000
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400 - 400, btop),
                Panel("하부판", "d", 2400 - 400, bbot),
            ]
        else:
            panels += [
                Panel("상부판", "u", 2400, 1000),
                Panel("상부판", "d", 2400, 1000),
            ]
            _, vtop, vbot = vertical_balance_round(H, TH)
            panels += [
                Panel("세로판", "u", W - 2400, vtop),
                Panel("세로판", "d", W - 2400, vbot),
            ]
            newH = H - 2000
            _, btop, bbot = vertical_balance_round(newH, TH)
            panels += [
                Panel("하부판", "u", 2400, btop),
                Panel("하부판", "d", 2400, bbot),
            ]

    elif (3400 < W <= 4800) and (1200 < H <= 2200):
        label = "3400<W<=4800 & 1200<H<=2200 : 좌우 가로발란스 + 상1하1"
        _, Lw, Rw = horizontal_balance_round(W, 400)
        panels += [
            Panel("상부판", "l", Lw, 1000),
            Panel("상부판", "r", Rw, 1000),
            Panel("하부판", "l", Lw, H - 1000),
            Panel("하부판", "r", Rw, H - 1000),
        ]

    elif (3400 < W <= 4800) and (2200 < H <= 3200):
        label = "3400<W<=4800 & 2200<H<=3200 : 좌우 가로발란스 + 상2하4(m1,round)"
        _, Lw, Rw = horizontal_balance_round(W, 400)
        panels += [Panel("상부판", "l", Lw, 1000), Panel("상부판", "r", Rw, 1000)]
        newH = H - 1000
        _, btop, bbot = vertical_balance_round(newH, TH)
        panels += [
            Panel("하부판", "l,u", Lw, btop),
            Panel("하부판", "r,u", Rw, btop),
            Panel("하부판", "l,d", Lw, bbot),
            Panel("하부판", "r,d", Rw, bbot),
        ]

    elif (3400 < W <= 4800) and (3200 < H <= 4200):
        label = "3400<W<=4800 & 3200<H<=4200 : 좌우 가로발란스 + 상4하4(m1,round)"
        _, Lw, Rw = horizontal_balance_round(W, 400)
        panels += [
            Panel("상부판", "l,u", Lw, 1000),
            Panel("상부판", "r,u", Rw, 1000),
            Panel("상부판", "l,d", Lw, 1000),
            Panel("상부판", "r,d", Rw, 1000),
        ]
        newH = H - 2000
        _, btop, bbot = vertical_balance_round(newH, TH)
        panels += [
            Panel("하부판", "l,u", Lw, btop),
            Panel("하부판", "r,u", Rw, btop),
            Panel("하부판", "l,d", Lw, bbot),
            Panel("하부판", "r,d", Rw, bbot),
        ]

    else:
        raise RuleError("NO_RULE_MATCHED_250x400")

    ensure_producible(panels)
    return panels, label


def compute_layout(W: int, H: int, TH: int, TW: int):
    if (TH, TW) == (300, 600):
        return layout_300x600(W, H)
    elif (TH, TW) == (250, 400):
        return layout_250x400(W, H)
    else:
        raise RuleError("UNSUPPORTED_TILE_SIZE")


# =========================================================
# 2) 벽/벽면(Face) 모델 & 생성
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


def normalize_door(
    W: int, s: float, d: float
) -> Tuple[float, float, float, float, int]:
    """도어 시작/폭 정규화: 반환 (s, e, L, R, n_faces)"""
    if d <= 0 or d > W:
        raise ValueError(
            "문 폭(d)이 유효하지 않습니다. 0 < d ≤ 문벽 폭(W)을 만족해야 합니다."
        )
    s = max(0.0, min(float(s), float(W)))
    if s == W:
        s = float(W - d)
    e = s + d
    if e > W:
        raise ValueError(
            "문 범위(s+d)가 문벽 폭(W)을 초과합니다. 시작점 또는 문폭을 줄이세요."
        )
    L = s
    R = W - e
    n_faces = (1 if L > 0 else 0) + (1 if R > 0 else 0)
    return s, e, L, R, n_faces


def rect_wall_width_of(wall_id: int, BW: int, BL: int) -> int:
    """사각형: 1=상(BL), 2=우(BW), 3=하(BL), 4=좌(BW)"""
    if wall_id == 1:
        return BL
    if wall_id == 2:
        return BW
    if wall_id == 3:
        return BL
    if wall_id == 4:
        return BW
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
    x0: int
    x1: int
    y0: int
    y1: int
    width_mm: int
    height_mm: int
    note: str  # "door-left"/"door-right"/"jendai-lower"/"jendai-upper"/"single"


def wall_label(shape: str, wall_id: int) -> str:
    return f"W{wall_id}"


def build_faces_for_wall(
    shape: str,
    wall_id: int,
    width_mm: int,
    height_mm: int,
    door_tuple: Optional[Tuple[float, float]] = None,  # (s,e) mm
    j_faces: Optional[List[int]] = None,
    band_h: int = 1000,
) -> List[FaceSpec]:
    """문/젠다이를 반영해 한 벽을 여러 '벽면'으로 분해한다."""
    wl = wall_label(shape, wall_id)
    faces: List[FaceSpec] = []
    s_mm = e_mm = None
    if door_tuple is not None:
        s_mm, e_mm = int(round(door_tuple[0])), int(round(door_tuple[1]))

    # A) 문 분할 (좌/우)
    if door_tuple is not None:
        L = s_mm
        R = max(0, width_mm - e_mm)
        fi = 1
        if L > 0:
            faces.append(
                FaceSpec(
                    wall_id,
                    wl,
                    fi,
                    f"{wl}F{fi}",
                    0,
                    L,
                    0,
                    height_mm,
                    L,
                    height_mm,
                    "door-left",
                )
            )
            fi += 1
        if R > 0:
            faces.append(
                FaceSpec(
                    wall_id,
                    wl,
                    fi,
                    f"{wl}F{fi}",
                    e_mm,
                    e_mm + R,
                    0,
                    height_mm,
                    R,
                    height_mm,
                    "door-right",
                )
            )
        return faces

    # B) 젠다이 분할 (하부 n개 + 상부 1개)
    if j_faces:
        fi = 1
        bh = min(band_h, height_mm)
        acc = 0
        for w in j_faces:
            w = int(w)
            if w <= 0:
                continue
            faces.append(
                FaceSpec(
                    wall_id,
                    wl,
                    fi,
                    f"{wl}F{fi}",
                    acc,
                    acc + w,
                    0,
                    bh,
                    w,
                    bh,
                    "jendai-lower",
                )
            )
            acc += w
            fi += 1
        uh = max(0, height_mm - bh)
        if uh > 0:
            faces.append(
                FaceSpec(
                    wall_id,
                    wl,
                    fi,
                    f"{wl}F{fi}",
                    0,
                    width_mm,
                    bh,
                    bh + uh,
                    width_mm,
                    uh,
                    "jendai-upper",
                )
            )
        return faces

    # C) 분할 없음
    faces.append(
        FaceSpec(
            wall_id,
            wl,
            1,
            f"{wl}F1",
            0,
            width_mm,
            0,
            height_mm,
            width_mm,
            height_mm,
            "single",
        )
    )
    return faces


# =========================================================
# 3) 도면 렌더링 (평면도 + 정면도/벽면 라벨)
# =========================================================
def draw_rect_preview(
    BL: int,
    BW: int,
    has_split: bool,
    X: Optional[int],
    door_info: Optional[Tuple[int, float, float, int]] = None,
) -> Image.Image:
    """사각형 평면도. 라벨: W1~W4"""
    if BW > BL:
        BL, BW = BW, BL
    CANVAS_W = 760
    MARGIN = 20
    sx = (CANVAS_W - 2 * MARGIN) / max(1, float(BL))
    sy = sx
    CANVAS_H = int(BW * sy + 2 * MARGIN)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = MARGIN, MARGIN
    x1 = x0 + int(BL * sx)
    y1 = y0 + int(BW * sy)

    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

    if has_split and X is not None:
        gx = x0 + int(X * sx)
        drw.line([gx, y0, gx, y1], fill="blue", width=3)

    if door_info:
        wall_id, s, e, W_wall = door_info
        if wall_id == 1:
            xs = x0 + int(s * sx)
            xe = x0 + int(e * sx)
            y = y1
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 3:
            xs = x0 + int(s * sx)
            xe = x0 + int(e * sx)
            y = y0
            drw.line([xs, y, xe, y], fill="red", width=5)
        elif wall_id == 2:
            ys = y0 + int(s * sy)
            ye = y0 + int(e * sy)
            x = x1
            drw.line([x, ys, x, ye], fill="red", width=5)
        elif wall_id == 4:
            ys = y0 + int(s * sy)
            ye = y0 + int(e * sy)
            x = x0
            drw.line([x, ys, x, ye], fill="red", width=5)

    off = 14
    drw.text(((x0 + x1) // 2 - 12, y1 + off - 8), "W1", fill="black")
    drw.text((x1 + off, (y0 + y1) // 2 - 8), "W2", fill="black")
    drw.text(((x0 + x1) // 2 - 12, y0 - off - 8), "W3", fill="black")
    drw.text((x0 - off - 18, (y0 + y1) // 2 - 8), "W4", fill="black")
    return img


def draw_corner_preview(
    W: dict,
    has_split: bool,
    canvas_w: int = 760,
    margin: int = 20,
) -> Image.Image:
    """코너형 평면도. 라벨: W1~W6"""
    W1, W2, W3, W4, W5, W6 = (int(W[i]) for i in range(1, 7))
    CANVAS_W = int(canvas_w)
    MARGIN = int(margin)
    sx = (CANVAS_W - 2 * MARGIN) / max(1.0, float(W1))
    sy = sx
    CANVAS_H = int(W2 * sy + 2 * MARGIN)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)

    x0, y0 = MARGIN, MARGIN

    def X(mm):
        return int(round(x0 + mm * sx))

    def Y(mm):
        return int(round(y0 + mm * sy))

    drw.rectangle([X(0), Y(0), X(W1), Y(W2)], outline="black", width=3)

    notch_x0, notch_x1 = W1 - W5, W1
    notch_y0, notch_y1 = 0, W6
    drw.rectangle(
        [X(notch_x0), Y(notch_y0), X(notch_x1), Y(notch_y1)],
        fill="white",
        outline="white",
    )
    drw.line([X(notch_x0), Y(0), X(notch_x0), Y(W6)], fill="black", width=3)
    drw.line([X(notch_x0), Y(W6), X(W1), Y(W6)], fill="black", width=3)

    if has_split:
        drw.line([X(W3), Y(0), X(W3), Y(W2)], fill="blue", width=3)

    off = 14
    drw.text((X(W1 / 2), Y(W2) + off), "W1", fill="black")
    drw.text((X(0) - off, Y(W2 / 2)), "W2", fill="black")
    drw.text((X((W1 - W5) / 2), Y(0) - off), "W3", fill="black")
    drw.text((X(notch_x0) - off, Y(W6 / 2)), "W4", fill="black")
    drw.text((X(W1 - W5 / 2), Y(W6) + off), "W5", fill="black")
    drw.text((X(W1) + off, Y(W2 / 2)), "W6", fill="black")
    return img


def draw_wall_elevation_with_faces(
    wall_label_str: str,
    width_mm: int,
    height_mm: int,
    faces: List[FaceSpec],
    target_h_px: int = 280,
    margin: int = 16,
) -> Image.Image:
    usable_h = target_h_px - 2 * margin
    s = usable_h / max(1.0, float(height_mm))
    W = int(round(width_mm * s))
    H = int(round(height_mm * s))
    CANVAS_W = int(W + 2 * margin)
    CANVAS_H = int(target_h_px + 28)

    img = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    drw = ImageDraw.Draw(img)
    x0, y0 = margin, margin + 20
    x1, y1 = x0 + W, y0 + H

    drw.text((margin, 4), f"{wall_label_str} : {width_mm}×{height_mm} mm", fill="black")
    drw.rectangle([x0, y0, x1, y1], outline="black", width=3)

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
# 4) 통합 파이프라인: 벽→벽면→엔진 호출→집계
# =========================================================
def collect_all_faces(
    shape: str,
    widths: Dict[int, int],
    H_eff: int,
    door_wall: Optional[int],
    door_s: Optional[float],
    door_e: Optional[float],
    j_wall: Optional[int],
    j_faces: Optional[List[int]],
) -> List[FaceSpec]:
    wall_ids = list(range(1, 5)) if shape == "사각형" else list(range(1, 7))
    all_faces: List[FaceSpec] = []
    for wid in wall_ids:
        Wk = int(widths[wid])
        door_tuple = None
        if (
            (door_wall is not None)
            and (int(door_wall) == wid)
            and (door_s is not None)
            and (door_e is not None)
        ):
            door_tuple = (float(door_s), float(door_e))
        jf = (
            [int(v) for v in (j_faces or [])]
            if (j_wall is not None and int(j_wall) == wid and j_faces)
            else None
        )
        faces = build_faces_for_wall(
            shape=shape,
            wall_id=wid,
            width_mm=Wk,
            height_mm=int(H_eff),
            door_tuple=door_tuple,
            j_faces=jf,
            band_h=int(j_band_h),
        )
        all_faces.extend(faces)
    return all_faces


def panels_for_faces(faces: List[FaceSpec], TH: int, TW: int):
    rows, errs = [], []
    for f in faces:
        try:
            panels, branch = compute_layout(f.width_mm, f.height_mm, TH, TW)
            for i, p in enumerate(panels, 1):
                rows.append(
                    {
                        "벽": f.wall_label,
                        "벽면": f.face_label,
                        "분할사유": f.note,
                        "타일": f"{TH}×{TW}",
                        "규칙분기": branch,
                        "panel_no": i,
                        "kind": p.kind,
                        "pos": p.pos,
                        "width_mm": p.w,
                        "height_mm": p.h,
                        "face_w": f.width_mm,
                        "face_h": f.height_mm,
                    }
                )
        except Exception as ex:
            errs.append(
                {
                    "벽": f.wall_label,
                    "벽면": f.face_label,
                    "face_w": f.width_mm,
                    "face_h": f.height_mm,
                    "타일": f"{TH}×{TW}",
                    "error": str(ex),
                    "분할사유": f.note,
                }
            )
    return rows, errs


# =========================================================
# 5) UI
# =========================================================
st.title("벽판 규격/개수 산출 (통합)")

with st.sidebar:
    st.header("기본 입력")
    shape = st.radio("욕실형태", ["사각형", "코너형"], horizontal=True)
    split_kind = st.radio("세면/샤워 구분", ["구분 없음", "구분 있음"], horizontal=True)
    H = st.number_input("벽 높이 H (mm)", min_value=300, value=2200, step=50)
    floor_type = st.radio("바닥판 유형", ["PVE", "그외(GRP/FRP)"], horizontal=True)
    tile = st.selectbox("벽타일 규격", ["300×600", "250×400"])
    H_eff = effective_height(H, floor_type)

    st.divider()
    st.subheader("문(도어) 설정")
    door_wall = st.number_input(
        "문벽 번호",
        min_value=1,
        max_value=(4 if shape == "사각형" else 6),
        value=1,
        step=1,
    )
    door_s = st.number_input("문 시작점 s (mm)", min_value=0.0, value=0.0, step=10.0)
    door_d = st.number_input("문 폭 d (mm)", min_value=0.0, value=800.0, step=10.0)

    st.divider()
    st.subheader("젠다이")
    has_jendai = st.checkbox("젠다이 있음")

    j_band_h = 1000  # 기본값
    if has_jendai:
        j_band_h = st.slider(
            "젠다이 높이 (mm)", min_value=900, max_value=1000, value=1000, step=10
        )

    if has_jendai:
        j_wall = st.number_input(
            "젠다이 벽 번호",
            min_value=1,
            max_value=(4 if shape == "사각형" else 6),
            value=1,
            step=1,
        )
        j_step = st.radio("젠다이 단차", ["없음", "있음"], horizontal=True)

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
        X = st.slider(
            "세면/샤워 경계 위치 X (mm)",
            min_value=100,
            max_value=int(BL),
            step=50,
            value=min(800, int(BL)),
        )

    door_W = rect_wall_width_of(int(door_wall), int(BW), int(BL))

    j_faces = []
    if has_jendai:
        j_W = rect_wall_width_of(int(j_wall), int(BW), int(BL))
        if j_step == "있음":
            j1 = st.number_input(
                "젠다이 면1 폭 (mm)", min_value=0, value=int(j_W // 3), step=10
            )
            j2 = st.number_input(
                "젠다이 면2 폭 (mm)", min_value=0, value=int(j_W // 3), step=10
            )
            j3 = st.number_input(
                "젠다이 면3 폭 (mm)", min_value=0, value=int(j_W - j1 - j2), step=10
            )
            if (j1 + j2 + j3) != j_W:
                errors.append(
                    f"젠다이 면 폭 합(={j1+j2+j3})이 젠다이 벽폭(={j_W})과 다릅니다."
                )
            j_faces = [j1, j2, j3]
        else:
            j1 = st.number_input(
                "젠다이 면1 폭 (mm)", min_value=0, value=int(j_W), step=10
            )
            if j1 != j_W:
                errors.append(
                    f"젠다이 면 폭(={j1})은 젠다이 벽폭(={j_W})과 같아야 합니다."
                )
            j_faces = [j1]

    if has_jendai and int(door_wall) == int(j_wall):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
            door_draw_info = (int(door_wall), s, e, int(door_W))
        except Exception as ex:
            errors.append(str(ex))
            door_draw_info = None

        if errors:
            for msg in errors:
                st.error(msg)
        else:
            preview_img = draw_rect_preview(
                BL=int(BL),
                BW=int(BW),
                has_split=(split_kind == "구분 있음"),
                X=(int(X) if X is not None else None),
                door_info=door_draw_info,
            )
            st.image(
                preview_img,
                caption="사각형 도면(평면) 미리보기",
                width=max(160, int(preview_img.width / 2)),
                use_container_width=False,
            )

            widths = {1: int(BL), 2: int(BW), 3: int(BL), 4: int(BW)}
            st.subheader("벽면(정면도) / 라벨: WnF#")

            cols = st.columns(2)
            TH, TW = parse_tile(tile)

            # 정면도 렌더 + 페이스 수집
            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1, 2, 3, 4]):
                Wk = widths[wid]
                door_tuple = None
                if door_draw_info and int(door_wall) == wid:
                    door_tuple = (float(s), float(e))
                jf = (
                    j_faces
                    if (has_jendai and "j_wall" in locals() and int(j_wall) == wid)
                    else None
                )

                faces = build_faces_for_wall(
                    "사각형", wid, Wk, int(H_eff), door_tuple, jf, 1000
                )
                all_faces.extend(faces)
                img = draw_wall_elevation_with_faces(
                    wall_label("사각형", wid), Wk, int(H_eff), faces, target_h_px=280
                )
                with cols[i % 2]:
                    st.image(
                        img,
                        caption=f"{wall_label('사각형', wid)} (벽면 {len(faces)}개)",
                        use_container_width=False,
                    )

            # 벽면별 엔진 결과
            st.subheader("벽면별 벽판 산출")
            rows, errs = panels_for_faces(all_faces, TH, TW)
            if rows:
                df = (
                    pd.DataFrame(rows)
                    .rename(
                        columns={
                            "face_w": "벽면폭",
                            "face_h": "벽면높이",
                            "width_mm": "벽판폭",
                            "height_mm": "벽판높이",
                        }
                    )
                    .drop(columns=["분할사유"], errors="ignore")
                )

                cols_order = [
                    "벽",
                    "벽면",
                    "타일",
                    "규칙분기",
                    "panel_no",
                    "kind",
                    "pos",
                    "벽판폭",
                    "벽판높이",
                    "벽면폭",
                    "벽면높이",
                ]
                df = df[[c for c in cols_order if c in df.columns]]
                st.dataframe(df, use_container_width=True)

                # 동일 치수 벽판 수량 집계 (새 컬럼명 사용)
                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (
                    df.groupby(["kind", "벽판폭", "벽판높이"], as_index=False)
                    .size()
                    .rename(columns={"size": "qty"})
                )
                order["치수"] = (
                    order["벽판폭"].astype(int).astype(str)
                    + "×"
                    + order["벽판높이"].astype(int).astype(str)
                )
                order = order[["kind", "치수", "qty", "벽판폭", "벽판높이"]]
                st.dataframe(order, use_container_width=True)

                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                df_err = pd.DataFrame(errs).rename(
                    columns={"face_w": "벽면폭", "face_h": "벽면높이"}
                )
                st.dataframe(df_err, use_container_width=True)

else:
    # 코너형
    st.subheader("코너형 입력 (W1~W6)")
    cA, cB = st.columns(2)
    with cA:
        st.markdown("**가로(바닥) 방향**")
        W3 = st.number_input(
            "W3 (mm)", min_value=100, value=800, step=50, key="corner_w3"
        )
        W5 = st.number_input(
            "W5 (mm)", min_value=100, value=1200, step=50, key="corner_w5"
        )
        W1 = W3 + W5
        st.text_input("W1 = W3 + W5", value=str(W1), disabled=True)
    with cB:
        st.markdown("**세로(좌우) 방향**")
        W4 = st.number_input(
            "W4 (mm)", min_value=100, value=600, step=50, key="corner_w4"
        )
        W6 = st.number_input(
            "W6 (mm)", min_value=100, value=1000, step=50, key="corner_w6"
        )
        W2 = W4 + W6
        st.text_input("W2 = W4 + W6", value=str(W2), disabled=True)

    W = {1: int(W1), 2: int(W2), 3: int(W3), 4: int(W4), 5: int(W5), 6: int(W6)}
    door_W = corner_wall_width_of(int(door_wall), W)

    j_faces = []
    if has_jendai:
        j_wall = st.number_input(
            "젠다이 벽 번호", min_value=1, max_value=6, value=2, step=1
        )
        j_W = corner_wall_width_of(int(j_wall), W)
        j_step = st.radio(
            "젠다이 단차", ["없음", "있음"], horizontal=True, key="corner_j_step"
        )
        if j_step == "있음":
            cj1 = st.number_input(
                "젠다이 면1 폭 (mm)",
                min_value=0,
                value=int(j_W // 2),
                step=10,
                key="cj1",
            )
            cj2 = st.number_input(
                "젠다이 면2 폭 (mm)",
                min_value=0,
                value=int(j_W - cj1),
                step=10,
                key="cj2",
            )
            if (cj1 + cj2) != j_W:
                errors.append(
                    f"젠다이 면 폭 합(={cj1+cj2})이 젠다이 벽폭(={j_W})과 다릅니다."
                )
            j_faces = [cj1, cj2]
        else:
            cj1 = st.number_input(
                "젠다이 면1 폭 (mm)",
                min_value=0,
                value=int(j_W),
                step=10,
                key="cj_only",
            )
            if cj1 != j_W:
                errors.append(
                    f"젠다이 면 폭(={cj1})은 젠다이 벽폭(={j_W})과 같아야 합니다."
                )
            j_faces = [cj1]

    if has_jendai and "j_wall" in locals() and int(door_wall) == int(j_wall):
        errors.append("같은 벽에 문과 젠다이를 동시에 설정할 수 없습니다.")

    if calc:
        errors.extend(validate_corner_dims(W))
        try:
            s, e, L, R, n = normalize_door(int(door_W), float(door_s), float(door_d))
        except Exception as ex:
            errors.append(str(ex))

        if errors:
            for msg in errors:
                st.error(msg)
        else:
            preview_img = draw_corner_preview(
                W=W, has_split=(split_kind == "구분 있음"), canvas_w=240
            )
            st.image(
                preview_img,
                caption="코너형 도면(평면) 미리보기",
                width=max(160, int(preview_img.width / 3)),
                use_container_width=False,
            )

            widths = {i: int(W[i]) for i in range(1, 7)}
            st.subheader("벽면(정면도) / 라벨: WnF#")
            cols = st.columns(3)
            TH, TW = parse_tile(tile)

            # 정면도 렌더 + 페이스 수집
            all_faces: List[FaceSpec] = []
            for i, wid in enumerate([1, 2, 3, 4, 5, 6]):
                Wk = widths[wid]
                door_tuple = (float(s), float(e)) if int(door_wall) == wid else None
                jf = (
                    j_faces
                    if (has_jendai and "j_wall" in locals() and int(j_wall) == wid)
                    else None
                )
                faces = build_faces_for_wall(
                    "코너형", wid, Wk, int(H_eff), door_tuple, jf, 1000
                )
                all_faces.extend(faces)
                img = draw_wall_elevation_with_faces(
                    wall_label("코너형", wid), Wk, int(H_eff), faces, target_h_px=280
                )
                with cols[i % 3]:
                    st.image(
                        img,
                        caption=f"{wall_label('코너형', wid)} (벽면 {len(faces)}개)",
                        use_container_width=False,
                    )

            # 벽면별 엔진 결과
            st.subheader("벽면별 벽판 산출")
            rows, errs = panels_for_faces(all_faces, TH, TW)
            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True)

                st.markdown("**동일 치수 벽판 수량 집계**")
                order = (
                    df.groupby(["kind", "width_mm", "height_mm"], as_index=False)
                    .size()
                    .rename(columns={"size": "qty"})
                )
                order["치수"] = (
                    order["width_mm"].astype(int).astype(str)
                    + "×"
                    + order["height_mm"].astype(int).astype(str)
                )
                order = order[["kind", "치수", "qty", "width_mm", "height_mm"]]
                st.dataframe(order, use_container_width=True)

                st.markdown(f"**총 벽판 개수:** {len(df)} 장")

            if errs:
                st.warning("규칙 적용 실패/제약 위반 벽면")
                st.dataframe(pd.DataFrame(errs), use_container_width=True)

st.caption(
    "※ 본 앱은 벽(W1~)과 벽면(W1F#)을 구분해 표기하고, 벽면 단위로 패널(벽판)을 산출·집계합니다. 지원 타일: 300×600, 250×400. 최소 가공치수 80mm 제약 적용."
)
