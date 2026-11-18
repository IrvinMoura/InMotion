"""Overlay de debug (spec 009): esqueleto conectado, normalizado (M=S*R*T), telemetria e A4.

Só lê o ``StepResult``; nunca decide nada. Mantém compatibilidade com imagens pequenas nos
testes legados (elementos avançados só desenham quando presentes no ``StepResult``).
"""

from __future__ import annotations

import math
from enum import Enum

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.exit_zone import ExitZoneState
from inmotion.gestures.base import GestureState
from inmotion.history import SeriesView
from inmotion.pipeline import GestureTelemetry, StepResult
from inmotion.safety import SafetyState
from inmotion.types import PoseFrame
from inmotion.vision.geometry import (
    NormalizedSkeleton,
    apply_h,
    inset_matrix,
    joint_angle,
    mirror_matrix,
)
from inmotion.vision.landmarks import PoseLandmark


class OverlayLevel(Enum):
    """Nível de detalhe (spec 009 RF1): ``off`` só a imagem, ``minimal`` o essencial e ``full``
    tudo."""

    OFF = "off"
    MINIMAL = "minimal"
    FULL = "full"


# Hershey só desenha ASCII: sem acentos nos textos do overlay.
_STATE_TEXT = {
    SafetyState.ACTIVE: ("ATIVO", (0, 200, 0)),
    SafetyState.PAUSED: ("PAUSADO", (0, 200, 255)),
    SafetyState.LOST: ("SEM POSE", (0, 0, 255)),
}
_GESTURE_STATE_STYLE: dict[GestureState, tuple[str, tuple[int, int, int]]] = {
    GestureState.IDLE: ("IDL", (170, 170, 170)),
    GestureState.ARMING: ("ARM", (0, 210, 255)),
    GestureState.ACTIVE: ("ACT", (60, 230, 80)),
    GestureState.COOLDOWN: ("CLD", (255, 170, 60)),
}
POSE_CONNECTIONS: tuple[tuple[PoseLandmark, PoseLandmark], ...] = (
    (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE),
    (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE),
    (PoseLandmark.LEFT_EYE, PoseLandmark.LEFT_EAR),
    (PoseLandmark.RIGHT_EYE, PoseLandmark.RIGHT_EAR),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
    (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
    (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    (PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL),
    (PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL),
    (PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_POINT_COLOR = (0, 255, 255)
_BONE_COLOR = (200, 220, 80)
_NORM_BONE_COLOR = (220, 180, 50)
_BONE_MIN_VIS = 0.5
_POINT_RADIUS = 3
_TEXT_ORIGIN_FPS = (8, 20)
_TEXT_ORIGIN_STATE = (8, 42)
_TEXT_ORIGIN_HAND = (8, 64)
_HAND_TEXT_SCALE = 0.5
_HAND_COLOR = (255, 255, 0)
_HAND_TEXT = {
    "right": "Destro - ataca: braco direito",
    "left": "Canhoto - ataca: braco esquerdo",
}
_TEXT_SCALE = 0.6
_ZONE_IDLE_COLOR = (0, 165, 255)  # laranja (BGR): visível em parede clara
_ZONE_ACTIVE_COLOR = (0, 0, 255)
_ZONE_THICKNESS = 2
_ZONE_TEXT_SCALE = 0.45
_ZONE_TEXT_PAD = (4, 16)
_ZONE_LABEL = "SAIR ({:.0f}s)"
_INSET_EXTENT = 2.0  # o miniquadro mostra [−2, 2]² em larguras de ombro
_ANGLE_COLOR = (255, 120, 255)
_ANGLE_RADIUS = 18
_ANGLE_TEXT_SCALE = 0.45
_ANGLE_TEXT_OFFSET = 22
_PLOT_W, _PLOT_H = 260, 122
_PLOT_MARGIN = 8
_PLOT_PAD = (8, 22, 8, 16)  # esquerda, topo (título), direita, base (legenda)
_PLOT_RAW_COLOR = (150, 150, 150)
_PLOT_FILTERED_COLOR = (80, 230, 120)
_PLOT_ON_COLOR = (0, 80, 255)
_PLOT_OFF_COLOR = (0, 220, 220)
_PLOT_SPAN_PAD = 0.1  # folga vertical, fração da faixa de valores
_ARMS: tuple[tuple[int, int, int], ...] = (
    (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
)


def _blend_rect(
    out: NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    alpha: float = 0.68,
) -> None:
    """Fundo semitransparente para leitura do painel de debug sobre qualquer cenário."""
    height, width = out.shape[:2]
    rx0, ry0 = max(0, x0), max(0, y0)
    rx1, ry1 = min(width, x1), min(height, y1)
    if rx1 <= rx0 or ry1 <= ry0:
        return
    roi = out[ry0:ry1, rx0:rx1].astype(np.float32)
    bg = np.array(color, dtype=np.float32)
    out[ry0:ry1, rx0:rx1] = np.clip((1.0 - alpha) * roi + alpha * bg, 0, 255).astype(np.uint8)
    cv2.rectangle(out, (rx0, ry0), (rx1 - 1, ry1 - 1), (85, 90, 100), 1)


def _draw_exit_zone(out: NDArray[np.uint8], zone: ExitZoneState, *, mirror: bool = False) -> None:
    """Bloco da zona de saída; a barra sobe de baixo para cima com o progresso."""
    height, width = out.shape[:2]
    cfg = zone.config
    x0_n, x1_n = (1.0 - cfg.x1, 1.0 - cfg.x0) if mirror else (cfg.x0, cfg.x1)
    x0, x1 = round(x0_n * width), round(x1_n * width)
    y0, y1 = round(cfg.y0 * height), round(cfg.y1 * height)
    color = _ZONE_ACTIVE_COLOR if zone.inside else _ZONE_IDLE_COLOR
    if zone.progress > 0:
        top = y1 - round(zone.progress * (y1 - y0))
        cv2.rectangle(out, (x0, top), (x1, y1), _ZONE_ACTIVE_COLOR, -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, _ZONE_THICKNESS)
    origin = (x0 + _ZONE_TEXT_PAD[0], y0 + _ZONE_TEXT_PAD[1])
    label = _ZONE_LABEL.format(cfg.hold_s)
    cv2.putText(out, label, origin, _FONT, _ZONE_TEXT_SCALE, color, 1)


def _draw_pose_skeleton(out: NDArray[np.uint8], pose: PoseFrame) -> None:
    """Desenha os ossos conectados (quando visíveis) e todos os pontos válidos da pose."""
    height, width = out.shape[:2]
    coords = pose.landmarks[:, :2] * np.array([width, height], dtype=np.float64)
    vis = pose.landmarks[:, 3]
    in_bounds = [
        bool(np.isfinite(xy).all() and 0 <= xy[0] < width and 0 <= xy[1] < height) for xy in coords
    ]
    for a, b in POSE_CONNECTIONS:
        idx_a, idx_b = int(a), int(b)
        if (
            in_bounds[idx_a]
            and in_bounds[idx_b]
            and float(vis[idx_a]) >= _BONE_MIN_VIS
            and float(vis[idx_b]) >= _BONE_MIN_VIS
        ):
            pa = (round(float(coords[idx_a, 0])), round(float(coords[idx_a, 1])))
            pb = (round(float(coords[idx_b, 0])), round(float(coords[idx_b, 1])))
            cv2.line(out, pa, pb, _BONE_COLOR, 2, cv2.LINE_AA)
    for ok, (x, y) in zip(in_bounds, coords, strict=True):
        if ok:
            cv2.circle(out, (round(float(x)), round(float(y))), _POINT_RADIUS, _POINT_COLOR, -1)


def _draw_normalized_skeleton(out: NDArray[np.uint8], skel: NormalizedSkeleton) -> None:
    """Miniquadro no canto inferior direito com o esqueleto normalizado M = S*R*T."""
    height, width = out.shape[:2]
    if width < 240 or height < 200:
        return
    box_w = min(144, width // 3)
    box_h = min(160, height // 2)
    bx0, by0 = width - box_w - 8, height - box_h - 8
    bx1, by1 = bx0 + box_w, by0 + box_h
    _blend_rect(out, bx0, by0, bx1, by1, (18, 20, 24))
    cv2.putText(out, "M = S*R*T", (bx0 + 6, by0 + 14), _FONT, 0.38, (220, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"s={skel.shoulder_px:.0f}px",
        (bx0 + 6, by1 - 6),
        _FONT,
        0.36,
        (170, 190, 210),
        1,
        cv2.LINE_AA,
    )
    inner_h = max(40, box_h - 40)
    cx = bx0 + box_w // 2
    cy = by0 + 20 + round(inner_h * 0.48)
    cv2.line(out, (bx0 + 6, cy), (bx1 - 6, cy), (55, 60, 70), 1)
    cv2.line(out, (cx, by0 + 20), (cx, by1 - 18), (55, 60, 70), 1)

    # p_inset = A·n_norm, A = T(c)·S(k) leva [−e, e]² à caixa (RF4).
    side = float(min(box_w - 18, inner_h))
    pixels = apply_h(inset_matrix((cx, cy), side, _INSET_EXTENT), skel.points[:, :2])
    pts: dict[int, tuple[int, int]] = {}
    for idx, (fx, fy) in enumerate(pixels):
        if not (np.isfinite(fx) and np.isfinite(fy)):
            continue
        px, py = round(float(fx)), round(float(fy))
        if bx0 + 2 <= px < bx1 - 2 and by0 + 18 <= py < by1 - 16:
            pts[idx] = (px, py)
    for a, b in POSE_CONNECTIONS:
        pa, pb = pts.get(int(a)), pts.get(int(b))
        if pa is not None and pb is not None:
            cv2.line(out, pa, pb, _NORM_BONE_COLOR, 1, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(out, pt, 2, _POINT_COLOR, -1)


def _fmt_val(value: float | None) -> str:
    return "  -- " if value is None else f"{value:+.2f}"


def _draw_telemetry_panel(
    out: NDArray[np.uint8],
    gestures: tuple[GestureTelemetry, ...],
    pressed: frozenset[str],
    look: tuple[int, int],
) -> None:
    """Painel inferior esquerdo: bruto vs filtrado, limiares, estado e inputs ativos."""
    height, width = out.shape[:2]
    has_io = bool(pressed) or look != (0, 0)
    if not gestures and not has_io:
        return
    if width < 240 or height < 160:
        return
    row_h = 20
    footer_h = 22
    panel_w = min(370, width - 170) if width >= 420 else width - 16
    panel_h = len(gestures) * row_h + footer_h + 10
    x0 = 8
    y1 = height - 8
    y0 = max(78, y1 - panel_h)
    _blend_rect(out, x0, y0, x0 + panel_w, y1, (18, 20, 24))

    bar_x0 = x0 + 275
    bar_x1 = x0 + panel_w - 8
    draw_bar = bar_x1 - bar_x0 >= 32
    y = y0 + 18
    for g in gestures:
        if y > y1 - footer_h:
            break
        badge, color = _GESTURE_STATE_STYLE[g.state]
        line = (
            f"{badge} {g.name[:6]:6s} r:{_fmt_val(g.raw_value)} "
            f"f:{_fmt_val(g.filtered_value)} ({g.theta_off:.2f}/{g.theta_on:.2f})"
        )
        cv2.putText(out, line, (x0 + 6, y), _FONT, 0.37, color, 1, cv2.LINE_AA)
        if draw_bar:
            top = max(1.0, max(abs(g.theta_on), abs(g.theta_off)) * 1.4)
            by0, by1 = y - 8, y - 1
            cv2.rectangle(out, (bar_x0, by0), (bar_x1, by1), (75, 80, 90), 1)
            if g.filtered_value is not None and np.isfinite(g.filtered_value):
                frac = min(max(g.filtered_value / top, 0.0), 1.0)
                fill_w = round(frac * (bar_x1 - bar_x0))
                if fill_w > 0:
                    cv2.rectangle(out, (bar_x0, by0 + 1), (bar_x0 + fill_w, by1 - 1), color, -1)
            for thresh, tcolor in ((g.theta_off, (0, 220, 220)), (g.theta_on, (0, 80, 255))):
                tx = bar_x0 + round(min(max(thresh / top, 0.0), 1.0) * (bar_x1 - bar_x0))
                cv2.line(out, (tx, by0 - 2), (tx, by1 + 2), tcolor, 1)
        y += row_h

    keys_text = "KEYS: " + (" ".join(sorted(pressed)) if pressed else "--")
    look_text = f"LOOK: ({look[0]:+d},{look[1]:+d})"
    io_color = (60, 230, 80) if pressed else (185, 195, 205)
    cv2.putText(out, f"{keys_text}  {look_text}", (x0 + 6, y1 - 6), _FONT, 0.40, io_color, 1)


def _draw_elbow_angles(out: NDArray[np.uint8], pose: PoseFrame, *, mirror: bool) -> None:
    """A7: arco e valor (graus) do ângulo em cada cotovelo visível, já na imagem exibida.

    Os pontos passam por A5 (pixels) e, com ``mirror``, por A4 com ``W − 1`` (a mesma
    conta do ``cv2.flip``), para o texto ser desenhado depois do espelho e ficar legível.
    """
    height, width = out.shape[:2]
    lm = pose.landmarks
    for idx in _ARMS:
        if not bool(np.all(lm[list(idx), 3] >= _BONE_MIN_VIS)):
            continue
        pts = lm[list(idx), :2] * np.array([width, height], dtype=np.float64)
        if mirror:
            pts = apply_h(mirror_matrix(width - 1), pts)
        if not np.isfinite(pts).all():
            continue
        shoulder, elbow, wrist = pts
        try:
            theta = joint_angle(shoulder, elbow, wrist)
        except ValueError:
            continue
        a_u = math.degrees(math.atan2(*(shoulder - elbow)[::-1]))
        a_v = math.degrees(math.atan2(*(wrist - elbow)[::-1]))
        sweep = (a_v - a_u + 540.0) % 360.0 - 180.0  # o arco curto, de u para v
        center = (round(float(elbow[0])), round(float(elbow[1])))
        start, end = sorted((a_u, a_u + sweep))
        axes = (_ANGLE_RADIUS, _ANGLE_RADIUS)
        cv2.ellipse(out, center, axes, 0.0, start, end, _ANGLE_COLOR, 2, cv2.LINE_AA)
        bisector = math.radians(a_u + sweep / 2.0 + 180.0)  # texto do lado de fora do arco
        tx = center[0] + round(_ANGLE_TEXT_OFFSET * math.cos(bisector)) - 10
        ty = center[1] + round(_ANGLE_TEXT_OFFSET * math.sin(bisector)) + 4
        text = f"{math.degrees(theta):.0f}"
        cv2.putText(out, text, (tx, ty), _FONT, _ANGLE_TEXT_SCALE, _ANGLE_COLOR, 1, cv2.LINE_AA)


def _segments(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> list[NDArray[np.int32]]:
    """Trechos contínuos (sem NaN) da linha, já em pixels inteiros para o ``polylines``."""
    finite = np.isfinite(ys)
    pieces: list[NDArray[np.int32]] = []
    start = None
    for i, ok in enumerate([*finite.tolist(), False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= 2:
                pts = np.stack([xs[start:i], ys[start:i]], axis=1)
                pieces.append(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
            start = None
    return pieces


def _draw_plot(out: NDArray[np.uint8], view: SeriesView, name: str | None) -> None:
    """RF2: bruto × filtrado nos últimos ``H = window_s`` segundos, com ``θ_on`` e ``θ_off``.

    ``x = x0 + (t − (t_n − H))/H·w`` e ``y = y0 + h·(1 − (f − f_min)/(f_max − f_min))``,
    com ``t_n`` o último instante e a faixa ``[f_min, f_max]`` cobrindo valores e limiares.
    """
    height, width = out.shape[:2]
    if width < 420 or height < 240:
        return
    x0 = width - _PLOT_W - _PLOT_MARGIN
    y0 = _PLOT_MARGIN
    _blend_rect(out, x0, y0, x0 + _PLOT_W, y0 + _PLOT_H, (18, 20, 24))
    title = f"{name or '?'}: bruto x filtrado ({view.window_s:.0f}s)"
    cv2.putText(out, title, (x0 + 6, y0 + 15), _FONT, 0.40, (220, 230, 240), 1, cv2.LINE_AA)
    left, top, right, bottom = _PLOT_PAD
    gx0, gy0 = x0 + left, y0 + top
    gw, gh = _PLOT_W - left - right, _PLOT_H - top - bottom
    legend_y = y0 + _PLOT_H - 4
    for i, (label, color) in enumerate(
        (
            ("bruto", _PLOT_RAW_COLOR),
            ("filtrado", _PLOT_FILTERED_COLOR),
            ("on", _PLOT_ON_COLOR),
            ("off", _PLOT_OFF_COLOR),
        )
    ):
        cv2.putText(out, label, (x0 + 6 + i * 62, legend_y), _FONT, 0.36, color, 1, cv2.LINE_AA)

    values = np.concatenate([view.raw, view.filtered, [view.theta_on, view.theta_off]])
    values = values[np.isfinite(values)]
    lo, hi = float(values.min()), float(values.max())
    pad = max(hi - lo, 1e-6) * _PLOT_SPAN_PAD
    lo, hi = lo - pad, hi + pad

    def to_y(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return gy0 + gh * (1.0 - (v - lo) / (hi - lo))

    for theta, color in ((view.theta_on, _PLOT_ON_COLOR), (view.theta_off, _PLOT_OFF_COLOR)):
        ty = round(float(to_y(np.array(theta))))
        cv2.line(out, (gx0, ty), (gx0 + gw, ty), color, 1)
    if view.t.size == 0:
        return
    t_end = float(view.t[-1])
    xs = gx0 + (view.t - (t_end - view.window_s)) / view.window_s * gw
    for series, color in ((view.raw, _PLOT_RAW_COLOR), (view.filtered, _PLOT_FILTERED_COLOR)):
        for piece in _segments(xs, to_y(series)):
            cv2.polylines(out, [piece], False, color, 1, cv2.LINE_AA)


def handedness_label(handedness: str) -> str:
    """Texto do modo (spec 017 RF6): "Destro"/"Canhoto" e o braço que ataca (ASCII, Hershey)."""
    return _HAND_TEXT[handedness]


def render(
    img: NDArray[np.uint8],
    result: StepResult,
    handedness: str | None = None,
    *,
    mirror: bool = False,
    level: OverlayLevel = OverlayLevel.FULL,
    plot: SeriesView | None = None,
    plot_name: str | None = None,
) -> NDArray[np.uint8]:
    """Desenha o overlay no nível ``level`` sobre uma cópia de ``img`` (O1: função pura).

    ``off``: só a imagem (espelhada, se ``mirror``). ``minimal``: esqueleto, FPS, estado,
    modo e zona de saída. ``full``: também ângulos (A7), miniquadro ``M = S·R·T``, tabela
    de gestos e, com ``plot``, o gráfico no tempo do gesto ``plot_name``.
    """
    out: NDArray[np.uint8] = np.array(img, dtype=np.uint8, copy=True)
    if level is OverlayLevel.OFF:
        return np.ascontiguousarray(cv2.flip(out, 1)) if mirror else out

    if result.pose is not None:
        _draw_pose_skeleton(out, result.pose)
    if mirror:
        out = np.ascontiguousarray(cv2.flip(out, 1))
    full = level is OverlayLevel.FULL
    if full and result.pose is not None:
        _draw_elbow_angles(out, result.pose, mirror=mirror)

    fps_text = "FPS: --" if result.fps is None else f"FPS: {result.fps:.1f}"
    if result.latency_ms is not None:
        fps_text += f"  lat: {result.latency_ms:.0f} ms"
    cv2.putText(out, fps_text, _TEXT_ORIGIN_FPS, _FONT, _TEXT_SCALE, (255, 255, 255), 1)
    if result.exit_zone is not None:
        _draw_exit_zone(out, result.exit_zone, mirror=mirror)
    label, color = _STATE_TEXT[result.safety_state]
    cv2.putText(out, label, _TEXT_ORIGIN_STATE, _FONT, _TEXT_SCALE, color, 2)
    if handedness is not None:
        text = handedness_label(handedness)
        cv2.putText(out, text, _TEXT_ORIGIN_HAND, _FONT, _HAND_TEXT_SCALE, _HAND_COLOR, 1)
    if not full:
        return out
    if result.normalized is not None:
        _draw_normalized_skeleton(out, result.normalized)
    _draw_telemetry_panel(out, result.gestures, result.pressed, result.look)
    if plot is not None:
        _draw_plot(out, plot, plot_name)
    return out
