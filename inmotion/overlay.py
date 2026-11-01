"""Overlay mínimo de debug: FPS, estado de segurança e pontos da pose (adaptador).

Só lê o ``StepResult``; nunca decide nada. Expandido na spec 009.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.exit_zone import ExitZoneState
from inmotion.pipeline import StepResult
from inmotion.safety import SafetyState

# Hershey só desenha ASCII: sem acentos nos textos do overlay.
_STATE_TEXT = {
    SafetyState.ACTIVE: ("ATIVO", (0, 200, 0)),
    SafetyState.PAUSED: ("PAUSADO", (0, 200, 255)),
    SafetyState.LOST: ("SEM POSE", (0, 0, 255)),
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_POINT_COLOR = (0, 255, 255)
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


def _draw_exit_zone(out: NDArray[np.uint8], zone: ExitZoneState) -> None:
    """Bloco da zona de saída; a barra sobe de baixo para cima com o progresso."""
    height, width = out.shape[:2]
    cfg = zone.config
    x0, x1 = round(cfg.x0 * width), round(cfg.x1 * width)
    y0, y1 = round(cfg.y0 * height), round(cfg.y1 * height)
    color = _ZONE_ACTIVE_COLOR if zone.inside else _ZONE_IDLE_COLOR
    if zone.progress > 0:
        top = y1 - round(zone.progress * (y1 - y0))
        cv2.rectangle(out, (x0, top), (x1, y1), _ZONE_ACTIVE_COLOR, -1)
    cv2.rectangle(out, (x0, y0), (x1, y1), color, _ZONE_THICKNESS)
    origin = (x0 + _ZONE_TEXT_PAD[0], y0 + _ZONE_TEXT_PAD[1])
    label = _ZONE_LABEL.format(cfg.hold_s)
    cv2.putText(out, label, origin, _FONT, _ZONE_TEXT_SCALE, color, 1)


def handedness_label(handedness: str) -> str:
    """Texto do modo (spec 017 RF6): "Destro"/"Canhoto" e o braço que ataca (ASCII, Hershey)."""
    return _HAND_TEXT[handedness]


def render(
    img: NDArray[np.uint8], result: StepResult, handedness: str | None = None
) -> NDArray[np.uint8]:
    """Devolve uma cópia de ``img`` com FPS, estado de segurança e pontos da pose.

    ``handedness`` (``"right"``/``"left"``, da config; ``None`` omite) acrescenta o modo
    destro/canhoto e o braço que ataca. Vem por parâmetro e não pelo ``StepResult`` porque é
    configuração fixa da sessão, não resultado do frame; o overlay continua só lendo.

    Os pontos usam A5 inversa: ``p = diag(W, H)·n``, com (W, H) o tamanho de ``img``.
    Nem ``img`` nem ``result`` são alterados.
    """
    out: NDArray[np.uint8] = np.array(img, dtype=np.uint8, copy=True)
    height, width = out.shape[:2]

    if result.pose is not None:
        scale = np.array([width, height], dtype=np.float64)
        for x, y in result.pose.landmarks[:, :2] * scale:
            cv2.circle(out, (round(float(x)), round(float(y))), _POINT_RADIUS, _POINT_COLOR, -1)

    fps_text = "FPS: --" if result.fps is None else f"FPS: {result.fps:.1f}"
    cv2.putText(out, fps_text, _TEXT_ORIGIN_FPS, _FONT, _TEXT_SCALE, (255, 255, 255), 1)
    if result.exit_zone is not None:
        _draw_exit_zone(out, result.exit_zone)
    label, color = _STATE_TEXT[result.safety_state]
    cv2.putText(out, label, _TEXT_ORIGIN_STATE, _FONT, _TEXT_SCALE, color, 2)
    if handedness is not None:
        text = handedness_label(handedness)
        cv2.putText(out, text, _TEXT_ORIGIN_HAND, _FONT, _HAND_TEXT_SCALE, _HAND_COLOR, 1)
    return out
