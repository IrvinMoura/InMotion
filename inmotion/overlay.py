"""Overlay mínimo de debug: FPS, estado de segurança e pontos da pose (adaptador).

Só lê o ``StepResult``; nunca decide nada. Expandido na spec 009.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

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
_TEXT_SCALE = 0.6


def render(img: NDArray[np.uint8], result: StepResult) -> NDArray[np.uint8]:
    """Devolve uma cópia de ``img`` com FPS, estado de segurança e pontos da pose.

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
    label, color = _STATE_TEXT[result.safety_state]
    cv2.putText(out, label, _TEXT_ORIGIN_STATE, _FONT, _TEXT_SCALE, color, 2)
    return out
