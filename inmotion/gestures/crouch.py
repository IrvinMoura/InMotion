"""Gesto de agachar (RF4): hold sobre ``(h − b)/s``."""

from __future__ import annotations

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import crouch_depth

DEFAULT_CROUCH_PARAMS = GestureParams(0.35, 0.25, Direction.ABOVE, debounce_frames=4)


def crouch_gesture(params: GestureParams = DEFAULT_CROUCH_PARAMS) -> FeatureGesture:
    """Gesto HOLD (sentido ABOVE) sobre ``crouch_depth``: segura enquanto o quadril está baixo.

    Ativa com ``(h − b)/s > θ_on`` por N frames e solta com ``(h − b)/s < θ_off`` (R5, R6).
    Enquanto o gesto não está em IDLE a base R7 fica congelada, então um agachamento longo
    não "ensina" a base a achar que a pessoa sempre esteve abaixada.
    """
    return FeatureGesture("crouch", GestureKind.HOLD, params, crouch_depth)
