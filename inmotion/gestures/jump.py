"""Gesto de pulo (RF2): trigger sobre ``j = (b − h)/s`` (J1)."""

from __future__ import annotations

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import jump_height

DEFAULT_JUMP_PARAMS = GestureParams(0.25, 0.15, Direction.ABOVE, debounce_frames=2)


def jump_gesture(params: GestureParams = DEFAULT_JUMP_PARAMS) -> FeatureGesture:
    """Gesto TRIGGER (sentido ABOVE) sobre ``jump_height``: um PULSE por salto.

    O quadril sobe acima da base R7 por mais de ``θ_on`` larguras de ombro durante N frames
    (R5). A base só se atualiza em repouso (todos os gestos de corpo em IDLE), então ela não
    acompanha o pulo (J2) mas absorve deriva lenta (J3).
    """
    return FeatureGesture("jump", GestureKind.TRIGGER, params, jump_height)
