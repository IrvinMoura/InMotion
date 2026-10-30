"""Gesto de inclinar o tronco para frente (RF5, A9): hold para andar."""

from __future__ import annotations

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import lean_forward
from inmotion.vision.geometry import GeometryConfig

DEFAULT_LEAN_PARAMS = GestureParams(0.30, 0.20, Direction.ABOVE, debounce_frames=4)  # rad
_DEFAULT_GEOMETRY = GeometryConfig()


def lean_gesture(
    params: GestureParams = DEFAULT_LEAN_PARAMS, geometry: GeometryConfig = _DEFAULT_GEOMETRY
) -> FeatureGesture:
    """Gesto HOLD (sentido ABOVE) sobre ``lean_forward`` (A9, rad): segura para andar.

    Ativa com a inclinação para frente ``> θ_on`` por N frames e solta abaixo de ``θ_off``
    (R5, R6). Inclinar para trás (valor negativo) nunca ativa. O sinal da feature está
    a confirmar com o MediaPipe real por ``test_lean_sign_on_hardware`` (``-m hardware``).
    """
    return FeatureGesture(
        "walk", GestureKind.HOLD, params, lambda ctx: lean_forward(ctx.pose, geometry)
    )
