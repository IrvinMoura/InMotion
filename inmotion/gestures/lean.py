"""Gestos de inclinação do tronco (RF5, A9; spec 019): frente, ré (``s``) e strafe (``a``/``d``)."""

from __future__ import annotations

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import lean_back_rel, lean_forward, lean_side
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side

DEFAULT_LEAN_PARAMS = GestureParams(0.30, 0.20, Direction.ABOVE, debounce_frames=4)  # rad
DEFAULT_WALK_BACK_PARAMS = GestureParams(0.14, 0.08, Direction.ABOVE, debounce_frames=3)
DEFAULT_STRAFE_PARAMS = DEFAULT_LEAN_PARAMS
_DEFAULT_GEOMETRY = GeometryConfig()


def lean_gesture(
    params: GestureParams = DEFAULT_LEAN_PARAMS, geometry: GeometryConfig = _DEFAULT_GEOMETRY
) -> FeatureGesture:
    """Gesto HOLD (sentido ABOVE) sobre ``lean_forward`` (A9, rad): segura enquanto inclinado.

    Ativa com a inclinação para frente ``> θ_on`` por N frames e solta abaixo de ``θ_off``
    (R5, R6). Inclinar para trás (valor negativo) nunca ativa. O sinal da feature está
    a confirmar com o MediaPipe real por ``test_lean_sign_on_hardware`` (``-m hardware``).
    """
    return FeatureGesture(
        "lean", GestureKind.HOLD, params, lambda ctx: lean_forward(ctx.pose, geometry), pure=True
    )


def walk_back_gesture(
    params: GestureParams = DEFAULT_WALK_BACK_PARAMS, geometry: GeometryConfig = _DEFAULT_GEOMETRY
) -> FeatureGesture:
    """Spec 019: gesto HOLD para recuar (``s``) ao inclinar o tronco para trás."""
    return FeatureGesture(
        "walk_back",
        GestureKind.HOLD,
        params,
        lambda ctx: lean_back_rel(ctx, geometry),
        pure=True,
    )


def strafe_gesture(
    name: str,
    side: Side,
    params: GestureParams = DEFAULT_STRAFE_PARAMS,
    geometry: GeometryConfig = _DEFAULT_GEOMETRY,
) -> FeatureGesture:
    """Spec 019: gesto HOLD para andar lateralmente (``a``/``d``) ao inclinar o tronco."""
    return FeatureGesture(
        name,
        GestureKind.HOLD,
        params,
        lambda ctx: lean_side(ctx.pose, side, geometry),
        pure=True,
    )
