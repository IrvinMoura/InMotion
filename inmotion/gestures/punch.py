"""Gesto de soco (RF1): histerese sobre o alcance frontal ``r`` (A7/A8)."""

from __future__ import annotations

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import forward_reach
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side

DEFAULT_PUNCH_PARAMS = GestureParams(0.70, 0.55, Direction.ABOVE, debounce_frames=3)


def punch_gesture(
    name: str, side: Side, params: GestureParams, geometry: GeometryConfig
) -> FeatureGesture:
    """Gesto HOLD do braço ``side``: ativa com ``r > θ_on`` por N frames, solta com ``r < θ_off``.

    ``r`` é o alcance frontal normalizado (``forward_reach``, ``|r| ≤ 1``); o parâmetro
    ``params`` deve ser ``ABOVE`` (``θ_on > θ_off``).
    """
    return FeatureGesture(
        name,
        GestureKind.HOLD,
        params,
        lambda ctx: forward_reach(ctx.pose, side, geometry),
    )


def opposite(side: Side) -> Side:
    """O outro lado do corpo."""
    return Side.LEFT if side is Side.RIGHT else Side.RIGHT


def punch_pair(
    handedness: Side,
    attack_params: GestureParams,
    use_params: GestureParams,
    geometry: GeometryConfig,
) -> tuple[FeatureGesture, FeatureGesture]:
    """CA2: ``(attack, use)``; o braço dominante (``handedness``) ataca e o outro usa item."""
    attack = punch_gesture("attack", handedness, attack_params, geometry)
    use = punch_gesture("use", opposite(handedness), use_params, geometry)
    return attack, use
