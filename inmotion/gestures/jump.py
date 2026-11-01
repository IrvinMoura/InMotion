"""Gesto de pulo (RF2): trigger sobre ``j = (b − h)/s`` (J1)."""

from __future__ import annotations

from collections.abc import Callable

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import FrameContext, jump_height, knee_lift
from inmotion.vision.landmarks import Side

DEFAULT_JUMP_PARAMS = GestureParams(0.25, 0.15, Direction.ABOVE, debounce_frames=2)


def _gated_jump_height(knee_gate: float) -> Callable[[FrameContext], float | None]:
    """RF6b (018): ``j`` vale ``0`` enquanto algum joelho tem ``k > knee_gate`` ou teve há
    menos de ``jump_gate_hold_s`` (``ctx.march.knee_lifted_recently``)."""

    def feature(ctx: FrameContext) -> float | None:
        if ctx.march is not None and ctx.march.knee_lifted_recently:
            return 0.0
        for side in (Side.LEFT, Side.RIGHT):
            k = knee_lift(ctx, side)
            if k is not None and k > knee_gate:
                return 0.0
        return jump_height(ctx)

    return feature


def jump_gesture(
    params: GestureParams = DEFAULT_JUMP_PARAMS, knee_gate: float | None = None
) -> FeatureGesture:
    """Gesto TRIGGER (sentido ABOVE) sobre ``jump_height``: um PULSE por salto.

    O quadril sobe acima da base R7 por mais de ``θ_on`` larguras de ombro durante N frames
    (R5). A base só se atualiza em repouso (todos os gestos de corpo em IDLE), então ela não
    acompanha o pulo (J2) mas absorve deriva lenta (J3).

    Spec 018 RF6b: com ``knee_gate`` (o ``θ_on`` do passo), a feature vale ``0`` enquanto algum
    joelho estiver erguido (``k > knee_gate``). O quique do quadril na marcha rápida chega a
    ``j = 0,33`` sempre com um joelho em ``k >= 0,41``; os pulos reais têm ``k <= 0,14``
    (T0, 01/11/2026). O limiar do pulo não muda.
    """
    feature = jump_height if knee_gate is None else _gated_jump_height(knee_gate)
    return FeatureGesture("jump", GestureKind.TRIGGER, params, feature)
