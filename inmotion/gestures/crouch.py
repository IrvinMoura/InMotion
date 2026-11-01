"""Gesto de agachar (RF4 da 008; RF3 da 017): hold sobre a queda dos ombros ou do quadril."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, get_args

from inmotion.gestures.base import Direction, GestureKind, GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.features import FrameContext, crouch_depth, crouch_drop

CrouchFeature = Literal["hip", "shoulders"]
CROUCH_FEATURES: tuple[str, ...] = get_args(CrouchFeature)

# Limiar PROVISÓRIO do ``crouch_drop`` (em larguras de ombro): agachadinha do pulo máx 0,66,
# inclinar máx 0,22, agachar mediana >= 1,18 (sessões guiadas de 01/11/2026). Recalibrar na
# spec 017 T7. Para ``feature="hip"`` os valores antigos eram theta_on 0,35 e theta_off 0,25.
DEFAULT_CROUCH_PARAMS = GestureParams(0.9, 0.6, Direction.ABOVE, debounce_frames=4)

_FEATURES: dict[str, Callable[[FrameContext], float | None]] = {
    "hip": crouch_depth,
    "shoulders": crouch_drop,
}


def crouch_gesture(
    params: GestureParams = DEFAULT_CROUCH_PARAMS,
    feature: CrouchFeature = "shoulders",
) -> FeatureGesture:
    """Gesto HOLD (sentido ABOVE) que segura enquanto o corpo está agachado.

    - ``feature="shoulders"`` (padrão, spec 017 RF3): ``c = (ȳ_ombros − b_o)/s``, a queda dos
      ombros em larguras de ombro; continua visível quando o quadril sai do quadro.
    - ``feature="hip"``: ``(h − b)/s`` da spec 008, pelo quadril.

    Ativa com ``c > θ_on`` por N frames e solta com ``c < θ_off`` (R5, R6). Enquanto o gesto
    não está em IDLE as bases R7 ficam congeladas, então um agachamento longo não "ensina" a
    base a achar que a pessoa sempre esteve abaixada.
    """
    if feature not in _FEATURES:
        raise ValueError(f"feature deve ser um de {CROUCH_FEATURES}, recebido {feature!r}")
    return FeatureGesture("crouch", GestureKind.HOLD, params, _FEATURES[feature])
