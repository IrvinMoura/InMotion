"""Marcha no lugar (spec 018): passos pelos joelhos, andar (``w``) e correr (``ctrl``).

Núcleo puro: o tempo entra como parâmetro ``t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from inmotion.gestures.base import Direction, GestureParams
from inmotion.gestures.body import FeatureGesture

# Passo (spec 018 T0): k parado <= 0,08, no pulo <= 0,14, marchando p95 >= 0,38.
DEFAULT_STEP_PARAMS = GestureParams(0.30, 0.15, Direction.ABOVE, debounce_frames=1)
DEFAULT_WALK_PARAMS = GestureParams(0.6, 0.4, Direction.ABOVE, debounce_frames=1)
DEFAULT_SPRINT_PARAMS = GestureParams(1.8, 1.5, Direction.ABOVE, debounce_frames=1)


@dataclass(frozen=True)
class MarchParams:
    """Parâmetros da marcha: passos para começar, prazo entre passos, α da EMA (R1), passo."""

    min_steps: int = 2
    step_timeout_s: float = 1.2
    cadence_alpha: float = 0.5
    step: GestureParams = field(default_factory=lambda: DEFAULT_STEP_PARAMS)

    def __post_init__(self) -> None:
        if self.min_steps < 1:
            raise ValueError("min_steps deve ser >= 1")
        if not (math.isfinite(self.step_timeout_s) and self.step_timeout_s > 0.0):
            raise ValueError("step_timeout_s deve ser finito e > 0")
        if not (math.isfinite(self.cadence_alpha) and 0.0 < self.cadence_alpha <= 1.0):
            raise ValueError("cadence_alpha deve estar em (0, 1]")
        if self.step.direction is not Direction.ABOVE:
            raise ValueError("step deve ter direction ABOVE (o joelho sobe com k > theta_on)")


@dataclass(frozen=True)
class MarchState:
    """Estado da marcha num frame: andando, cadência (passos/s) e passos da sequência."""

    walking: bool
    cadence: float | None
    steps: int


class MarchTracker:
    """Passos alternados, prazo e cadência (R1) a partir de ``k`` de cada joelho."""

    def __init__(self, params: MarchParams) -> None:
        raise NotImplementedError

    def update(self, k_left: float | None, k_right: float | None, t: float) -> MarchState:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


def walk_gesture(params: GestureParams = DEFAULT_WALK_PARAMS) -> FeatureGesture:
    """Gesto HOLD ``walk`` sobre ``ctx.march``."""
    raise NotImplementedError


def sprint_gesture(params: GestureParams = DEFAULT_SPRINT_PARAMS) -> FeatureGesture:
    """Gesto HOLD ``sprint`` sobre a cadência."""
    raise NotImplementedError
