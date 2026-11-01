"""Marcha no lugar (spec 018): passos pelos joelhos, andar (``w``) e correr (``ctrl``).

Núcleo puro: o tempo entra como parâmetro ``t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from inmotion.gestures.base import (
    Direction,
    EventKind,
    Gesture,
    GestureKind,
    GestureParams,
)
from inmotion.gestures.body import FeatureGesture
from inmotion.vision.landmarks import Side

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
    """Passos alternados, prazo e cadência (R1) a partir da elevação ``k`` de cada joelho.

    Cada joelho tem um ``Gesture`` TRIGGER (R5/R6) sobre ``k``: cada PULSE é um passo do
    lado. Regras:

    - **Outro joelho abaixado:** um passo só conta se ``k_outro < θ_on`` (no agachar os dois
      joelhos sobem juntos e isso não é passo).
    - **Começar (RF3):** ``min_steps`` passos de lados alternados, cada um a no máximo
      ``step_timeout_s`` do anterior. Dois passos seguidos do mesmo lado reiniciam a
      sequência (M2: ficar num pé só nunca anda).
    - **Manter:** depois de começar, qualquer passo renova o prazo; passou
      ``t − t_último > step_timeout_s`` sem passo, para (M3) e zera cadência e contagem.
    - **Cadência (R1):** com ``Δ_n`` o intervalo entre passos consecutivos,
      ``Δ̂_n = (1 − α)·Δ̂_{n−1} + α·Δ_n`` (``Δ̂_1 = Δ_1``) e ``f = 1/Δ̂`` passos/s. Vale R2
      (``Δ̂`` fica entre o menor e o maior intervalo) e, com ``Δ`` constante, ``f → 1/Δ``
      (M4), com constante de tempo ``1/α`` passos (R3).

    O tempo entra como parâmetro ``t`` (núcleo puro).
    """

    def __init__(self, params: MarchParams) -> None:
        self._params = params
        self._gestures = {
            Side.LEFT: Gesture("step_left", GestureKind.TRIGGER, params.step),
            Side.RIGHT: Gesture("step_right", GestureKind.TRIGGER, params.step),
        }
        self._clear()

    def _clear(self) -> None:
        self._last_t: float | None = None
        self._last_side: Side | None = None
        self._steps = 0
        self._walking = False
        self._interval: float | None = None  # Δ̂, EMA do intervalo entre passos (s)

    def update(self, k_left: float | None, k_right: float | None, t: float) -> MarchState:
        """Avança os detectores de passo com ``k`` de cada lado em ``t`` e devolve o estado."""
        if self._last_t is not None and t - self._last_t > self._params.step_timeout_s:
            self._clear()  # M3: sem passo por step_timeout_s, para
        on = self._params.step.theta_on
        # O outro joelho tem de estar abaixado (None = não visível = não erguido).
        gated_left = k_left if k_right is None or k_right < on else None
        gated_right = k_right if k_left is None or k_left < on else None
        for side, k in ((Side.LEFT, gated_left), (Side.RIGHT, gated_right)):
            event = self._gestures[side].update(k, t)
            if event is not None and event.kind is EventKind.PULSE:
                self._on_step(side, t)
        return self.state

    @property
    def state(self) -> MarchState:
        """Estado atual: andando, cadência ``f = 1/Δ̂`` (``None`` sem intervalos) e passos."""
        interval = self._interval
        cadence = 1.0 / interval if interval is not None and interval > 0.0 else None
        return MarchState(self._walking, cadence, self._steps)

    def _on_step(self, side: Side, t: float) -> None:
        if self._last_t is None:
            self._steps = 1
        elif self._walking or side is not self._last_side:
            self._steps += 1
            self._add_interval(t - self._last_t)
        else:  # mesmo lado antes de começar: reinicia a sequência neste passo
            self._steps = 1
            self._interval = None
        self._last_t, self._last_side = t, side
        if self._steps >= self._params.min_steps:
            self._walking = True

    def _add_interval(self, delta: float) -> None:
        """R1: ``Δ̂ ← (1 − α)·Δ̂ + α·Δ``; o primeiro intervalo inicializa ``Δ̂``."""
        if not (math.isfinite(delta) and delta > 0.0):
            return
        if self._interval is None:
            self._interval = delta
        else:
            alpha = self._params.cadence_alpha
            self._interval = (1.0 - alpha) * self._interval + alpha * delta

    def reset(self) -> None:
        """Esquece passos, andar e cadência SEM evento (o ``SafetyManager`` já soltou)."""
        for gesture in self._gestures.values():
            gesture.reset()
        self._clear()


def walk_gesture(params: GestureParams = DEFAULT_WALK_PARAMS) -> FeatureGesture:
    """Gesto HOLD ``walk`` sobre ``ctx.march``."""
    raise NotImplementedError


def sprint_gesture(params: GestureParams = DEFAULT_SPRINT_PARAMS) -> FeatureGesture:
    """Gesto HOLD ``sprint`` sobre a cadência."""
    raise NotImplementedError
