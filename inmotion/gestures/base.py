"""Base dos gestos: máquina de estados com debounce (R5) e histerese (R6).

STUB da spec 007 T1: a implementação vem nas tarefas seguintes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal


class GestureKind(Enum):
    HOLD = auto()
    TRIGGER = auto()
    ANALOG = auto()


class Direction(Enum):
    BELOW = auto()
    ABOVE = auto()


class GestureState(Enum):
    IDLE = auto()
    ARMING = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


class EventKind(Enum):
    START = auto()
    END = auto()
    PULSE = auto()
    VALUE = auto()


@dataclass(frozen=True)
class GestureParams:
    """Parâmetros de um gesto de limiar (`hold` ou `trigger`).

    Sentido `BELOW` (ativa com f < θ_on): exige θ_on < θ_off (RF8, CA7).
    Sentido `ABOVE` (ativa com f > θ_on): exige θ_on > θ_off. A faixa [θ_on, θ_off] é a
    banda de histerese R6; o debounce R5 exige `debounce_frames` = N ≥ 1 frames seguidos.
    """

    theta_on: float
    theta_off: float
    direction: Direction = Direction.BELOW
    debounce_frames: int = 3
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.theta_on) and math.isfinite(self.theta_off)):
            raise ValueError("theta_on e theta_off devem ser finitos")
        if self.direction is Direction.BELOW and not self.theta_on < self.theta_off:
            raise ValueError("BELOW exige theta_on < theta_off")
        if self.direction is Direction.ABOVE and not self.theta_on > self.theta_off:
            raise ValueError("ABOVE exige theta_on > theta_off")
        if self.debounce_frames < 1:
            raise ValueError("debounce_frames deve ser >= 1")
        if not (math.isfinite(self.cooldown_s) and self.cooldown_s >= 0.0):
            raise ValueError("cooldown_s deve ser finito e >= 0")


@dataclass(frozen=True)
class AnalogParams:
    """Parâmetros do gesto analógico: zona morta (≥ 0) e ganho."""

    dead_zone: float = 0.0
    gain: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.dead_zone) and self.dead_zone >= 0.0):
            raise ValueError("dead_zone deve ser finito e >= 0")
        if not math.isfinite(self.gain):
            raise ValueError("gain deve ser finito")


@dataclass(frozen=True)
class GestureEvent:
    gesture: str
    kind: EventKind
    t: float
    value: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class GestureSnapshot:
    name: str
    kind: GestureKind
    state: GestureState
    counter: int
    last_value: float | None
    theta_on: float
    theta_off: float
    confidence: float


class Gesture:
    """Máquina de estados com R5 (debounce) e R6 (histerese)."""

    def __init__(
        self,
        name: str,
        kind: Literal[GestureKind.HOLD, GestureKind.TRIGGER],
        params: GestureParams,
    ) -> None:
        raise NotImplementedError

    def update(self, f: float | None, t: float) -> GestureEvent | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self, t: float) -> GestureEvent | None:
        raise NotImplementedError

    def snapshot(self) -> GestureSnapshot:
        raise NotImplementedError


class AnalogGesture:
    def __init__(self, name: str, params: AnalogParams) -> None:
        raise NotImplementedError

    def update(self, x: float | None, t: float) -> GestureEvent:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def snapshot(self) -> GestureSnapshot:
        raise NotImplementedError
