"""Base dos gestos: máquina de estados com debounce (R5) e histerese (R6).

STUB da spec 007 T1: a implementação vem nas tarefas seguintes.
"""

from __future__ import annotations

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
    theta_on: float
    theta_off: float
    direction: Direction = Direction.BELOW
    debounce_frames: int = 3
    cooldown_s: float = 0.0


@dataclass(frozen=True)
class AnalogParams:
    dead_zone: float = 0.0
    gain: float = 1.0


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
