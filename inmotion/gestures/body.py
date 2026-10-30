"""Gestos de corpo: uma feature escalar sobre o ``FrameContext`` alimenta um ``Gesture``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from inmotion.gestures.base import (
    GestureEvent,
    GestureKind,
    GestureParams,
    GestureSnapshot,
)
from inmotion.gestures.features import FrameContext


class BodyGesture(Protocol):
    """Contrato dos gestos de corpo usados pelo pipeline."""

    name: str
    kind: GestureKind

    def update(self, ctx: FrameContext) -> GestureEvent | None: ...

    def reset(self) -> None: ...

    def close(self, t: float) -> GestureEvent | None: ...

    def snapshot(self) -> GestureSnapshot: ...

    @property
    def feature_value(self) -> float | None: ...


class FeatureGesture:
    """Satisfaz ``BodyGesture``: ``feature(ctx)`` alimenta a máquina ``Gesture`` (R5, R6)."""

    def __init__(
        self,
        name: str,
        kind: Literal[GestureKind.HOLD, GestureKind.TRIGGER],
        params: GestureParams,
        feature: Callable[[FrameContext], float | None],
    ) -> None:
        raise NotImplementedError

    def update(self, ctx: FrameContext) -> GestureEvent | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self, t: float) -> GestureEvent | None:
        raise NotImplementedError

    def snapshot(self) -> GestureSnapshot:
        raise NotImplementedError

    @property
    def feature_value(self) -> float | None:
        raise NotImplementedError


def all_idle(gestures: Sequence[BodyGesture]) -> bool:
    """Repouso do corpo (RF3): todos os gestos em IDLE."""
    raise NotImplementedError
