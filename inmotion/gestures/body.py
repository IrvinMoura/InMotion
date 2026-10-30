"""Gestos de corpo: uma feature escalar sobre o ``FrameContext`` alimenta um ``Gesture``."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from inmotion.gestures.base import (
    Gesture,
    GestureEvent,
    GestureKind,
    GestureParams,
    GestureSnapshot,
    GestureState,
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
        self.name = name
        self.kind = kind
        self._gesture = Gesture(name, kind, params)
        self._feature = feature
        self._value: float | None = None

    def update(self, ctx: FrameContext) -> GestureEvent | None:
        """Calcula a feature do contexto e avança a máquina no instante ``ctx.t``."""
        self._value = self._feature(ctx)
        return self._gesture.update(self._value, ctx.t)

    def reset(self) -> None:
        """RF5b da 007: volta a IDLE sem evento (a segurança já soltou tudo)."""
        self._gesture.reset()
        self._value = None

    def close(self, t: float) -> GestureEvent | None:
        """Encerra o HOLD ativo com END."""
        return self._gesture.close(t)

    def snapshot(self) -> GestureSnapshot:
        return self._gesture.snapshot()

    @property
    def feature_value(self) -> float | None:
        """Última feature calculada (para o overlay e o histórico)."""
        return self._value


def all_idle(gestures: Sequence[BodyGesture]) -> bool:
    """Repouso do corpo (RF3): todos os gestos em IDLE (a base R7 só anda nesse caso)."""
    return all(g.snapshot().state is GestureState.IDLE for g in gestures)
