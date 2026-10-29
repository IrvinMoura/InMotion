"""Árbitro de conflitos entre gestos (grupos exclusivos e combinações).

STUB da spec 007 T1: a implementação vem na T8.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from inmotion.gestures.base import GestureEvent


class GestureArbiter:
    def __init__(
        self,
        priority: Sequence[str],
        exclusive_groups: Sequence[frozenset[str]] = (),
        combos: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        raise NotImplementedError

    def resolve(self, events: Sequence[GestureEvent], t: float) -> list[GestureEvent]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    @property
    def granted(self) -> frozenset[str]:
        raise NotImplementedError
