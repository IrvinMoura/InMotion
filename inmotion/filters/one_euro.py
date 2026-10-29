"""One Euro Filter (R4). Stub da T1."""

from __future__ import annotations

from dataclasses import dataclass

from inmotion.filters import FloatArray


@dataclass(frozen=True)
class OneEuroParams:
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    gap_max_s: float = 0.3


def smoothing_factor(cutoff_hz: float | FloatArray, dt: float) -> float | FloatArray:
    raise NotImplementedError


class OneEuro:
    def __init__(self, params: OneEuroParams) -> None:
        self._params = params

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
