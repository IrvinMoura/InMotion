"""Média móvel exponencial (R1-R3). Stub da T1."""

from __future__ import annotations

from dataclasses import dataclass

from inmotion.filters import FloatArray


@dataclass(frozen=True)
class EmaParams:
    alpha: float = 0.5
    gap_max_s: float = 0.3


class Ema:
    def __init__(self, params: EmaParams) -> None:
        self._params = params

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
