"""Média móvel exponencial (R1-R3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inmotion.filters import FloatArray


@dataclass(frozen=True)
class EmaParams:
    """Parâmetros da EMA.

    ``alpha`` em (0, 1]: peso da amostra nova (1 = sem suavização).
    ``gap_max_s`` > 0: lacuna (s) acima da qual o filtro reinicia.
    """

    alpha: float = 0.5
    gap_max_s: float = 0.3

    def __post_init__(self) -> None:
        if not (math.isfinite(self.alpha) and 0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha deve estar em (0, 1], recebido {self.alpha}")
        if not (math.isfinite(self.gap_max_s) and self.gap_max_s > 0.0):
            raise ValueError(f"gap_max_s deve ser > 0, recebido {self.gap_max_s}")


class Ema:
    """Média móvel exponencial, elemento a elemento.

    - R1: ``s₀ = x₀``; ``sₙ = α·xₙ + (1−α)·sₙ₋₁``, com ``0 < α ≤ 1``.
    - R2 (forma fechada): ``sₙ = α·Σ_{k=0}^{n−1}(1−α)^k·x_{n−k} + (1−α)^n·x₀``.
    - R3: ``α·Σ_{k=0}^{n−1}(1−α)^k + (1−α)^n = 1``; logo ``sₙ`` está no fecho convexo
      das entradas (``min xᵢ ≤ sₙ ≤ max xᵢ``).

    Aceita ``float`` ou ``ndarray`` de qualquer forma e devolve sempre
    ``NDArray[np.float64]`` (0-d para escalar).
    """

    def __init__(self, params: EmaParams) -> None:
        self._params = params
        self._state: FloatArray | None = None

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        """Aplica R1 à amostra ``x`` no instante ``t`` e devolve ``sₙ``."""
        arr = np.asarray(x, dtype=np.float64)
        if self._state is None:
            self._state = arr.copy()
        else:
            if arr.shape != self._state.shape:
                raise ValueError(f"forma mudou de {self._state.shape} para {arr.shape} sem reset()")
            a = self._params.alpha
            self._state = np.asarray(a * arr + (1.0 - a) * self._state, dtype=np.float64)
        return self._state.copy()

    def reset(self) -> None:
        """Esquece o estado: a próxima amostra vira ``s₀``."""
        self._state = None
