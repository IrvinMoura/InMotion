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
        self._state: FloatArray | None = None  # último valor filtrado (NaN = ainda não há)
        self._t_valid: FloatArray | None = None  # instante da última amostra válida por elemento
        self._t_last: float = math.nan  # instante da chamada anterior (qualquer amostra)

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        """Aplica R1 à amostra ``x`` no instante ``t`` e devolve ``sₙ``.

        Tempo: ``t`` deve ser finito e estritamente maior que o ``t`` da chamada anterior
        (``Δt = t − t_anterior > 0``, mesmo que aquela chamada só tivesse NaN); senão
        ``ValueError``. A EMA em si não usa ``Δt``, só a detecção de lacuna.

        NaN: um elemento NaN mantém o estado e devolve o último valor filtrado (NaN se
        ainda não há valor). A lacuna é medida por elemento, contra a última amostra
        VÁLIDA dele: se ``t − t_válido > gap_max_s``, o elemento reinicia e a amostra
        nova vira ``s₀``. Mudar a forma sem ``reset()`` levanta ``ValueError``.
        """
        arr = np.asarray(x, dtype=np.float64)
        if not math.isfinite(t):
            raise ValueError(f"t deve ser finito, recebido {t}")
        if self._state is None or self._t_valid is None:
            self._state = np.full(arr.shape, np.nan)
            self._t_valid = np.full(arr.shape, np.nan)
        else:
            if arr.shape != self._state.shape:
                raise ValueError(f"forma mudou de {self._state.shape} para {arr.shape} sem reset()")
            if not t > self._t_last:
                raise ValueError(f"dt deve ser > 0 (t={t}, anterior={self._t_last})")
        valid = ~np.isnan(arr)
        has_state = ~np.isnan(self._state)
        in_gap = has_state & ((t - self._t_valid) > self._params.gap_max_s)
        blend = valid & has_state & ~in_gap
        a = self._params.alpha
        smoothed = a * arr + (1.0 - a) * self._state
        self._state = np.asarray(
            np.where(blend, smoothed, np.where(valid, arr, self._state)), dtype=np.float64
        )
        self._t_valid = np.asarray(np.where(valid, t, self._t_valid), dtype=np.float64)
        self._t_last = t
        return self._state.copy()

    def reset(self) -> None:
        """Esquece o estado: a próxima amostra vira ``s₀``."""
        self._state = None
        self._t_valid = None
        self._t_last = math.nan
