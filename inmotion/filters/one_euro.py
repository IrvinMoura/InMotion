"""One Euro Filter (R4): EMA com corte adaptativo à velocidade."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inmotion.filters import FloatArray


@dataclass(frozen=True)
class OneEuroParams:
    """Parâmetros do One Euro.

    ``min_cutoff`` (Hz, > 0): corte em repouso (f_min). ``beta`` (≥ 0): ganho da velocidade
    no corte. ``d_cutoff`` (Hz, > 0): corte do filtro da derivada. ``gap_max_s`` (> 0):
    lacuna (s) acima da qual o filtro reinicia.
    """

    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    gap_max_s: float = 0.3

    def __post_init__(self) -> None:
        if not (math.isfinite(self.min_cutoff) and self.min_cutoff > 0.0):
            raise ValueError(f"min_cutoff deve ser > 0, recebido {self.min_cutoff}")
        if not (math.isfinite(self.beta) and self.beta >= 0.0):
            raise ValueError(f"beta deve ser >= 0, recebido {self.beta}")
        if not (math.isfinite(self.d_cutoff) and self.d_cutoff > 0.0):
            raise ValueError(f"d_cutoff deve ser > 0, recebido {self.d_cutoff}")
        if not (math.isfinite(self.gap_max_s) and self.gap_max_s > 0.0):
            raise ValueError(f"gap_max_s deve ser > 0, recebido {self.gap_max_s}")


def _alpha(cutoff_hz: FloatArray, dt: float) -> FloatArray:
    """R4 vetorizado: ``τ = 1/(2π·f_c)``; ``α = 1/(1 + τ/Δt)``."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return np.asarray(1.0 / (1.0 + tau / dt), dtype=np.float64)


def smoothing_factor(cutoff_hz: float | FloatArray, dt: float) -> float | FloatArray:
    """R4: ``τ = 1/(2π·f_c)``; ``α = 1/(1 + τ/Δt)``.

    F1: ``α ∈ (0, 1)`` para ``Δt > 0`` e ``f_c > 0``. Escalar entra, ``float`` sai;
    array entra, array sai. ``Δt ≤ 0`` ou ``f_c ≤ 0`` levanta ``ValueError``.
    """
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt deve ser > 0, recebido {dt}")
    cutoff = np.asarray(cutoff_hz, dtype=np.float64)
    if not np.all(cutoff > 0.0):
        raise ValueError("cutoff deve ser > 0")
    alpha = _alpha(cutoff, dt)
    if alpha.ndim == 0:
        return float(alpha)
    return alpha


class OneEuro:
    """One Euro Filter (R4), elemento a elemento.

    - ``ẋ = (x − x̂ₙ₋₁)/Δt``, com ``Δt = tₙ − tₙ₋₁``;
    - ``ẋ̂ = EMA_{α(d_cutoff)}(ẋ)`` (derivada filtrada);
    - ``f_c = f_min + β·|ẋ̂|`` (corte adaptativo);
    - ``x̂ = EMA_{α(f_c)}(x)``, com ``α = 1/(1 + τ/Δt)``, ``τ = 1/(2π·f_c)``.

    A primeira amostra vira ``x̂₀ = x₀`` (derivada inicial 0). Como ``α ∈ (0, 1)`` (F1), a
    saída é combinação convexa das entradas. Com ``β = 0`` e ``Δt`` constante é uma EMA de
    ``α = 1/(1 + 1/(2π·f_min·Δt))`` (F2).
    """

    def __init__(self, params: OneEuroParams) -> None:
        self._params = params
        self._x_hat: FloatArray | None = None
        self._dx_hat: FloatArray | None = None
        self._t_prev: float = math.nan

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        """Consome ``x`` em ``t`` e devolve ``x̂``."""
        arr = np.asarray(x, dtype=np.float64)
        p = self._params
        if self._x_hat is None or self._dx_hat is None:
            self._x_hat = arr.copy()
            self._dx_hat = np.zeros_like(arr)
        else:
            if arr.shape != self._x_hat.shape:
                raise ValueError(f"forma mudou de {self._x_hat.shape} para {arr.shape} sem reset()")
            dt = t - self._t_prev
            if not dt > 0.0:
                raise ValueError(f"dt deve ser > 0, recebido {dt}")
            dx = (arr - self._x_hat) / dt
            a_d = _alpha(np.asarray(p.d_cutoff), dt)
            self._dx_hat = np.asarray(a_d * dx + (1.0 - a_d) * self._dx_hat, dtype=np.float64)
            cutoff = p.min_cutoff + p.beta * np.abs(self._dx_hat)
            a = _alpha(cutoff, dt)
            self._x_hat = np.asarray(a * arr + (1.0 - a) * self._x_hat, dtype=np.float64)
        self._t_prev = t
        return self._x_hat.copy()

    def reset(self) -> None:
        """Esquece o estado: a próxima amostra vira ``x̂₀``."""
        self._x_hat = None
        self._dx_hat = None
        self._t_prev = math.nan
