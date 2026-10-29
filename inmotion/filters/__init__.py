"""Filtros temporais (spec 006): EMA (R1-R3) e One Euro (R4).

Este módulo define só o tipo ``FloatArray`` e o Protocol ``TemporalFilter``; não importa
``ema`` nem ``one_euro`` (evita dependência circular).
"""

from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class TemporalFilter(Protocol):
    """Filtro causal elemento a elemento, com tempo explícito."""

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        """Consome a amostra ``x`` no instante ``t`` (s) e devolve a estimativa filtrada."""
        ...

    def reset(self) -> None:
        """Esquece o estado: a próxima amostra vira o valor inicial."""
        ...
