"""Histórico dos sinais dos gestos numa janela de tempo (núcleo puro, spec 009).

Guarda, por gesto, o valor bruto e o filtrado de cada frame dos últimos ``window_s``
segundos, para o gráfico do overlay. Quem alimenta é a composição (``OpenCvDisplay``);
o tempo chega explícito em ``push``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from inmotion.pipeline import GestureTelemetry


@dataclass(frozen=True)
class SeriesView:
    """Cópia somente leitura da janela de um gesto; ``NaN`` = frame sem valor."""

    t: NDArray[np.float64]
    raw: NDArray[np.float64]
    filtered: NDArray[np.float64]
    theta_on: float
    theta_off: float


class SignalHistory:
    """Janela deslizante por tempo: mantém as amostras com ``t ≥ t_último − window_s``."""

    def __init__(self, window_s: float) -> None:
        raise NotImplementedError

    def push(self, t: float, telemetry: Sequence[GestureTelemetry]) -> None:
        raise NotImplementedError

    def view(self, name: str) -> SeriesView | None:
        raise NotImplementedError

    def names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
