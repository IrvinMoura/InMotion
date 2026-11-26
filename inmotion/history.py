"""Histórico dos sinais dos gestos numa janela de tempo (núcleo puro, spec 009).

Guarda, por gesto, o valor bruto e o filtrado de cada frame dos últimos ``window_s``
segundos, para o gráfico do overlay. Quem alimenta é a composição (``OpenCvDisplay``);
o tempo chega explícito em ``push``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from inmotion.pipeline import GestureTelemetry

_Sample = tuple[float, float, float]  # (t, bruto, filtrado); NaN = sem valor


@dataclass(frozen=True)
class SeriesView:
    """Cópia somente leitura da janela de um gesto; ``NaN`` = frame sem valor."""

    t: NDArray[np.float64]
    raw: NDArray[np.float64]
    filtered: NDArray[np.float64]
    theta_on: float
    theta_off: float
    window_s: float


@dataclass
class _Series:
    samples: deque[_Sample]
    theta_on: float
    theta_off: float


def _value(v: float | None) -> float:
    return math.nan if v is None else float(v)


def _read_only(values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.array(values, dtype=np.float64, copy=True)
    arr.setflags(write=False)
    return arr


class SignalHistory:
    """Janela deslizante por tempo: mantém as amostras com ``t ≥ t_último − window_s``.

    Se o tempo voltar (replay recomeçado), a janela recomeça: amostras com ``t`` maior que
    o do ``push`` atual são descartadas.
    """

    def __init__(self, window_s: float) -> None:
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError(f"window_s deve ser finito e > 0, recebido {window_s}")
        self._window_s = window_s
        self._series: dict[str, _Series] = {}
        self._last_t: float | None = None

    def push(self, t: float, telemetry: Sequence[GestureTelemetry]) -> None:
        """Acrescenta o frame ``t`` e descarta, em todos os gestos, o que saiu da janela."""
        if self._last_t is not None and t < self._last_t:
            for stale in self._series.values():
                stale.samples.clear()
        self._last_t = t
        for item in telemetry:
            series = self._series.get(item.name)
            if series is None:
                series = _Series(deque(), item.theta_on, item.theta_off)
                self._series[item.name] = series
            series.theta_on, series.theta_off = item.theta_on, item.theta_off
            series.samples.append((t, _value(item.raw_value), _value(item.filtered_value)))
        oldest = t - self._window_s
        for kept in self._series.values():
            while kept.samples and kept.samples[0][0] < oldest:
                kept.samples.popleft()

    def view(self, name: str) -> SeriesView | None:
        """Cópias somente leitura da janela de ``name``; ``None`` se o gesto nunca apareceu."""
        series = self._series.get(name)
        if series is None:
            return None
        table = np.array(series.samples, dtype=np.float64).reshape(-1, 3)
        return SeriesView(
            t=_read_only(table[:, 0]),
            raw=_read_only(table[:, 1]),
            filtered=_read_only(table[:, 2]),
            theta_on=series.theta_on,
            theta_off=series.theta_off,
            window_s=self._window_s,
        )

    def names(self) -> tuple[str, ...]:
        """Gestos na ordem em que apareceram pela primeira vez."""
        return tuple(self._series)

    def clear(self) -> None:
        self._series.clear()
        self._last_t = None
