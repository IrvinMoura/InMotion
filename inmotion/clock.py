"""Relógio monotônico: o único módulo que lê o tempo do sistema (adaptador)."""

from __future__ import annotations

import time


class MonotonicClock:
    """Satisfaz o port ``Clock`` sobre ``time.perf_counter`` (segundos, monotônico)."""

    def now(self) -> float:
        return time.perf_counter()
