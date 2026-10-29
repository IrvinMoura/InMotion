"""Testes do relógio monotônico e dos ports (spec 002 T3)."""

from __future__ import annotations

import itertools

import pytest

from inmotion.clock import MonotonicClock
from inmotion.ports import Clock


def test_monotonic_clock_increases() -> None:
    clock = MonotonicClock()
    readings = [clock.now() for _ in range(100)]
    assert all(b >= a for a, b in itertools.pairwise(readings))


def test_monotonic_clock_advances_over_real_time() -> None:
    clock = MonotonicClock()
    t0 = clock.now()
    while clock.now() - t0 < 0.002:
        pass
    assert clock.now() > t0


def test_monotonic_clock_reads_perf_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("inmotion.clock.time.perf_counter", lambda: 42.5)
    assert MonotonicClock().now() == 42.5


def test_monotonic_clock_satisfies_clock_protocol() -> None:
    clock: Clock = MonotonicClock()  # checado pelo mypy
    assert isinstance(clock.now(), float)
