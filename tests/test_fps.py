"""Spec 001 — medidor de FPS por janela deslizante (CA9)."""

import math

import pytest

from kinect.fps import FpsMeter


def test_first_frame_returns_none() -> None:
    assert FpsMeter().update(0.0) is None


def test_constant_rate_gives_exact_fps() -> None:
    meter = FpsMeter(window=30)
    meter.update(0.0)

    readings = [meter.update(k / 30) for k in range(1, 100)]

    assert readings == [pytest.approx(30.0, rel=1e-9)] * 99


def test_window_slides() -> None:
    meter = FpsMeter(window=2)

    readings = [meter.update(t) for t in (0.0, 1.0, 2.0, 2.5)]

    # janela de 2 intervalos: o último usa só os instantes 1,0 → 2,5
    assert readings == [None, pytest.approx(1.0), pytest.approx(1.0), pytest.approx(2 / 1.5)]


@pytest.mark.parametrize("window", [0, -1])
def test_invalid_window_raises(window: int) -> None:
    with pytest.raises(ValueError, match="window"):
        FpsMeter(window=window)


@pytest.mark.parametrize("t", [1.0, 0.5])
def test_non_increasing_time_raises(t: float) -> None:
    meter = FpsMeter()
    meter.update(1.0)

    with pytest.raises(ValueError, match="crescente"):
        meter.update(t)


@pytest.mark.parametrize("bad_t", [math.nan, math.inf, -math.inf])
def test_non_finite_time_raises(bad_t: float) -> None:
    meter = FpsMeter()
    with pytest.raises(ValueError, match="finito"):
        meter.update(bad_t)

    meter.update(0.0)
    with pytest.raises(ValueError, match="finito"):
        meter.update(bad_t)
