"""Testes da ResponseCurve (spec 014, T2; CA1: propriedades C1-C4)."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.gestures.head_look import ResponseCurve

_pos = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False)
_curves = st.builds(
    ResponseCurve,
    gain=_pos,
    dead_zone=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    gamma=st.floats(min_value=1.0, max_value=3.0, allow_nan=False),
    v_max=_pos,
)
_xs = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)


@given(_curves, _xs, _xs)
def test_response_curve_properties(curve: ResponseCurve, x: float, y: float) -> None:
    v = curve(x)
    assert curve(-x) == -v  # C1
    if abs(x) <= curve.dead_zone:
        assert v == 0.0  # C2
    assert abs(v) <= curve.v_max  # C4
    lo, hi = sorted((abs(x), abs(y)))
    assert abs(curve(lo)) <= abs(curve(hi))  # C3


@given(_curves, st.floats(min_value=1e-12, max_value=1e-6))
def test_continuity_at_dead_zone(curve: ResponseCurve, eps: float) -> None:
    d = curve.dead_zone
    assert curve(d) == 0.0
    assert abs(curve(d + eps)) <= curve.gain * eps**curve.gamma + 1e-9


def test_known_values() -> None:
    c = ResponseCurve(gain=1500.0, dead_zone=0.08, gamma=1.0, v_max=1200.0)
    assert c(0.18) == pytest.approx(150.0)
    assert c(-0.18) == pytest.approx(-150.0)
    assert c(0.05) == 0.0
    assert c(5.0) == 1200.0
    assert ResponseCurve()(0.08 + 0.1) == pytest.approx(1500.0 * 0.1**1.5)


def test_non_finite_inputs() -> None:
    c = ResponseCurve()
    assert c(math.nan) == 0.0
    assert c(math.inf) == c.v_max
    assert c(-math.inf) == -c.v_max


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gain": 0.0},
        {"gain": -1.0},
        {"dead_zone": -0.01},
        {"gamma": 0.99},
        {"v_max": 0.0},
        {"gain": math.nan},
        {"dead_zone": math.inf},
        {"gamma": math.inf},
        {"v_max": math.inf},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"deve|devem"):
        ResponseCurve(**kwargs)
