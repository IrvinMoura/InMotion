"""Spec 009 — histórico dos sinais dos gestos para o gráfico do overlay (CA4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inmotion.gestures.base import GestureKind, GestureState
from inmotion.history import SignalHistory
from inmotion.pipeline import GestureTelemetry

NOT_YET = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="spec 009 T1")


def tele(
    name: str = "attack",
    raw: float | None = 0.5,
    filtered: float | None = 0.4,
    theta_on: float = 0.8,
    theta_off: float = 0.6,
) -> GestureTelemetry:
    return GestureTelemetry(
        name=name,
        kind=GestureKind.TRIGGER,
        state=GestureState.IDLE,
        raw_value=raw,
        filtered_value=filtered,
        theta_on=theta_on,
        theta_off=theta_off,
    )


@NOT_YET
def test_window_drops_old_samples() -> None:
    hist = SignalHistory(window_s=3.0)
    for i in range(9):  # t = 0, 0.5, ..., 4.0
        hist.push(i * 0.5, [tele(raw=float(i))])
    view = hist.view("attack")
    assert view is not None
    assert view.t[0] == pytest.approx(1.0)  # 4.0 − 3.0: a borda fica
    assert view.t[-1] == pytest.approx(4.0)
    assert list(view.raw) == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


@NOT_YET
def test_records_raw_filtered_and_thresholds() -> None:
    hist = SignalHistory(window_s=3.0)
    hist.push(0.0, [tele(raw=0.5, filtered=0.4, theta_on=0.8, theta_off=0.6)])
    hist.push(0.1, [tele(raw=0.7, filtered=0.5, theta_on=0.9, theta_off=0.7)])
    view = hist.view("attack")
    assert view is not None
    assert list(view.raw) == [0.5, 0.7]
    assert list(view.filtered) == [0.4, 0.5]
    assert (view.theta_on, view.theta_off) == (0.9, 0.7)  # os limiares mais recentes


@NOT_YET
def test_missing_values_become_nan() -> None:
    hist = SignalHistory(window_s=3.0)
    hist.push(0.0, [tele(raw=None, filtered=None)])
    view = hist.view("attack")
    assert view is not None
    assert math.isnan(view.raw[0])
    assert math.isnan(view.filtered[0])


@NOT_YET
def test_view_is_a_read_only_copy() -> None:
    hist = SignalHistory(window_s=3.0)
    hist.push(0.0, [tele()])
    view = hist.view("attack")
    assert view is not None
    with pytest.raises(ValueError, match="read-only"):
        view.raw[0] = 9.0
    hist.push(0.1, [tele()])
    assert len(view.t) == 1  # a vista antiga não muda com o push


@NOT_YET
def test_names_follow_first_seen_order_and_unknown_is_none() -> None:
    hist = SignalHistory(window_s=3.0)
    hist.push(0.0, [tele("attack"), tele("jump")])
    hist.push(0.1, [tele("crouch"), tele("attack")])
    assert hist.names() == ("attack", "jump", "crouch")
    assert hist.view("nope") is None


@NOT_YET
def test_gesture_not_seen_keeps_its_old_samples_until_they_expire() -> None:
    hist = SignalHistory(window_s=1.0)
    hist.push(0.0, [tele("jump")])
    hist.push(0.5, [tele("attack")])
    hist.push(2.0, [tele("attack")])
    jump = hist.view("jump")
    assert jump is not None
    assert len(jump.t) == 0  # expirou pelo tempo do último push


@NOT_YET
def test_clear_empties_everything() -> None:
    hist = SignalHistory(window_s=3.0)
    hist.push(0.0, [tele()])
    hist.clear()
    assert hist.names() == ()
    assert hist.view("attack") is None


@NOT_YET
@pytest.mark.parametrize("window", [0.0, -1.0, math.inf, math.nan])
def test_rejects_invalid_window(window: float) -> None:
    with pytest.raises(ValueError, match="window_s"):
        SignalHistory(window_s=window)


@NOT_YET
def test_time_going_backwards_restarts_the_window() -> None:
    """Replay recomeçado ou relógio novo: amostras do "futuro" não ficam no gráfico."""
    hist = SignalHistory(window_s=3.0)
    hist.push(10.0, [tele(raw=1.0)])
    hist.push(1.0, [tele(raw=2.0)])
    view = hist.view("attack")
    assert view is not None
    assert np.array_equal(view.raw, [2.0])
