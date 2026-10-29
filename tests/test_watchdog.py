"""Spec 002 T6 — Watchdog: expira uma única vez por travamento."""

import math

import pytest

from inmotion.watchdog import Watchdog


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_rejects_invalid_timeout(bad: float) -> None:
    with pytest.raises(ValueError, match="finito"):
        Watchdog(timeout_s=bad)


def test_not_expired_before_first_beat() -> None:
    assert not Watchdog(0.5).expired(100.0)


def test_not_expired_within_timeout() -> None:
    wd = Watchdog(0.5)
    wd.beat(1.0)
    assert not wd.expired(1.49)


def test_expires_at_timeout() -> None:
    wd = Watchdog(0.5)
    wd.beat(1.0)
    assert wd.expired(1.5)


def test_expires_once_per_stall() -> None:
    wd = Watchdog(0.5)
    wd.beat(1.0)
    assert wd.expired(2.0)
    assert not wd.expired(2.1)
    assert not wd.expired(50.0)


def test_rearms_after_beat() -> None:
    wd = Watchdog(0.5)
    wd.beat(1.0)
    assert wd.expired(2.0)
    wd.beat(3.0)
    assert not wd.expired(3.2)
    assert wd.expired(3.6)
    assert not wd.expired(4.0)


def test_beat_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finito"):
        Watchdog(0.5).beat(math.nan)
