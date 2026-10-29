"""Spec 007 — GestureArbiter (RF7, CA6)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.arbiter import GestureArbiter
from inmotion.gestures.base import EventKind, Gesture, GestureEvent, GestureKind, GestureParams

xfail_t1 = pytest.mark.xfail(strict=True, raises=NotImplementedError)

PARAMS = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
NAMES = ("a", "b", "both", "c", "d")

_row = st.lists(
    st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    min_size=len(NAMES),
    max_size=len(NAMES),
)


@xfail_t1
@settings(max_examples=150, deadline=None)
@given(columns=st.lists(_row, max_size=80))
def test_arbiter_output_well_formed(columns: list[list[float | None]]) -> None:
    """CA6: a saída do arbiter preserva E3 por gesto e respeita combo e exclusividade."""
    gestures = {n: Gesture(n, GestureKind.HOLD, PARAMS) for n in NAMES}
    arbiter = GestureArbiter(
        priority=NAMES,
        exclusive_groups=[frozenset({"c", "d"})],
        combos={"both": frozenset({"a", "b"})},
    )

    out: list[GestureEvent] = []
    t = 0.0
    for row in columns:
        t += 0.1
        events = [e for g, v in zip(gestures.values(), row, strict=True) if (e := g.update(v, t))]
        out += arbiter.resolve(events, t)
        assert not {"c", "d"} <= arbiter.granted
        assert not ("both" in arbiter.granted and arbiter.granted & {"a", "b"})
    closing = [e for g in gestures.values() if (e := g.close(t))]
    out += arbiter.resolve(closing, t)

    for name in NAMES:
        seq = [e.kind for e in out if e.gesture == name]
        expected = [EventKind.START, EventKind.END] * (len(seq) // 2 + 1)
        assert seq == expected[: len(seq)]
        assert len(seq) % 2 == 0
    assert arbiter.granted == frozenset()
