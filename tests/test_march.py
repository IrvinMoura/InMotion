"""Spec 018 — marcha no lugar: passos alternados, prazo, cadência (R1) e gestos walk/sprint."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.base import EventKind, GestureEvent
from inmotion.gestures.features import FrameContext
from inmotion.gestures.march import (
    MarchParams,
    MarchState,
    MarchTracker,
    sprint_gesture,
    walk_gesture,
)
from inmotion.vision.landmarks import Side
from tests.helpers import make_body

DT = 1.0 / 30.0
K_UP = 0.8  # joelho erguido (acima de theta_on = 0,30)
PARAMS = MarchParams()
L, R = Side.LEFT, Side.RIGHT

NOT_YET = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="spec 018 T1")


def run_steps(
    tracker: MarchTracker,
    steps: Sequence[tuple[float, Side]],
    t_end: float,
    t_start: float = 0.0,
) -> list[tuple[float, MarchState]]:
    """Alimenta o tracker a 30 fps; cada passo é o joelho do lado erguido por 2 frames."""
    out: list[tuple[float, MarchState]] = []
    n = round((t_end - t_start) / DT)
    for i in range(n + 1):
        t = t_start + i * DT
        k = {L: 0.0, R: 0.0}
        for ts, side in steps:
            if ts - 1e-9 <= t < ts + 2 * DT - 1e-9:
                k[side] = K_UP
        out.append((t, tracker.update(k[L], k[R], t)))
    return out


def alternating(t0: float, interval: float, n: int, first: Side = L) -> list[tuple[float, Side]]:
    other = R if first is L else L
    return [(t0 + i * interval, first if i % 2 == 0 else other) for i in range(n)]


def first_walking(trace: Sequence[tuple[float, MarchState]]) -> float | None:
    return next((t for t, s in trace if s.walking), None)


@NOT_YET
def test_walk_starts_after_two_alternating_steps_and_ends_after_timeout() -> None:
    """CA1: a 2 passos/s, `walking` liga no 2º passo alternado e desliga `step_timeout_s` depois
    do último."""
    steps = alternating(1.0, 0.5, 5)  # E D E D E; último em t = 3,0
    trace = run_steps(MarchTracker(PARAMS), steps, 6.0)
    start = first_walking(trace)
    assert start is not None
    assert 1.5 <= start <= 1.5 + 2 * DT  # logo no 2º passo
    assert not any(s.walking for t, s in trace if t < 1.5)
    last = 3.0
    assert all(s.walking for t, s in trace if start <= t <= last + PARAMS.step_timeout_s - DT)
    end = next(t for t, s in trace if t > start and not s.walking)
    assert last + PARAMS.step_timeout_s <= end <= last + PARAMS.step_timeout_s + 2 * DT


@NOT_YET
def test_single_raised_knee_never_walks() -> None:
    """CA2: um joelho erguido e parado (um pé só) gera no máximo um passo: não anda."""
    tracker = MarchTracker(PARAMS)
    for i in range(300):
        state = tracker.update(K_UP, 0.0, i * DT)
        assert not state.walking


@NOT_YET
@settings(max_examples=60, deadline=None)
@given(
    side=st.sampled_from([L, R]),
    gaps=st.lists(st.floats(0.2, 1.1), min_size=1, max_size=15),
)
def test_same_side_steps_never_start_walk(side: Side, gaps: list[float]) -> None:
    """M2: uma sequência só de passos do mesmo lado nunca inicia `walk`."""
    t, steps = 1.0, []
    for g in gaps:
        steps.append((t, side))
        t += g
    trace = run_steps(MarchTracker(PARAMS), steps, t + 2.0)
    assert not any(s.walking for _, s in trace)


@NOT_YET
def test_walk_stops_and_state_clears_without_steps() -> None:
    """M3: sem passos por `step_timeout_s`, `walking` cai, a cadência zera e os passos também."""
    trace = run_steps(MarchTracker(PARAMS), alternating(1.0, 0.5, 4), 8.0)
    _, state = trace[-1]
    assert state == MarchState(walking=False, cadence=None, steps=0)
    assert any(s.walking for _, s in trace)


@NOT_YET
@settings(max_examples=40, deadline=None)
@given(interval=st.floats(0.25, 1.0), n=st.integers(6, 14))
def test_cadence_converges_to_inverse_interval(interval: float, n: int) -> None:
    """M4 (R1): passos com intervalo constante Δ levam a cadência a 1/Δ."""
    steps = alternating(1.0, interval, n)
    trace = run_steps(MarchTracker(PARAMS), steps, steps[-1][0] + DT)
    cadence = trace[-1][1].cadence
    assert cadence is not None
    assert cadence == pytest.approx(1.0 / interval, rel=0.05)


def _ctx(t: float, state: MarchState) -> FrameContext:
    return FrameContext(t, make_body(t=t), None, None, None, None, march=state)


def _kinds(events: list[GestureEvent | None]) -> list[EventKind]:
    return [e.kind for e in events if e is not None]


@NOT_YET
def test_walk_gesture_follows_walking() -> None:
    gesture = walk_gesture()
    states = [MarchState(False, None, 0), MarchState(True, None, 2), MarchState(False, None, 0)]
    events = [gesture.update(_ctx(i * DT, s)) for i, s in enumerate(states)]
    assert _kinds(events) == [EventKind.START, EventKind.END]


@NOT_YET
def test_sprint_hysteresis_and_ends_with_walk() -> None:
    """CA4: cadência > f_on liga `sprint`; entre f_off e f_on mantém; < f_off ou fim do
    `walk` desliga."""
    gesture = sprint_gesture()
    walking = [MarchState(True, f, 4) for f in (1.2, 2.2, 1.6, 1.4, 2.0)]
    events = [gesture.update(_ctx(i * DT, s)) for i, s in enumerate(walking)]
    assert _kinds(events) == [EventKind.START, EventKind.END, EventKind.START]
    assert gesture.update(_ctx(1.0, MarchState(False, 2.0, 0))) is not None  # fim do walk -> END
