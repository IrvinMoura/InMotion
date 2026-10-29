"""Spec 007 — máquina de estados dos gestos (R5, R6; propriedades E1 a E4)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.base import (
    Direction,
    EventKind,
    Gesture,
    GestureEvent,
    GestureKind,
    GestureParams,
    GestureState,
)

DT = 0.1  # intervalo entre frames dos testes determinísticos (s)

xfail_t1 = pytest.mark.xfail(strict=True, raises=NotImplementedError)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Feature = float | None


def feed(
    gesture: Gesture, values: Sequence[Feature], t0: float = 0.0, dt: float = DT
) -> list[GestureEvent | None]:
    """Alimenta o gesto com `values` em t0, t0+dt, ... e devolve um evento por frame."""
    return [gesture.update(v, t0 + i * dt) for i, v in enumerate(values)]


def kinds(events: Sequence[GestureEvent | None]) -> list[EventKind | None]:
    return [None if e is None else e.kind for e in events]


def is_active(gesture: Gesture) -> bool:
    return gesture.snapshot().state is GestureState.ACTIVE


# ---------------------------------------------------------------------------
# Estratégias hypothesis
# ---------------------------------------------------------------------------

finite = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
features = st.one_of(st.none(), finite)


@st.composite
def gesture_params(
    draw: st.DrawFn, *, debounce: int | None = None, cooldown: float | None = None
) -> GestureParams:
    """Parâmetros válidos para qualquer direção; `theta_on`/`theta_off` respeitam RF8."""
    theta_on = draw(st.floats(min_value=-2.0, max_value=2.0))
    width = draw(st.floats(min_value=0.05, max_value=2.0))
    direction = draw(st.sampled_from(Direction))
    n = debounce if debounce is not None else draw(st.integers(min_value=1, max_value=5))
    t_cd = cooldown if cooldown is not None else draw(st.floats(min_value=0.0, max_value=2.0))
    sign = 1.0 if direction is Direction.BELOW else -1.0
    return GestureParams(
        theta_on=theta_on,
        theta_off=theta_on + sign * width,
        direction=direction,
        debounce_frames=n,
        cooldown_s=t_cd,
    )


# ---------------------------------------------------------------------------
# T1 — propriedades centrais
# ---------------------------------------------------------------------------


@xfail_t1
@settings(max_examples=200, deadline=None)
@given(
    params=gesture_params(debounce=1, cooldown=0.0),
    base=finite,
    noise=st.lists(st.floats(min_value=-1.0, max_value=1.0), min_size=1, max_size=200),
)
def test_bounded_noise_at_most_one_transition(
    params: GestureParams, base: float, noise: list[float]
) -> None:
    """E1: com |ε| < (θ_off − θ_on)/2, há no máximo uma mudança ativo/não ativo."""
    half_width = abs(params.theta_off - params.theta_on) / 2
    gesture = Gesture("g", GestureKind.HOLD, params)

    flags = []
    for i, u in enumerate(noise):
        gesture.update(base + u * half_width * 0.999, i * DT)
        flags.append(is_active(gesture))

    changes = sum(a != b for a, b in itertools.pairwise(flags))
    assert changes <= 1


@xfail_t1
@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_activation_needs_n_consecutive_frames(n: int) -> None:
    """E2/CA2: N−1 frames ativos e depois inativo não dão START; N frames dão, no N-ésimo."""
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=n)

    short = Gesture("g", GestureKind.HOLD, params)
    out = feed(short, [0.0] * (n - 1) + [1.0, 1.0])
    assert EventKind.START not in kinds(out)

    full = Gesture("g", GestureKind.HOLD, params)
    out = feed(full, [0.0] * n + [0.0])
    assert kinds(out) == [None] * (n - 1) + [EventKind.START, None]


@xfail_t1
@settings(max_examples=200, deadline=None)
@given(
    params=gesture_params(),
    values=st.lists(features, max_size=120),
    steps=st.lists(st.floats(min_value=0.001, max_value=0.5), min_size=120, max_size=120),
)
def test_hold_events_well_formed(
    params: GestureParams, values: list[Feature], steps: list[float]
) -> None:
    """E3/CA3: START/END alternam e, após close(), os START igualam os END."""
    gesture = Gesture("g", GestureKind.HOLD, params)
    t = 0.0
    events: list[GestureEvent] = []
    for v, step in zip(values, steps, strict=False):
        t += step
        ev = gesture.update(v, t)
        if ev is not None:
            events.append(ev)
    closing = gesture.close(t)
    if closing is not None:
        events.append(closing)

    expected = [EventKind.START, EventKind.END] * (len(events) // 2 + 1)
    assert [e.kind for e in events] == expected[: len(events)]
    assert len(events) % 2 == 0


@xfail_t1
@settings(max_examples=200, deadline=None)
@given(
    values=st.lists(st.floats(min_value=-1.0, max_value=2.0), min_size=1, max_size=150),
    steps=st.lists(st.floats(min_value=0.01, max_value=0.3), min_size=150, max_size=150),
)
def test_trigger_pulses_respect_cooldown(values: list[float], steps: list[float]) -> None:
    """E4/CA4: pulsos consecutivos distam ≥ T_cd, com entrada oscilando."""
    t_cd = 0.5
    params = GestureParams(theta_on=0.0, theta_off=0.5, debounce_frames=1, cooldown_s=t_cd)
    gesture = Gesture("g", GestureKind.TRIGGER, params)

    t = 0.0
    pulses: list[float] = []
    for v, step in zip(values, steps, strict=False):
        t += step
        ev = gesture.update(v, t)
        if ev is not None:
            assert ev.kind is EventKind.PULSE
            pulses.append(ev.t)

    assert all(b - a >= t_cd for a, b in itertools.pairwise(pulses))


@xfail_t1
def test_reset_rearms_after_n_frames() -> None:
    """CA9/RF5b: ACTIVE + reset() → IDLE sem evento; pose mantida re-arma com START no N-ésimo."""
    n = 3
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=n)
    )
    feed(gesture, [0.0] * n)
    assert is_active(gesture)

    gesture.reset()

    assert gesture.snapshot().state is GestureState.IDLE
    assert gesture.snapshot().counter == 0
    out = feed(gesture, [0.0] * n, t0=10.0)
    assert kinds(out) == [None] * (n - 1) + [EventKind.START]
