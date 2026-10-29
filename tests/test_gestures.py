"""Spec 007 — máquina de estados dos gestos (R5, R6; propriedades E1 a E4)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.base import (
    AnalogParams,
    Direction,
    EventKind,
    Gesture,
    GestureEvent,
    GestureKind,
    GestureParams,
    GestureSnapshot,
    GestureState,
)

DT = 0.1  # intervalo entre frames dos testes determinísticos (s)

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


# ---------------------------------------------------------------------------
# T3 — máquina de estados HOLD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, float("nan"), float("inf")])
def test_none_while_active_ends_hold(missing: float | None) -> None:
    """RF4: feature inválida (None; também NaN/inf) em ACTIVE encerra o hold na hora."""
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=2)
    )
    feed(gesture, [0.0, 0.0])
    assert is_active(gesture)

    event = gesture.update(missing, 1.0)

    assert event is not None
    assert (event.gesture, event.kind, event.t) == ("g", EventKind.END, 1.0)
    assert not is_active(gesture)


def test_none_while_arming_returns_to_idle() -> None:
    """RF2/R5: None em ARMING zera o contador; é preciso N frames seguidos de novo."""
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=3)
    )

    out = feed(gesture, [0.0, 0.0, None, 0.0, 0.0, 0.0])

    assert kinds(out) == [None] * 5 + [EventKind.START]


def test_hysteresis_band_keeps_hold_active() -> None:
    """R6: dentro de [θ_on, θ_off] o estado se mantém; só f > θ_off encerra."""
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    )

    out = feed(gesture, [0.0, 0.5, 0.4, 0.6, 0.61])

    assert kinds(out) == [EventKind.START, None, None, None, EventKind.END]


def test_band_does_not_activate_from_idle() -> None:
    """R6: um f na banda (θ_on ≤ f ≤ θ_off) não ativa quem estava inativo."""
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    )

    assert kinds(feed(gesture, [0.5, 0.4, 0.6])) == [None, None, None]


# ---------------------------------------------------------------------------
# T4 — direção ABOVE por espelhamento do sinal
# ---------------------------------------------------------------------------


def test_above_activates_when_feature_exceeds_theta_on() -> None:
    gesture = Gesture(
        "g",
        GestureKind.HOLD,
        GestureParams(theta_on=0.6, theta_off=0.4, direction=Direction.ABOVE, debounce_frames=2),
    )

    out = feed(gesture, [0.5, 0.7, 0.7, 0.5, 0.39])

    assert kinds(out) == [None, None, EventKind.START, None, EventKind.END]


@settings(max_examples=200, deadline=None)
@given(
    params=gesture_params(),
    values=st.lists(features, max_size=120),
    steps=st.lists(st.floats(min_value=0.001, max_value=0.5), min_size=120, max_size=120),
    kind=st.sampled_from([GestureKind.HOLD, GestureKind.TRIGGER]),
)
def test_above_mirrors_below(
    params: GestureParams,
    values: list[Feature],
    steps: list[float],
    kind: Literal[GestureKind.HOLD, GestureKind.TRIGGER],
) -> None:
    """ABOVE(θ_on, θ_off) com f ≡ BELOW(−θ_on, −θ_off) com −f: mesmos eventos e mesmos t."""
    mirrored = GestureParams(
        theta_on=-params.theta_on,
        theta_off=-params.theta_off,
        direction=Direction.ABOVE if params.direction is Direction.BELOW else Direction.BELOW,
        debounce_frames=params.debounce_frames,
        cooldown_s=params.cooldown_s,
    )
    original = Gesture("g", kind, params)
    twin = Gesture("g", kind, mirrored)

    t = 0.0
    for v, step in zip(values, steps, strict=False):
        t += step
        a = original.update(v, t)
        b = twin.update(None if v is None else -v, t)
        assert a == b
        assert original.snapshot().state is twin.snapshot().state


# ---------------------------------------------------------------------------
# T5 — TRIGGER e COOLDOWN
# ---------------------------------------------------------------------------


def test_trigger_needs_return_to_idle() -> None:
    """RF3: pose mantida (ou na banda) dá um único PULSE; só dispara de novo após soltar."""
    gesture = Gesture(
        "g", GestureKind.TRIGGER, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    )

    held = feed(gesture, [0.0, 0.0, 0.5, 0.0, 0.0])  # nunca passou de θ_off
    assert kinds(held) == [EventKind.PULSE, None, None, None, None]

    released = feed(gesture, [0.7, 0.0], t0=1.0)  # soltou (> θ_off) e voltou
    assert kinds(released) == [None, EventKind.PULSE]


def test_trigger_never_emits_end() -> None:
    gesture = Gesture(
        "g", GestureKind.TRIGGER, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    )

    out = feed(gesture, [0.0, 1.0, 0.0, 1.0, None])

    assert set(kinds(out)) == {None, EventKind.PULSE}
    assert gesture.close(9.0) is None


def test_cooldown_blocks_reactivation_until_elapsed() -> None:
    """RF2: em COOLDOWN, ativar só após t − t_end ≥ T_cd, reavaliando f no mesmo frame."""
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1, cooldown_s=0.5)
    gesture = Gesture("g", GestureKind.TRIGGER, params)

    assert gesture.update(0.0, 0.0) is not None  # PULSE em t = 0
    assert gesture.update(1.0, 0.1) is None  # saiu: COOLDOWN, t_end = 0.1
    assert gesture.update(0.0, 0.3) is None  # 0,2 s < T_cd: ignorado
    assert gesture.snapshot().state is GestureState.COOLDOWN
    assert gesture.update(0.0, 0.5) is None  # 0,4 s < T_cd
    event = gesture.update(0.0, 0.6)  # 0,5 s ≥ T_cd: IDLE e reavalia f no mesmo frame
    assert event is not None
    assert (event.kind, event.t) == (EventKind.PULSE, 0.6)


def test_hold_with_cooldown_delays_restart() -> None:
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1, cooldown_s=1.0)
    gesture = Gesture("g", GestureKind.HOLD, params)

    assert gesture.update(0.0, 0.0) is not None  # START
    end = gesture.update(1.0, 1.0)  # END, t_end = 1,0
    assert end is not None
    assert end.kind is EventKind.END
    assert gesture.update(0.0, 1.5) is None
    event = gesture.update(0.0, 2.0)
    assert event is not None
    assert event.kind is EventKind.START


# ---------------------------------------------------------------------------
# T6 — reset, close e snapshot
# ---------------------------------------------------------------------------


def test_reset_from_cooldown_and_arming_clears_everything_without_events() -> None:
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=2, cooldown_s=10.0)
    gesture = Gesture("g", GestureKind.HOLD, params)
    feed(gesture, [0.0, 0.0, 1.0])  # START e END: em COOLDOWN
    assert gesture.snapshot().state is GestureState.COOLDOWN
    gesture.reset()
    assert gesture.snapshot().state is GestureState.IDLE
    assert gesture.snapshot().last_value is None

    gesture.update(0.0, 5.0)  # ARMING
    assert gesture.snapshot().state is GestureState.ARMING
    gesture.reset()
    snap = gesture.snapshot()
    assert (snap.state, snap.counter, snap.confidence) == (GestureState.IDLE, 0, 0.0)
    # o cooldown também foi zerado: ativa em 2 frames mesmo logo após o END
    assert kinds(feed(gesture, [0.0, 0.0], t0=5.1)) == [None, EventKind.START]


def test_close_ends_active_hold() -> None:
    """RF5: close() emite END se o HOLD está ACTIVE; depois não há mais nada a soltar."""
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    )
    feed(gesture, [0.0])

    event = gesture.close(3.0)

    assert event is not None
    assert (event.gesture, event.kind, event.t) == ("g", EventKind.END, 3.0)
    assert gesture.close(3.1) is None


@pytest.mark.parametrize("prefix", [[], [0.0]])  # IDLE; ARMING (N = 2)
def test_close_is_silent_when_not_active(prefix: list[float]) -> None:
    gesture = Gesture(
        "g", GestureKind.HOLD, GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=2)
    )
    feed(gesture, prefix)

    assert gesture.close(1.0) is None


@pytest.mark.parametrize(
    ("direction", "theta_on", "theta_off", "values", "expected"),
    [
        (Direction.BELOW, 0.4, 0.6, [0.3, 0.0, 0.4, 0.5, 0.9], [0.5, 1.0, 0.0, 0.0, 0.0]),
        (Direction.ABOVE, 0.6, 0.4, [0.7, 1.0, 0.6, 0.5, 0.1], [0.5, 1.0, 0.0, 0.0, 0.0]),
    ],
)
def test_confidence_is_normalized_margin_on_active_side(
    direction: Direction,
    theta_on: float,
    theta_off: float,
    values: list[float],
    expected: list[float],
) -> None:
    """Premissa 2: c = clip(|f − θ_on| / |θ_off − θ_on|, 0, 1) no lado ativo; 0 fora dele."""
    params = GestureParams(theta_on=theta_on, theta_off=theta_off, direction=direction)
    gesture = Gesture("g", GestureKind.HOLD, params)

    got = []
    for i, v in enumerate(values):
        gesture.update(v, i * DT)
        got.append(gesture.snapshot().confidence)

    assert got == pytest.approx(expected)


def test_snapshot_fields_and_confidence_range() -> None:
    """CA8/RF6: o snapshot traz estado, c, último f, θ_on, θ_off e confiança em [0, 1]."""
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=3)
    gesture = Gesture("punch", GestureKind.HOLD, params)
    gesture.update(0.2, 0.0)

    snap = gesture.snapshot()

    assert snap == GestureSnapshot(
        name="punch",
        kind=GestureKind.HOLD,
        state=GestureState.ARMING,
        counter=1,
        last_value=0.2,
        theta_on=0.4,
        theta_off=0.6,
        confidence=snap.confidence,
    )
    assert snap.confidence == pytest.approx(1.0)


@settings(max_examples=200, deadline=None)
@given(params=gesture_params(), values=st.lists(features, max_size=60))
def test_confidence_always_within_unit_interval(
    params: GestureParams, values: list[Feature]
) -> None:
    gesture = Gesture("g", GestureKind.HOLD, params)
    for i, v in enumerate(values):
        gesture.update(v, i * DT)
        assert 0.0 <= gesture.snapshot().confidence <= 1.0


def test_start_and_pulse_events_carry_confidence() -> None:
    params = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
    start = Gesture("h", GestureKind.HOLD, params).update(0.3, 0.0)
    pulse = Gesture("p", GestureKind.TRIGGER, params).update(0.0, 0.0)

    assert start is not None
    assert start.confidence == pytest.approx(0.5)
    assert pulse is not None
    assert pulse.confidence == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# T2 — parâmetros validados
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "theta_on", "theta_off"),
    [
        (Direction.BELOW, 0.5, 0.5),  # iguais
        (Direction.BELOW, 0.6, 0.4),  # invertidos
        (Direction.ABOVE, 0.5, 0.5),
        (Direction.ABOVE, 0.4, 0.6),
        (Direction.BELOW, float("nan"), 0.6),
        (Direction.BELOW, 0.4, float("inf")),
        (Direction.ABOVE, float("-inf"), -1.0),
    ],
)
def test_params_reject_wrong_threshold_order(
    direction: Direction, theta_on: float, theta_off: float
) -> None:
    """CA7/RF8: BELOW exige θ_on < θ_off; ABOVE exige θ_on > θ_off; θ finitos."""
    with pytest.raises(ValueError, match="theta"):
        GestureParams(theta_on=theta_on, theta_off=theta_off, direction=direction)


def test_params_accept_valid_order_in_both_directions() -> None:
    assert GestureParams(theta_on=0.4, theta_off=0.6).direction is Direction.BELOW
    above = GestureParams(theta_on=0.6, theta_off=0.4, direction=Direction.ABOVE)
    assert above.theta_on > above.theta_off


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"debounce_frames": 0}, "debounce_frames"),
        ({"debounce_frames": -2}, "debounce_frames"),
        ({"cooldown_s": -0.1}, "cooldown_s"),
        ({"cooldown_s": float("nan")}, "cooldown_s"),
        ({"cooldown_s": float("inf")}, "cooldown_s"),
    ],
)
def test_params_reject_invalid_debounce_and_cooldown(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GestureParams(theta_on=0.4, theta_off=0.6, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dead_zone": -0.01}, "dead_zone"),
        ({"dead_zone": float("nan")}, "dead_zone"),
        ({"gain": float("inf")}, "gain"),
        ({"gain": float("nan")}, "gain"),
    ],
)
def test_analog_params_reject_invalid_values(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AnalogParams(**kwargs)
