"""Spec 007 — GestureArbiter (RF7, CA6)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.arbiter import ArbiterConfig, GestureArbiter
from inmotion.gestures.base import EventKind, Gesture, GestureEvent, GestureKind, GestureParams
from inmotion.gestures.factory import build_arbiter


def ev(name: str, kind: EventKind, t: float = 0.0) -> GestureEvent:
    return GestureEvent(name, kind, t)


def summary(events: list[GestureEvent]) -> list[tuple[str, EventKind]]:
    return [(e.gesture, e.kind) for e in events]


START, END, PULSE, VALUE = EventKind.START, EventKind.END, EventKind.PULSE, EventKind.VALUE

PARAMS = GestureParams(theta_on=0.4, theta_off=0.6, debounce_frames=1)
NAMES = ("a", "b", "both", "c", "d")

_row = st.lists(
    st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
    min_size=len(NAMES),
    max_size=len(NAMES),
)


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


def make_arbiter() -> GestureArbiter:
    return GestureArbiter(
        priority=["a", "b", "both", "c", "d"],
        exclusive_groups=[frozenset({"c", "d"})],
        combos={"both": frozenset({"a", "b"})},
    )


def test_independent_gestures_pass_through() -> None:
    arbiter = make_arbiter()

    out = arbiter.resolve([ev("a", START), ev("b", START)], 0.0)

    assert summary(out) == [("a", START), ("b", START)]
    assert arbiter.granted == {"a", "b"}


def test_combo_suppresses_members() -> None:
    """CA6: o combinado ativo encerra os individuais com END e suprime os seus START."""
    arbiter = make_arbiter()
    arbiter.resolve([ev("a", START)], 0.0)

    out = arbiter.resolve([ev("both", START), ev("b", START)], 0.1)

    assert summary(out) == [("a", END), ("both", START)]  # b nunca chegou a ser liberado
    assert arbiter.granted == {"both"}

    # com o combinado ativo, mais eventos dos membros são engolidos
    assert arbiter.resolve([ev("a", END), ev("a", START)], 0.2) == []

    # o combinado termina: os membros ainda "querem" estar ativos e voltam com START
    out = arbiter.resolve([ev("both", END)], 0.3)
    assert summary(out) == [("both", END), ("a", START), ("b", START)]
    assert arbiter.granted == {"a", "b"}


def test_combo_suppresses_member_pulses_but_not_values() -> None:
    arbiter = make_arbiter()
    arbiter.resolve([ev("both", START)], 0.0)

    out = arbiter.resolve([ev("a", PULSE), ev("lean", VALUE)], 0.1)

    assert summary(out) == [("lean", VALUE)]


def test_exclusive_group_priority() -> None:
    """RF7: no grupo exclusivo só o de maior prioridade (ordem do perfil) fica liberado."""
    arbiter = make_arbiter()

    out = arbiter.resolve([ev("d", START)], 0.0)
    assert summary(out) == [("d", START)]

    out = arbiter.resolve([ev("c", START)], 0.1)  # c tem prioridade sobre d
    assert summary(out) == [("d", END), ("c", START)]
    assert arbiter.granted == {"c"}

    out = arbiter.resolve([ev("c", END)], 0.2)  # d ainda quer: volta
    assert summary(out) == [("c", END), ("d", START)]

    out = arbiter.resolve([ev("d", END)], 0.3)
    assert summary(out) == [("d", END)]
    assert arbiter.granted == frozenset()


def test_exclusive_group_drops_lower_priority_pulse_in_same_frame() -> None:
    arbiter = make_arbiter()

    out = arbiter.resolve([ev("d", PULSE), ev("c", PULSE)], 0.0)

    assert summary(out) == [("c", PULSE)]


def test_exclusive_group_drops_pulse_when_higher_priority_hold_is_granted() -> None:
    arbiter = make_arbiter()
    arbiter.resolve([ev("c", START)], 0.0)

    assert arbiter.resolve([ev("d", PULSE)], 0.1) == []


def test_synthesized_events_use_resolve_time() -> None:
    arbiter = make_arbiter()
    arbiter.resolve([ev("a", START, 1.0)], 1.0)

    out = arbiter.resolve([ev("both", START, 2.0)], 2.0)

    assert [(e.gesture, e.kind, e.t) for e in out] == [("a", END, 2.0), ("both", START, 2.0)]


def test_reset_clears_state_without_events() -> None:
    arbiter = make_arbiter()
    arbiter.resolve([ev("a", START), ev("c", START)], 0.0)

    arbiter.reset()

    assert arbiter.granted == frozenset()
    assert arbiter.resolve([], 0.1) == []  # nada a soltar: o SafetyManager já soltou
    # o upstream fez reset() também: um novo START volta a ser liberado
    assert summary(arbiter.resolve([ev("a", START)], 0.2)) == [("a", START)]


def test_unlisted_gesture_passes_through() -> None:
    arbiter = GestureArbiter(priority=["a"])

    out = arbiter.resolve([ev("z", START), ev("a", START)], 0.0)

    assert summary(out) == [("a", START), ("z", START)]  # fora do perfil: prioridade mínima


@pytest.mark.parametrize(
    "kwargs",
    [
        {"priority": ["a", "a"]},
        {"priority": ["a"], "exclusive_groups": [frozenset({"a", "x"})]},
        {"priority": ["a", "b"], "combos": {"both": frozenset({"a", "b"})}},
        {"priority": ["a", "both"], "combos": {"both": frozenset({"a", "b"})}},
        {"priority": ["both"], "combos": {"both": frozenset({"both"})}},
    ],
)
def test_arbiter_rejects_inconsistent_configuration(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"arbiter|priority"):
        GestureArbiter(**kwargs)  # type: ignore[arg-type]


# --- Spec 017: prioridades do mapa de controles do Minecraft (RF2) ----------------------------

# Seção ``arbiter`` do RF2 (a 017 só lista gestos existentes; aqui valem também os futuros).
MINECRAFT_ARBITER = ArbiterConfig(
    priority=(
        "both_arms",
        "crouch",
        "jump",
        "attack",
        "use",
        "walk",
        "sprint",
        "walk_back",
        "strafe_left",
        "strafe_right",
    ),
    exclusive=(
        frozenset({"crouch", "jump"}),
        frozenset({"crouch", "walk", "sprint", "walk_back"}),
        frozenset({"walk", "sprint", "walk_back"}),
        frozenset({"strafe_left", "strafe_right"}),
    ),
    combos={"both_arms": frozenset({"attack", "use"})},
)

_XFAIL_T2 = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="spec 017 T2")


def _kinds(events: list[GestureEvent]) -> list[tuple[str, EventKind]]:
    return [(e.gesture, e.kind) for e in events]


@_XFAIL_T2
def test_crouch_beats_walk_and_walk_returns() -> None:
    """CA3: agachar e andar juntos liberam só ``crouch``; ao soltar, ``walk`` volta com START."""
    arbiter, _ = build_arbiter({"crouch", "walk"}, MINECRAFT_ARBITER)
    out = arbiter.resolve([ev("walk", START), ev("crouch", START)], 0.0)
    assert _kinds(out) == [("crouch", START)]
    out = arbiter.resolve([ev("crouch", END, 1.0)], 1.0)
    assert _kinds(out) == [("crouch", END), ("walk", START)]
    assert arbiter.granted == frozenset({"walk"})


@_XFAIL_T2
def test_walk_coexists_with_jump() -> None:
    """CA4: ``walk`` convive com ``jump`` e com ``strafe_left`` (fora dos grupos de ``crouch``)."""
    arbiter, _ = build_arbiter({"walk", "jump", "strafe_left"}, MINECRAFT_ARBITER)
    out = arbiter.resolve([ev("walk", START), ev("strafe_left", START)], 0.0)
    assert {e.gesture for e in out} == {"walk", "strafe_left"}
    out = arbiter.resolve([ev("jump", PULSE, 1.0)], 1.0)
    assert _kinds(out) == [("jump", PULSE)]
    assert arbiter.granted == frozenset({"walk", "strafe_left"})
