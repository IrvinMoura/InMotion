"""Spec 008 T9 — fábrica dos gestos de corpo, árbitro e combo ``both_arms``."""

from __future__ import annotations

import pytest

from inmotion.gestures.arbiter import ArbiterConfig
from inmotion.gestures.base import EventKind, GestureEvent
from inmotion.gestures.combo import ComboGesture
from inmotion.gestures.factory import BODY_GESTURE_NAMES, build_arbiter, build_body_gestures


def ev(gesture: str, kind: EventKind, t: float = 0.0) -> GestureEvent:
    return GestureEvent(gesture, kind, t)


def test_only_gestures_in_the_profile_exist() -> None:
    gestures = build_body_gestures({"jump", "walk"})
    assert [g.name for g in gestures] == ["jump", "walk"]
    assert {g.name for g in build_body_gestures(BODY_GESTURE_NAMES)} == BODY_GESTURE_NAMES


def test_gesture_names_of_the_body_set() -> None:
    """Spec 018: `walk` e `sprint` são a marcha; o inclinar chama `lean` (sem binding)."""
    assert {"attack", "use", "jump", "crouch", "walk", "sprint", "lean"} == BODY_GESTURE_NAMES
    names = [g.name for g in build_body_gestures({"lean", "sprint", "walk"})]
    assert sorted(names) == ["lean", "sprint", "walk"]


def test_arbiter_groups_only_when_members_exist() -> None:
    arbiter, combo = build_arbiter({"attack"})
    assert combo is None
    assert arbiter.resolve([ev("attack", EventKind.START)], 0.0)[0].kind is EventKind.START
    _, combo = build_arbiter({"attack", "use"})
    assert combo is not None
    assert combo.members == {"attack", "use"}


def test_jump_and_crouch_are_exclusive() -> None:
    arbiter, _ = build_arbiter(BODY_GESTURE_NAMES)
    out = arbiter.resolve([ev("crouch", EventKind.START), ev("jump", EventKind.PULSE)], 0.0)
    assert [e.gesture for e in out] == ["jump"]  # o pulo é mais prioritário que agachar


def test_combo_starts_when_all_members_active_and_ends_when_one_leaves() -> None:
    combo = ComboGesture("both", frozenset({"a", "b"}))
    assert combo.derive([ev("a", EventKind.START)], 0.0) == []
    assert combo.derive([ev("b", EventKind.START)], 1.0) == [ev("both", EventKind.START, 1.0)]
    assert combo.derive([], 2.0) == []  # sem mudança, sem evento
    assert combo.derive([ev("a", EventKind.END, 3.0)], 3.0) == [ev("both", EventKind.END, 3.0)]
    combo.reset()
    assert combo.derive([ev("b", EventKind.START)], 4.0) == []


def test_combo_same_frame_and_validation() -> None:
    combo = ComboGesture("both", frozenset({"a", "b"}))
    out = combo.derive([ev("a", EventKind.START), ev("b", EventKind.START)], 0.0)
    assert [e.kind for e in out] == [EventKind.START]
    for members in (frozenset(), frozenset({"both"})):
        with pytest.raises(ValueError, match="combo"):
            ComboGesture("both", members)


def test_arbiter_built_from_profile_config() -> None:
    """CA1: o árbitro segue a prioridade, os grupos e o combo da config (não do código)."""
    config = ArbiterConfig(
        priority=("both_arms", "crouch", "jump", "attack", "use"),
        exclusive=(frozenset({"crouch", "jump"}),),
        combos={"both_arms": frozenset({"attack", "use"})},
    )
    arbiter, combo = build_arbiter({"attack", "use", "jump", "crouch"}, config)
    assert combo is not None
    assert combo.name == "both_arms"
    assert combo.members == {"attack", "use"}
    # Prioridade da config: agachar vence o pulo (o padrão antigo dava o contrário).
    out = arbiter.resolve([ev("crouch", EventKind.START), ev("jump", EventKind.PULSE)], 0.0)
    assert [e.gesture for e in out] == ["crouch"]
