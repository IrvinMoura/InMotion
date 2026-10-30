"""Spec 008 T8 — ActionMapper: eventos de gesto -> input, com taps temporizados."""

from __future__ import annotations

import pytest

from inmotion.config.schema import Binding, GameProfile
from inmotion.gestures.base import EventKind, GestureEvent, GestureKind
from inmotion.input.backend import MOUSE_BUTTONS
from inmotion.mapping import ActionMapper, MapperParams
from inmotion.safety import TrackedInput
from tests.fakes import FakeInputBackend

PROFILE = GameProfile(
    "teste",
    {
        "attack": Binding("mouse_left", GestureKind.HOLD),
        "jump": Binding("space", GestureKind.TRIGGER),
    },
)


def make_mapper(
    tap_hold_s: float = 0.08,
) -> tuple[ActionMapper, FakeInputBackend, TrackedInput]:
    fake = FakeInputBackend()
    out = TrackedInput(fake, MOUSE_BUTTONS)
    return ActionMapper(PROFILE, out, MapperParams(tap_hold_s=tap_hold_s)), fake, out


def ev(gesture: str, kind: EventKind, t: float) -> GestureEvent:
    return GestureEvent(gesture, kind, t)


def test_hold_press_release() -> None:
    """CA6: START aperta e END solta o input do binding."""
    mapper, fake, _ = make_mapper()
    mapper.handle([ev("attack", EventKind.START, 0.0)], 0.0)
    assert fake.events == [("press", "mouse_left")]
    mapper.update(1.0)
    assert fake.pressed == {"mouse_left"}  # segura enquanto durar
    mapper.handle([ev("attack", EventKind.END, 1.0)], 1.0)
    assert fake.events == [("press", "mouse_left"), ("release", "mouse_left")]


def test_trigger_becomes_timed_tap() -> None:
    """CA6: PULSE aperta em t e solta em t + tap_hold_s (no primeiro update vencido)."""
    mapper, fake, _ = make_mapper(tap_hold_s=0.08)
    mapper.handle([ev("jump", EventKind.PULSE, 1.0)], 1.0)
    assert fake.events == [("press", "space")]
    mapper.update(1.05)
    assert fake.pressed == {"space"}  # ainda dentro dos 80 ms
    mapper.update(1.08)
    assert fake.events == [("press", "space"), ("release", "space")]
    mapper.update(1.2)
    assert len(fake.events) == 2  # solta uma vez só


def test_reset_drops_pending_taps() -> None:
    """reset() esquece os taps pendentes: nada é solto depois (a segurança já soltou)."""
    mapper, fake, out = make_mapper()
    mapper.handle([ev("jump", EventKind.PULSE, 0.0)], 0.0)
    out.release_all()  # o SafetyManager solta tudo
    events_before = list(fake.events)
    mapper.reset()
    out.press("space")  # input novo, de outro caminho
    mapper.update(10.0)
    assert fake.events == [*events_before, ("press", "space")]  # o mapper não o solta


def test_gesture_without_binding_is_ignored() -> None:
    mapper, fake, _ = make_mapper()
    mapper.handle([ev("both_arms", EventKind.START, 0.0), ev("walk", EventKind.PULSE, 0.0)], 0.0)
    assert fake.events == []


def test_repeated_pulse_extends_the_tap() -> None:
    mapper, fake, _ = make_mapper(tap_hold_s=0.08)
    mapper.handle([ev("jump", EventKind.PULSE, 0.0)], 0.0)
    mapper.handle([ev("jump", EventKind.PULSE, 0.05)], 0.05)
    mapper.update(0.09)
    assert fake.pressed == {"space"}  # 0,05 + 0,08 = 0,13
    mapper.update(0.13)
    assert fake.pressed == set()
    assert fake.events == [("press", "space"), ("release", "space")]


def test_due_tap_released_before_new_events_in_same_update() -> None:
    mapper, fake, _ = make_mapper(tap_hold_s=0.08)
    mapper.handle([ev("jump", EventKind.PULSE, 0.0)], 0.0)
    mapper.update(0.1)
    mapper.handle([ev("jump", EventKind.PULSE, 0.1)], 0.1)
    assert fake.events == [("press", "space"), ("release", "space"), ("press", "space")]


def test_pending_keys_exposed_and_tap_hold_validation() -> None:
    mapper, _, _ = make_mapper()
    mapper.handle([ev("jump", EventKind.PULSE, 0.0)], 0.0)
    assert mapper.pending == frozenset({"space"})
    mapper.reset()
    assert mapper.pending == frozenset()
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tap_hold_s"):
            MapperParams(tap_hold_s=bad)
