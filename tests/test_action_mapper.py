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


def test_punch_survives_pause_resume() -> None:
    """CA6b: soco mantido + F8 duas vezes -> ``mouse_left`` solto e re-apertado após N frames."""
    from inmotion.fps import FpsMeter
    from inmotion.gestures.factory import build_body_gestures
    from inmotion.pipeline import Pipeline
    from inmotion.safety import SafetyConfig, SafetyManager
    from tests.helpers import ARM_STRAIGHT_FORWARD, make_body

    fake = FakeInputBackend()
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(min_track_s=0.0))
    profile = GameProfile("teste", {"attack": Binding("mouse_left", GestureKind.HOLD)})
    pipeline = Pipeline(safety, build_body_gestures({"attack"}), FpsMeter(), profile=profile)
    dt = 1 / 30
    frame = 0

    def step(n: int) -> None:
        nonlocal frame
        for _ in range(n):
            t = frame * dt
            pipeline.step(make_body(t=t, right_arm=ARM_STRAIGHT_FORWARD), t)
            frame += 1

    def left() -> list[tuple[str, str]]:
        return [e for e in fake.events if e[1] == "mouse_left"]  # release_all solta os 3 botões

    step(10)
    assert left() == [("press", "mouse_left")]
    safety.on_hotkeys(frozenset({"f8"}))  # pausa: solta tudo
    step(3)
    assert left() == [("press", "mouse_left"), ("release", "mouse_left")]
    safety.on_hotkeys(frozenset())
    safety.on_hotkeys(frozenset({"f8"}))  # retoma
    step(2)  # N = 3: ainda não re-armou
    assert fake.pressed == set()
    step(1)
    assert left()[-1] == ("press", "mouse_left")  # re-apertado após N frames
    assert left().count(("press", "mouse_left")) == 2


def test_shared_key_remains_pressed_until_all_holds_end() -> None:
    fake = FakeInputBackend()
    profile = GameProfile(
        "shared",
        {
            "walk": Binding("w", GestureKind.HOLD),
            "lean": Binding("w", GestureKind.HOLD),
        },
    )
    mapper = ActionMapper(profile, TrackedInput(fake), MapperParams())
    mapper.handle([ev("walk", EventKind.START, 0.0), ev("lean", EventKind.START, 0.0)], 0.0)
    mapper.handle([ev("walk", EventKind.END, 0.1)], 0.1)
    assert fake.pressed == {"w"}
    mapper.handle([ev("lean", EventKind.END, 0.2)], 0.2)
    assert fake.events == [("press", "w"), ("release", "w")]


@pytest.mark.parametrize("tap_first", [True, False])
def test_shared_hold_and_tap_do_not_release_each_other(tap_first: bool) -> None:
    fake = FakeInputBackend()
    profile = GameProfile(
        "shared",
        {
            "walk": Binding("space", GestureKind.HOLD),
            "jump": Binding("space", GestureKind.TRIGGER),
        },
    )
    mapper = ActionMapper(profile, TrackedInput(fake), MapperParams())
    mapper.handle([ev("walk", EventKind.START, 0.0), ev("jump", EventKind.PULSE, 0.0)], 0.0)
    if tap_first:
        mapper.update(0.1)
        assert fake.pressed == {"space"}
        mapper.handle([ev("walk", EventKind.END, 0.2)], 0.2)
    else:
        mapper.handle([ev("walk", EventKind.END, 0.02)], 0.02)
        assert fake.pressed == {"space"}
        mapper.update(0.1)
    assert fake.pressed == set()
    assert fake.events == [("press", "space"), ("release", "space")]
