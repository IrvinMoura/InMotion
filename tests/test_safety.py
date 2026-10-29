"""Spec 001 — SafetyManager: nenhum caminho deixa input pressionado."""

from collections.abc import Callable

import pytest

from kinect.input.backend import MOUSE_BUTTONS, KeyName
from kinect.safety import (
    SafetyConfig,
    SafetyManager,
    SafetyState,
    TrackedInput,
    register_atexit,
)
from tests.fakes import FakeInputBackend, SimulatedReleaseError


def pending(task: str) -> pytest.MarkDecorator:
    """Teste escrito antes da implementação (TDD); a tarefa indicada remove a marca."""
    return pytest.mark.xfail(raises=NotImplementedError, strict=True, reason=f"spec 001 {task}")


# Exceções próprias: NotImplementedError herda de RuntimeError, então usar RuntimeError
# aqui faria o stub "passar" pelo pytest.raises pelo motivo errado.
class BoomError(Exception):
    pass


class FakeFailSafeError(Exception):
    """Imita pydirectinput.FailSafeException sem importar o pydirectinput."""


def make_manager() -> tuple[SafetyManager, FakeInputBackend]:
    fake = FakeInputBackend()
    manager = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig())
    return manager, fake


def press_then_raise(manager: SafetyManager, key: KeyName, exc: BaseException) -> None:
    with manager.guard():
        manager.input.press(key)
        raise exc


# --- TrackedInput.release_all (CA7, CA4b) ---------------------------------------


def test_release_all_empties_pressed() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.press("space")
    tracked.press("w")

    tracked.release_all()

    assert tracked.pressed == frozenset()
    assert fake.pressed == set()


def test_release_all_twice_is_idempotent() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.press("space")
    tracked.release_all()
    fake.events.clear()

    tracked.release_all()

    assert tracked.pressed == frozenset()
    assert set(fake.released()) == MOUSE_BUTTONS


def test_release_all_continues_after_partial_failure() -> None:
    fake = FakeInputBackend(fail_on_release={"w"})
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    for key in ("space", "w", "shift"):
        tracked.press(key)

    with pytest.raises(SimulatedReleaseError, match="w"):
        tracked.release_all()

    assert {"space", "shift"} <= set(fake.released())
    assert fake.pressed == {"w"}
    assert tracked.pressed == frozenset({"w"})  # continua registrada: dá para tentar de novo


def test_press_ignored_when_disabled() -> None:
    fake = FakeInputBackend()
    tracked = TrackedInput(fake, MOUSE_BUTTONS)
    tracked.set_enabled(False)

    tracked.press("space")

    assert not tracked.enabled
    assert tracked.pressed == frozenset()
    assert fake.events == []


# --- guard() (CA4, CA5) ------------------------------------------------------------


def test_guard_releases_on_exception_and_reraises() -> None:
    manager, fake = make_manager()

    with pytest.raises(BoomError):
        press_then_raise(manager, "mouse_left", BoomError())

    assert fake.pressed == set()


def test_guard_releases_on_keyboard_interrupt() -> None:
    manager, fake = make_manager()

    with pytest.raises(KeyboardInterrupt):
        press_then_raise(manager, "space", KeyboardInterrupt())

    assert fake.pressed == set()


def test_guard_releases_on_failsafe_like_exception() -> None:
    manager, fake = make_manager()

    with pytest.raises(FakeFailSafeError):
        press_then_raise(manager, "mouse_right", FakeFailSafeError())

    assert fake.pressed == set()


def test_guard_releases_on_normal_exit() -> None:
    manager, fake = make_manager()

    with manager.guard():
        manager.input.press("w")

    assert fake.pressed == set()


def test_guard_keeps_original_exception_if_release_fails() -> None:
    fake = FakeInputBackend(fail_on_release={"mouse_middle"})
    manager = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig())

    with pytest.raises(BoomError):
        press_then_raise(manager, "space", BoomError())

    assert "space" not in fake.pressed


def test_register_atexit_callback_releases_all() -> None:
    manager, fake = make_manager()
    callbacks: list[Callable[[], None]] = []

    register_atexit(manager, register=callbacks.append)
    manager.input.press("shift")
    assert len(callbacks) == 1

    callbacks[0]()

    assert fake.pressed == set()


# --- Perda de tracking (CA1, CA2) --------------------------------------------------


@pending("T4")
def test_lost_tracking_for_timeout_releases_all() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.input.press("mouse_left")

    manager.on_frame(0.1, tracking_ok=False)
    assert fake.pressed == {"mouse_left"}  # 0,1 s < T

    manager.on_frame(0.3, tracking_ok=False)  # 0,3 s ≥ T = 0,2 s
    assert fake.pressed == set()
    assert manager.state is SafetyState.LOST


@pending("T4")
def test_short_tracking_loss_keeps_keys() -> None:
    manager, fake = make_manager()
    manager.on_frame(0.0, tracking_ok=True)
    manager.input.press("space")

    manager.on_frame(0.05, tracking_ok=False)
    manager.on_frame(0.15, tracking_ok=False)
    manager.on_frame(0.18, tracking_ok=True)

    assert fake.pressed == {"space"}
    assert fake.released() == []
    assert manager.state is SafetyState.ACTIVE


# --- Pausa F8 (CA6) ------------------------------------------------------------------


@pending("T6")
def test_f8_edge_toggles_pause_once() -> None:
    manager, fake = make_manager()
    manager.input.press("mouse_left")

    for _ in range(5):  # F8 segurado por 5 frames
        manager.on_hotkeys(frozenset({"f8"}))

    assert manager.state is SafetyState.PAUSED
    assert fake.pressed == set()
