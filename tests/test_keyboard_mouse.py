"""Spec 001 — backend de teclado e mouse sobre o pydirectinput (CA8)."""

import pytest

from kinect.input.keyboard_mouse import KeyboardMouseBackend
from tests.fakes import FakeDirectInput


def test_init_sets_pause_zero_and_failsafe_off() -> None:
    api = FakeDirectInput()

    KeyboardMouseBackend(api=api)

    assert api.PAUSE == 0
    assert api.FAILSAFE is False


@pytest.mark.parametrize(
    ("key", "button"),
    [("mouse_left", "left"), ("mouse_right", "right"), ("mouse_middle", "middle")],
)
def test_mouse_buttons_map_to_mouse_calls(key: str, button: str) -> None:
    api = FakeDirectInput()
    backend = KeyboardMouseBackend(api=api)

    backend.press(key)
    backend.release(key)

    assert api.calls == [("mouseDown", button), ("mouseUp", button)]


def test_keys_map_to_key_calls() -> None:
    api = FakeDirectInput()
    backend = KeyboardMouseBackend(api=api)

    backend.press("space")
    backend.release("space")

    assert api.calls == [("keyDown", "space"), ("keyUp", "space")]


@pytest.mark.parametrize("key", ["tecla_inexistente", "f13"])
def test_unknown_key_raises(key: str) -> None:
    api = FakeDirectInput()
    backend = KeyboardMouseBackend(api=api)

    with pytest.raises(ValueError, match=key):
        backend.press(key)
    with pytest.raises(ValueError, match=key):
        backend.release(key)
    assert api.calls == []
