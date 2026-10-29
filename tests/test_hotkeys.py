"""Spec 001 — teclas globais via GetAsyncKeyState (CA8b, ADR 0003)."""

import pytest

from inmotion.input.hotkeys import WindowsHotkeys
from tests.fakes import FakeUser32

VK_F8, VK_F9 = 0x77, 0x78
PRESSED, PRESSED_SINCE_LAST_CALL = 0x8000, 0x0001


def test_poll_returns_pressed_keys() -> None:
    user32 = FakeUser32({VK_F8: PRESSED, VK_F9: 0})
    hotkeys = WindowsHotkeys(frozenset({"f8", "f9"}), user32=user32)

    assert hotkeys.poll() == frozenset({"f8"})

    user32.states = {VK_F8: PRESSED, VK_F9: PRESSED | PRESSED_SINCE_LAST_CALL}
    assert hotkeys.poll() == frozenset({"f8", "f9"})


def test_poll_ignores_low_bit() -> None:
    """Só o bit 0x8000 (pressionada agora) conta; o bit 0x0001 não é confiável."""
    user32 = FakeUser32({VK_F8: PRESSED_SINCE_LAST_CALL})
    hotkeys = WindowsHotkeys(frozenset({"f8"}), user32=user32)

    assert hotkeys.poll() == frozenset()


def test_unknown_key_name_raises() -> None:
    with pytest.raises(ValueError, match="tecla_inexistente"):
        WindowsHotkeys(frozenset({"f8", "tecla_inexistente"}), user32=FakeUser32())


def test_digit_keys_supported() -> None:
    """Spec 011 T4: dígitos 0 a 9 mapeiam para VK 0x30 a 0x39."""
    user32 = FakeUser32({0x31: PRESSED, 0x35: PRESSED, 0x30: 0, 0x39: PRESSED_SINCE_LAST_CALL})
    hotkeys = WindowsHotkeys(frozenset(str(n) for n in range(10)), user32=user32)

    assert hotkeys.poll() == frozenset({"1", "5"})
