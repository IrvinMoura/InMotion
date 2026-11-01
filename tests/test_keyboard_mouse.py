"""Spec 001 — backend de teclado e mouse sobre o pydirectinput (CA8)."""

import statistics
import sys
import time

import pytest

from inmotion.input.hotkeys import WindowsHotkeys
from inmotion.input.keyboard_mouse import KeyboardMouseBackend
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


def test_supported_keys() -> None:
    """Teclas com scancode mais os botões do mouse; ``f13`` (sem scancode) fica de fora."""
    keys = KeyboardMouseBackend(api=FakeDirectInput()).supported_keys()

    assert keys == frozenset(
        {"space", "shift", "ctrl", "w", "mouse_left", "mouse_right", "mouse_middle"}
    )


# --- Hardware (rodar com `pytest -m hardware`; enviam input real ao Windows) ----------

# Scroll Lock alterna um LED: os testes sempre dão um número PAR de toques, para que o
# estado volte ao original. Shift (Teclas de Aderência) e Ctrl (atalhos de duplo toque)
# foram evitados de propósito.
HARMLESS_KEY = "scrolllock"
windows_only = pytest.mark.skipif(sys.platform != "win32", reason="input real só no Windows")


def tap(backend: KeyboardMouseBackend, hotkeys: WindowsHotkeys) -> tuple[bool, bool]:
    """Aperta e solta a tecla; devolve (vista pressionada, vista solta) pelo SO."""
    try:
        backend.press(HARMLESS_KEY)
        down = HARMLESS_KEY in hotkeys.poll()
    finally:
        backend.release(HARMLESS_KEY)
    up = HARMLESS_KEY not in hotkeys.poll()
    return down, up


@pytest.mark.hardware
@windows_only
def test_real_press_release_roundtrip() -> None:
    backend = KeyboardMouseBackend()
    hotkeys = WindowsHotkeys(frozenset({HARMLESS_KEY}))

    results = [tap(backend, hotkeys) for _ in range(2)]

    assert results == [(True, True), (True, True)]


@pytest.mark.hardware
@windows_only
def test_press_release_latency_p95() -> None:
    """RNF2 / CA12: cada chamada de press/release leva < 5 ms (p95)."""
    backend = KeyboardMouseBackend()
    durations_ms: list[float] = []
    for _ in range(100):  # 100 pares: número par de toques
        try:
            start = time.perf_counter()
            backend.press(HARMLESS_KEY)
            durations_ms.append((time.perf_counter() - start) * 1000)
        finally:
            start = time.perf_counter()
            backend.release(HARMLESS_KEY)
            durations_ms.append((time.perf_counter() - start) * 1000)

    p95 = statistics.quantiles(durations_ms, n=20)[-1]
    print(f"latência press/release: p95 = {p95:.3f} ms, máx = {max(durations_ms):.3f} ms")
    assert p95 < 5.0
