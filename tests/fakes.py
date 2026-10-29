"""Dublês de teste para os ports de entrada (sem hardware, sem Windows)."""

from collections.abc import Iterable


class SimulatedReleaseError(Exception):
    """Falha simulada ao soltar uma tecla (não herda de RuntimeError de propósito)."""


class FakeInputBackend:
    """Registra cada press/release; pode simular falha ao soltar certas teclas."""

    def __init__(self, fail_on_release: Iterable[str] = ()) -> None:
        self.events: list[tuple[str, str]] = []
        self.pressed: set[str] = set()
        self.fail_on_release = set(fail_on_release)

    def press(self, key: str) -> None:
        self.events.append(("press", key))
        self.pressed.add(key)

    def release(self, key: str) -> None:
        self.events.append(("release", key))
        if key in self.fail_on_release:
            raise SimulatedReleaseError(f"falha simulada ao soltar {key}")
        self.pressed.discard(key)

    def released(self) -> list[str]:
        return [key for kind, key in self.events if kind == "release"]


class FakeHotkeys:
    """Devolve uma sequência roteirizada de teclas globais; depois, nenhuma."""

    def __init__(self, script: Iterable[frozenset[str]]) -> None:
        self._script = list(script)

    def poll(self) -> frozenset[str]:
        return self._script.pop(0) if self._script else frozenset()


class FakeDirectInput:
    """Imita a API do pydirectinput usada pelo backend (sem enviar input real)."""

    def __init__(self) -> None:
        self.PAUSE = 0.1
        self.FAILSAFE = True
        self.KEYBOARD_MAPPING: dict[str, int | None] = {
            "space": 57,
            "shift": 42,
            "w": 17,
            "f13": None,  # existe no dicionário do pydirectinput, mas sem scancode
        }
        self.calls: list[tuple[str, str]] = []

    def keyDown(self, key: str) -> None:  # noqa: N802 - nome da API real
        self.calls.append(("keyDown", key))

    def keyUp(self, key: str) -> None:  # noqa: N802
        self.calls.append(("keyUp", key))

    def mouseDown(self, *, button: str) -> None:  # noqa: N802
        self.calls.append(("mouseDown", button))

    def mouseUp(self, *, button: str) -> None:  # noqa: N802
        self.calls.append(("mouseUp", button))
