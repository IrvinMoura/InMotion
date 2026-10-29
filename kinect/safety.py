"""Segurança de input: nenhum caminho de execução deixa tecla ou botão pressionado.

Núcleo puro: o tempo entra como parâmetro e o input sai por um ``InputBackend``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto

from kinect.input.backend import MOUSE_BUTTONS, InputBackend, KeyName


@dataclass(frozen=True)
class SafetyConfig:
    lost_timeout_s: float = 0.2
    pause_key: KeyName = "f8"
    exit_key: KeyName = "f9"
    always_release: frozenset[KeyName] = MOUSE_BUTTONS


class SafetyState(Enum):
    ACTIVE = auto()
    PAUSED = auto()
    LOST = auto()


class SafetyAction(Enum):
    NONE = auto()
    EXIT = auto()


class TrackedInput:
    """Único caminho até o ``InputBackend``; mantém o conjunto P de pressionados."""

    def __init__(
        self, backend: InputBackend, always_release: frozenset[KeyName] = MOUSE_BUTTONS
    ) -> None:
        self._backend = backend
        self._always_release = always_release

    @property
    def pressed(self) -> frozenset[KeyName]:
        raise NotImplementedError

    @property
    def enabled(self) -> bool:
        raise NotImplementedError

    def set_enabled(self, enabled: bool) -> None:
        raise NotImplementedError

    def press(self, key: KeyName) -> None:
        raise NotImplementedError

    def release(self, key: KeyName) -> None:
        raise NotImplementedError

    def release_all(self) -> None:
        raise NotImplementedError


class SafetyManager:
    """Política de segurança: perda de tracking, pausa, saída e exceções."""

    def __init__(self, tracked: TrackedInput, config: SafetyConfig) -> None:
        self._tracked = tracked
        self._config = config

    @property
    def state(self) -> SafetyState:
        raise NotImplementedError

    @property
    def input(self) -> TrackedInput:
        raise NotImplementedError

    def on_frame(self, t: float, tracking_ok: bool) -> None:
        raise NotImplementedError

    def on_hotkeys(self, keys: frozenset[KeyName]) -> SafetyAction:
        raise NotImplementedError

    @contextmanager
    def guard(self) -> Iterator["SafetyManager"]:
        raise NotImplementedError
        yield self


def register_atexit(
    manager: SafetyManager,
    register: Callable[[Callable[[], None]], object] | None = None,
) -> None:
    raise NotImplementedError
