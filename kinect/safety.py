"""Segurança de input: nenhum caminho de execução deixa tecla ou botão pressionado.

Núcleo puro: o tempo entra como parâmetro e o input sai por um ``InputBackend``.
"""

import threading
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
    """Único caminho até o ``InputBackend``; mantém o conjunto P de pressionados.

    Thread-safe: um watchdog pode chamar ``release_all()`` de outra thread.
    """

    def __init__(
        self, backend: InputBackend, always_release: frozenset[KeyName] = MOUSE_BUTTONS
    ) -> None:
        self._backend = backend
        self._always_release = always_release
        self._pressed: set[KeyName] = set()
        self._enabled = True
        self._lock = threading.Lock()

    @property
    def pressed(self) -> frozenset[KeyName]:
        with self._lock:
            return frozenset(self._pressed)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Desabilitado, ``press`` é ignorado; ``release`` continua funcionando."""
        with self._lock:
            self._enabled = enabled

    def press(self, key: KeyName) -> None:
        with self._lock:
            if not self._enabled or key in self._pressed:
                return
            self._backend.press(key)
            self._pressed.add(key)

    def release(self, key: KeyName) -> None:
        with self._lock:
            if key not in self._pressed:
                return
            self._backend.release(key)
            self._pressed.discard(key)

    def release_all(self) -> None:
        """Solta P ∪ ``always_release``; sem falhas do backend, P = ∅ ao final (S4).

        Tenta soltar todas as teclas mesmo se alguma falhar; as que falharam continuam
        em P (permite nova tentativa) e o primeiro erro é relançado ao final.
        """
        with self._lock:
            first_error: Exception | None = None
            for key in sorted(self._pressed | self._always_release):
                try:
                    self._backend.release(key)
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                self._pressed.discard(key)
            if first_error is not None:
                raise first_error


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
