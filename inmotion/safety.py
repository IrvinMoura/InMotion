"""Segurança de input: nenhum caminho de execução deixa tecla ou botão pressionado.

Núcleo puro: o tempo entra como parâmetro e o input sai por um ``InputBackend``.
"""

import atexit
import logging
import math
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto

from inmotion.input.backend import MOUSE_BUTTONS, InputBackend, KeyName

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SafetyConfig:
    """Parâmetros de segurança (os valores efetivos vêm de ``config/default.yaml``, spec 004).

    Attributes:
        lost_timeout_s: tempo sem pose, em segundos, até soltar tudo (``T`` da R5 temporal).
        pause_key: tecla global que alterna a pausa.
        exit_key: tecla global que pede a saída.
        always_release: soltas em todo ``release_all()``, mesmo fora de P. Tem de ser o
            mesmo valor do ``TrackedInput``; o ``SafetyManager`` confere na construção.
        min_torso_visibility: visibilidade mínima de ombros e quadris para a pose contar
            como rastreada (``ℓₙ = 1`` na R5). Abaixo disso é pose "fantasma" ou parcial
            e conta como perda de tracking.
        min_track_s: tempo de pose contínua até ela valer (debounce R5 no tempo). A pose
            "fantasma" de objetos pisca por no máximo 0,73 s (recordings/fantasma.npz,
            29/10/2026); 1,0 s dá folga de 1,37×. ``0`` desliga.
        watchdog_timeout_s: sem batida do loop por mais que isso, o watchdog dá o loop
            como travado e solta todo o input.
    """

    lost_timeout_s: float = 0.2
    pause_key: KeyName = "f8"
    exit_key: KeyName = "f9"
    always_release: frozenset[KeyName] = MOUSE_BUTTONS
    min_torso_visibility: float = 0.5
    min_track_s: float = 1.0
    watchdog_timeout_s: float = 0.5

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lost_timeout_s) and self.lost_timeout_s > 0):
            raise ValueError(f"lost_timeout_s deve ser finito e > 0: {self.lost_timeout_s}")
        for field in ("pause_key", "exit_key"):
            if not getattr(self, field):
                raise ValueError(f"{field} não pode ser vazio")
        if not (math.isfinite(self.min_torso_visibility) and 0 <= self.min_torso_visibility <= 1):
            raise ValueError(
                f"min_torso_visibility deve estar em [0, 1]: {self.min_torso_visibility}"
            )
        if not (math.isfinite(self.min_track_s) and self.min_track_s >= 0):
            raise ValueError(f"min_track_s deve ser finito e >= 0: {self.min_track_s}")
        if not (math.isfinite(self.watchdog_timeout_s) and self.watchdog_timeout_s > 0):
            raise ValueError(f"watchdog_timeout_s deve ser finito e > 0: {self.watchdog_timeout_s}")
        if self.pause_key == self.exit_key:
            raise ValueError(f"pause_key e exit_key devem ser diferentes: {self.pause_key!r}")


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
    def always_release(self) -> frozenset[KeyName]:
        return self._always_release

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
        """Monta a política sobre um ``TrackedInput`` já criado.

        Raises:
            ValueError: se ``tracked.always_release`` diferir de ``config.always_release``.
        """
        if tracked.always_release != config.always_release:
            raise ValueError(
                "always_release do TrackedInput difere do SafetyConfig: "
                f"{sorted(tracked.always_release)} != {sorted(config.always_release)}"
            )
        self._tracked = tracked
        self._config = config
        self._last_t: float | None = None
        self._last_ok_t: float | None = None
        self._lost = False
        self._paused = False
        self._previous_keys: frozenset[KeyName] = frozenset()
        # Reentrante: o watchdog chama ``on_stall`` de outra thread.
        self._lock = threading.RLock()

    @property
    def state(self) -> SafetyState:
        """PAUSED tem prioridade sobre LOST."""
        with self._lock:
            if self._paused:
                return SafetyState.PAUSED
            return SafetyState.LOST if self._lost else SafetyState.ACTIVE

    @property
    def config(self) -> SafetyConfig:
        return self._config

    @property
    def input(self) -> TrackedInput:
        """O que os gestos usam para apertar e soltar."""
        return self._tracked

    def on_frame(self, t: float, tracking_ok: bool) -> None:
        """R5 temporal: solta tudo quando o tempo sem pose atinge ``lost_timeout_s``.

        Recorrência da spec: ``L₀ = 0``, ``Lₙ = ℓₙ·(Lₙ₋₁ + Δtₙ)``, com disparo quando
        ``Lₙ₋₁ < T ≤ Lₙ``. Pela propriedade S1', ``Lₙ = tₙ − t_ok`` (tempo desde o último
        frame com pose), que é o que se calcula aqui, sem acumular erro de ponto
        flutuante. Antes da primeira pose, a referência é o primeiro frame recebido.

        Raises:
            ValueError: se ``t`` não for finito (com NaN, toda comparação é falsa e a
                perda nunca dispararia) ou não for estritamente crescente.
        """
        if not math.isfinite(t):
            raise ValueError(f"t deve ser finito: {t}")
        with self._lock:
            if self._last_t is not None and t <= self._last_t:
                raise ValueError(f"t deve ser estritamente crescente: {t} <= {self._last_t}")
            self._last_t = t

            if tracking_ok:
                self._last_ok_t = t
                if self._lost:
                    self._lost = False
                    self._sync_enabled()
                return

            if self._last_ok_t is None:
                self._last_ok_t = t
            if not self._lost and t - self._last_ok_t >= self._config.lost_timeout_s:
                self._lost = True
                self._sync_enabled()  # antes de soltar: nenhum press entra no meio
                self._tracked.release_all()

    def on_stall(self) -> None:
        """Loop travado (watchdog): solta tudo e desabilita até o próximo ``on_frame``."""
        with self._lock:
            self._lost = True
            self._sync_enabled()  # antes de soltar: nenhum press entra no meio
            self._tracked.release_all()

    def on_hotkeys(self, keys: frozenset[KeyName]) -> SafetyAction:
        """Reage às bordas de subida das teclas globais (pressionar, não segurar).

        Borda de ``pause_key``: alterna a pausa (S3: nº de alternâncias = nº de bordas);
        ao pausar, solta tudo. Borda de ``exit_key``: devolve ``SafetyAction.EXIT``.
        """
        with self._lock:
            rising = keys - self._previous_keys
            self._previous_keys = keys

            if self._config.pause_key in rising:
                self._paused = not self._paused
                self._sync_enabled()
                if self._paused:
                    self._tracked.release_all()

            return SafetyAction.EXIT if self._config.exit_key in rising else SafetyAction.NONE

    def _sync_enabled(self) -> None:
        """Input só é aceito em ACTIVE: nem pausado (S5), nem sem pose."""
        self._tracked.set_enabled(not (self._paused or self._lost))

    @contextmanager
    def guard(self) -> Iterator["SafetyManager"]:
        """Solta todo o input em qualquer saída do bloco; exceções são propagadas.

        Se ``release_all()`` falhar enquanto outra exceção já está em curso, a falha é
        registrada no log e a exceção original segue (não é mascarada).
        """
        try:
            yield self
        except BaseException:
            try:
                with self._lock:
                    self._tracked.release_all()
            except Exception:
                _log.exception("release_all() falhou durante o tratamento de outra exceção")
            raise
        else:
            with self._lock:
                self._tracked.release_all()


def register_atexit(
    manager: SafetyManager,
    register: Callable[[Callable[[], None]], object] | None = None,
) -> None:
    """Registra ``release_all()`` para a saída do interpretador (padrão: ``atexit``)."""
    (register or atexit.register)(manager.input.release_all)
