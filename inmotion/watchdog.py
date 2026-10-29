"""Watchdog de loop travado: temporizador puro, com o tempo passado como parâmetro.

Núcleo puro. Uma thread do app consulta ``expired(now)`` e, se o loop parou de bater,
chama ``SafetyManager.on_stall()``.
"""

import math
import threading


class Watchdog:
    """Detecta ausência de ``beat`` por mais de ``timeout_s`` segundos.

    Thread-safe: o loop chama ``beat`` e a thread do watchdog chama ``expired``.
    """

    def __init__(self, timeout_s: float = 0.5) -> None:
        if not (math.isfinite(timeout_s) and timeout_s > 0):
            raise ValueError(f"timeout_s deve ser finito e > 0: {timeout_s}")
        self._timeout_s = timeout_s
        self._last_beat: float | None = None
        self._armed = False
        self._lock = threading.Lock()

    def beat(self, t: float) -> None:
        """Registra que o loop está vivo em ``t`` e rearma o watchdog."""
        if not math.isfinite(t):
            raise ValueError(f"t deve ser finito: {t}")
        with self._lock:
            self._last_beat = t
            self._armed = True

    def expired(self, now: float) -> bool:
        """True uma única vez por travamento: quando ``now - último beat >= timeout_s``.

        Só volta a ser True depois de um novo ``beat``. Antes do primeiro ``beat``,
        é sempre False (o loop ainda não começou).
        """
        with self._lock:
            if not self._armed or self._last_beat is None:
                return False
            if now - self._last_beat >= self._timeout_s:
                self._armed = False
                return True
            return False
