"""``ActionMapper``: traduz eventos de gesto já arbitrados em input pelo perfil de jogo.

Núcleo puro: o tempo entra como parâmetro ``t`` e o input sai por ``TrackedInput`` (que
mantém o conjunto ``P`` de pressionados da spec 001).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from inmotion.gestures.base import EventKind, GestureEvent
from inmotion.input.backend import KeyName
from inmotion.safety import TrackedInput

if TYPE_CHECKING:  # só para tipos: o esquema da config importa ``MapperParams`` daqui
    from inmotion.config.schema import GameProfile


@dataclass(frozen=True)
class MapperParams:
    """``tap_hold_s``: duração mínima do tap de um PULSE (> 1 tick do Minecraft, 50 ms)."""

    tap_hold_s: float = 0.08

    def __post_init__(self) -> None:
        if not (math.isfinite(self.tap_hold_s) and self.tap_hold_s > 0.0):
            raise ValueError(f"tap_hold_s deve ser finito e > 0: {self.tap_hold_s}")


class ActionMapper:
    """Eventos → press/release, conforme os ``bindings`` do ``GameProfile``.

    - START → ``press`` do input; END → ``release`` (hold).
    - PULSE → ``press`` agora e ``release`` em ``t + tap_hold_s`` (tap), solto por
      ``update(t)``; um novo PULSE da mesma tecla antes do fim estende o prazo.
    - VALUE (analógico) e gestos sem binding (ex.: o combo ``both_arms``) são ignorados.

    Segurança: o mapper nunca é a única barreira; ``release_all()`` do ``SafetyManager``
    solta tudo (pausa, perda, saída, exceção) e ``reset()`` esquece os taps pendentes.
    """

    def __init__(self, profile: GameProfile, out: TrackedInput, params: MapperParams) -> None:
        self._profile = profile
        self._out = out
        self._params = params
        self._due: dict[KeyName, float] = {}  # tecla -> instante em que o tap termina
        self._holds: dict[KeyName, set[str]] = {}

    @property
    def pending(self) -> frozenset[KeyName]:
        """Teclas com tap em andamento (a soltar por tempo)."""
        return frozenset(self._due)

    def handle(self, events: Sequence[GestureEvent], t: float) -> None:
        """Aplica os eventos do frame; antes, solta os taps já vencidos em ``t``."""
        self.update(t)
        for event in events:
            binding = self._profile.bindings.get(event.gesture)
            if binding is None:
                continue
            key = binding.input
            if event.kind is EventKind.START:
                self._out.press(key)
                self._holds.setdefault(key, set()).add(event.gesture)
            elif event.kind is EventKind.END:
                owners = self._holds.get(key, set())
                owners.discard(event.gesture)
                if not owners:
                    self._holds.pop(key, None)
                    if key not in self._due:
                        self._out.release(key)
            elif event.kind is EventKind.PULSE:
                self._out.press(key)
                self._due[key] = t + self._params.tap_hold_s

    def update(self, t: float) -> None:
        """Solta os taps com ``t ≥ t₀ + tap_hold_s``."""
        for key in [k for k, due in self._due.items() if t >= due]:
            del self._due[key]
            if key not in self._holds:
                self._out.release(key)

    def reset(self) -> None:
        """Esquece taps e holds SEM soltar nada (o ``SafetyManager`` já soltou)."""
        self._due.clear()
        self._holds.clear()
