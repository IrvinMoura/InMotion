"""Combos derivados: um gesto virtual ativo enquanto todos os membros estão ativos.

Núcleo puro. O combo não lê a pose: deriva START/END dos eventos dos membros, e o
``GestureArbiter`` o usa para suprimir os membros (ex.: ``both_arms`` = ``{attack, use}``).
"""

from __future__ import annotations

from collections.abc import Sequence

from inmotion.gestures.base import EventKind, GestureEvent


class ComboGesture:
    """``c_n = ⋀ m_n`` sobre os membros HOLD ``m`` (ativos = START sem END).

    Emite START no frame em que o último membro ativa e END quando algum deixa de estar
    ativo. Como depende só de START/END (bem formados, E3), seus eventos também são E3.
    """

    def __init__(self, name: str, members: frozenset[str]) -> None:
        if not members or name in members:
            raise ValueError("combo deve ter membros e não conter a si mesmo")
        self.name = name
        self.members = members
        self._active_members: set[str] = set()
        self._active = False

    def derive(self, events: Sequence[GestureEvent], t: float) -> list[GestureEvent]:
        """Eventos do combo (0 ou 1) dado o que os membros emitiram neste frame."""
        for event in events:
            if event.gesture not in self.members:
                continue
            if event.kind is EventKind.START:
                self._active_members.add(event.gesture)
            elif event.kind is EventKind.END:
                self._active_members.discard(event.gesture)
        active = self._active_members >= self.members
        if active == self._active:
            return []
        self._active = active
        return [GestureEvent(self.name, EventKind.START if active else EventKind.END, t)]

    def reset(self) -> None:
        """Esquece tudo SEM emitir evento (o ``SafetyManager`` já soltou)."""
        self._active_members.clear()
        self._active = False
