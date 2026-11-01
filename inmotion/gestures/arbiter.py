"""Árbitro de conflitos entre gestos: grupos exclusivos e combinações (RF7).

Recebe os eventos de todos os gestos no frame e devolve os eventos "liberados" para o
`ActionMapper`. A saída preserva a bem-formação E3 por gesto: START/END alternam e,
quando o upstream fecha todos os gestos (`close()`), os START igualam os END.

Modelo: o árbitro guarda o conjunto `wanted` de gestos HOLD que o upstream considera
ativos (START sem END) e deriva dele, a cada frame e sem histórico, o conjunto `granted`:

1. combos ativos, na ordem de prioridade, suprimem os seus membros;
2. os demais gestos entram na ordem de prioridade, e num grupo exclusivo só o primeiro
   (o mais prioritário) é liberado.

A saída é a diferença entre o `granted` anterior e o novo: END para os que saem (antes),
START para os que entram. Gestos suprimidos que continuam "querendo" voltam com START
quando o bloqueio termina. PULSE passa pelas mesmas regras no frame; VALUE (analógico)
sempre passa.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from inmotion.gestures.base import EventKind, GestureEvent


@dataclass(frozen=True)
class ArbiterConfig:
    """Prioridade (antes = mais forte), grupos exclusivos e combos do árbitro (spec 017 RF2).

    Vem da seção ``arbiter`` do perfil do jogo. Nomes sem gesto montado são ignorados na
    montagem do árbitro (``build_arbiter``): um gesto sem binding não influencia nada.
    """

    priority: tuple[str, ...]
    exclusive: tuple[frozenset[str], ...] = ()
    combos: dict[str, frozenset[str]] = field(default_factory=dict)


# Comportamento anterior à 017 (premissa 4): perfis sem seção ``arbiter`` não mudam.
DEFAULT_ARBITER = ArbiterConfig(
    priority=("both_arms", "attack", "use", "jump", "crouch", "walk"),
    exclusive=(frozenset({"jump", "crouch"}),),
    combos={"both_arms": frozenset({"attack", "use"})},
)


class GestureArbiter:
    """Aplica exclusividade e combinações aos eventos de gestos de um frame.

    `priority` é a ordem do perfil de jogo (antes = mais prioritário); gestos que não
    constam dela têm prioridade mínima. Nomes de grupos e combos devem constar em `priority`.
    """

    def __init__(
        self,
        priority: Sequence[str],
        exclusive_groups: Sequence[frozenset[str]] = (),
        combos: Mapping[str, frozenset[str]] | None = None,
    ) -> None:
        if len(set(priority)) != len(priority):
            raise ValueError("arbiter: priority tem nomes repetidos")
        known = set(priority)
        self._rank = {name: i for i, name in enumerate(priority)}
        self._groups = tuple(frozenset(g) for g in exclusive_groups)
        self._combos = {name: frozenset(members) for name, members in (combos or {}).items()}
        for group in self._groups:
            if not group <= known:
                raise ValueError(
                    f"arbiter: grupo exclusivo fora de priority: {sorted(group - known)}"
                )
        for name, members in self._combos.items():
            if name not in known or not members <= known:
                raise ValueError(f"arbiter: combo {name!r} usa nomes fora de priority")
            if name in members or not members:
                raise ValueError(
                    f"arbiter: combo {name!r} deve ter membros e não conter a si mesmo"
                )
        self._wanted: set[str] = set()
        self._granted: frozenset[str] = frozenset()

    @property
    def granted(self) -> frozenset[str]:
        """Gestos HOLD liberados agora (o que o `ActionMapper` deve manter pressionado)."""
        return self._granted

    def reset(self) -> None:
        """Esquece tudo SEM emitir evento (o SafetyManager já soltou; RF5b)."""
        self._wanted.clear()
        self._granted = frozenset()

    def _order(self, names: set[str]) -> list[str]:
        return sorted(names, key=lambda n: (self._rank.get(n, len(self._rank)), n))

    def _group_mates(self, name: str) -> set[str]:
        return {m for group in self._groups if name in group for m in group if m != name}

    def resolve(self, events: Sequence[GestureEvent], t: float) -> list[GestureEvent]:
        """Aplica exclusividade e combinações; a saída preserva E3 por gesto."""
        upstream: dict[tuple[str, EventKind], GestureEvent] = {}
        pulses: list[GestureEvent] = []
        values: list[GestureEvent] = []
        for event in events:
            if event.kind is EventKind.START:
                self._wanted.add(event.gesture)
            elif event.kind is EventKind.END:
                self._wanted.discard(event.gesture)
            elif event.kind is EventKind.PULSE:
                pulses.append(event)
            else:
                values.append(event)
            upstream[(event.gesture, event.kind)] = event

        selected = self._select(self._wanted | {e.gesture for e in pulses})
        new_granted = frozenset(selected & self._wanted)

        out: list[GestureEvent] = []
        for name in self._order(set(self._granted - new_granted)):
            out.append(upstream.get((name, EventKind.END)) or GestureEvent(name, EventKind.END, t))
        for name in self._order(set(new_granted - self._granted)):
            out.append(
                upstream.get((name, EventKind.START)) or GestureEvent(name, EventKind.START, t)
            )
        out.extend(e for e in pulses if e.gesture in selected)
        out.extend(values)
        self._granted = new_granted
        return out

    def _select(self, candidates: set[str]) -> set[str]:
        """Escolhe, entre os candidatos do frame, os que passam (combos e grupos)."""
        selected: set[str] = set()
        blocked: set[str] = set()

        def free(name: str) -> bool:
            return name not in blocked and not (self._group_mates(name) & selected)

        # Passo 1: combos ativos suprimem os membros, mesmo que estes sejam mais prioritários.
        for name in self._order({n for n in candidates & self._wanted if n in self._combos}):
            if free(name):
                selected.add(name)
                blocked |= self._combos[name]
        # Passo 2: os demais, por prioridade; em grupo exclusivo vale o primeiro.
        for name in self._order(candidates):
            if name not in selected and free(name):
                selected.add(name)
        return selected
