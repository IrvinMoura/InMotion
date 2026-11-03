"""Replay determinístico do pipeline sobre gravações (spec 011, RF3/RF4, CA2).

Núcleo puro: sem arquivo e sem relógio do sistema. O "relógio" do backend é o ``t`` da
própria gravação, então duas execuções sobre a mesma gravação dão o mesmo resultado (D1).
Quem carrega o ``.npz`` é a ferramenta (``tools/replay.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from inmotion.gestures.base import GestureEvent
from inmotion.input.backend import InputBackend, KeyName
from inmotion.pipeline import Pipeline, StepResult
from inmotion.recording import Recording
from inmotion.safety import SafetyManager

InputKind = Literal["press", "release"]
InputRecord = tuple[float, InputKind, KeyName]


class RecordingInputBackend:
    """``InputBackend`` que só registra ``(t, "press"|"release", tecla)``; não envia nada.

    ``t`` vem de ``clock``, que no replay devolve o timestamp do frame em processamento.
    """

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self.records: list[InputRecord] = []

    def press(self, key: KeyName) -> None:
        self.records.append((self._clock(), "press", key))

    def release(self, key: KeyName) -> None:
        self.records.append((self._clock(), "release", key))


@dataclass(frozen=True)
class ReplayResult:
    """Saída de um replay.

    ``events`` contém os eventos liberados pelo árbitro em cada frame.
    ``snapshots`` só existe se pedido.
    """

    events: tuple[GestureEvent, ...]
    inputs: tuple[InputRecord, ...]
    snapshots: tuple[StepResult, ...] | None


def run_replay(
    recording: Recording,
    build_pipeline: Callable[[InputBackend], tuple[Pipeline, SafetyManager]],
    keep_snapshots: bool = False,
) -> ReplayResult:
    """Roda ``pipeline.step(pose, t)`` para cada frame da gravação, com os ``t`` originais.

    ``build_pipeline`` recebe o backend de registro e devolve ``(pipeline, safety)``: o
    ``Pipeline`` não expõe o ``SafetyManager``, e o replay precisa dele para envolver o
    laço em ``safety.guard()``. Assim o replay também passa por ``release_all()`` ao
    terminar (ou ao falhar), e as soltadas finais aparecem em ``inputs``.
    """
    now = [0.0]
    backend = RecordingInputBackend(lambda: now[0])
    pipeline, safety = build_pipeline(backend)
    snapshots: list[StepResult] = []
    events: list[GestureEvent] = []
    with safety.guard():
        for t, pose in recording.poses():
            now[0] = t
            result = pipeline.step(pose, t)
            events.extend(result.events)
            if keep_snapshots:
                snapshots.append(result)
    return ReplayResult(
        events=tuple(events),
        inputs=tuple(backend.records),
        snapshots=tuple(snapshots) if keep_snapshots else None,
    )
