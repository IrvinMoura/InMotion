"""Formato de gravação de sessões de landmarks rotuladas (spec 011, ADR 0008).

Núcleo puro: sem arquivo, sem relógio. A hora de criação (``created``) vem do chamador.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import NDArray

from inmotion.input.backend import KeyName
from inmotion.types import PoseFrame

SCHEMA_VERSION: Final = 1


@dataclass(frozen=True)
class RecordingMeta:
    """Metadados de uma sessão (pessoa anônima, condições de coleta, hash da config)."""

    schema_version: int
    created: str
    person: str
    distance_m: float | None
    lighting: Literal["clara", "escura", "outra"]
    camera: str
    config_hash: str


@dataclass(frozen=True)
class LabelInterval:
    """Intervalo [start, end] em que o gesto ``gesture`` estava sendo executado."""

    gesture: str
    start: float
    end: float


class RecordingFormatError(ValueError):
    """Arquivo de gravação inválido, corrompido ou de versão incompatível."""


@dataclass(frozen=True, eq=False)
class Recording:
    """Sessão gravada. Arrays com ``N`` frames; NaN onde não há pose."""

    meta: RecordingMeta
    image_size: tuple[int, int]
    t: NDArray[np.float64]
    landmarks: NDArray[np.float64]
    world: NDArray[np.float64]
    present: NDArray[np.bool_]
    labels: tuple[LabelInterval, ...]

    def __post_init__(self) -> None:
        raise NotImplementedError

    def poses(self) -> Iterator[tuple[float, PoseFrame | None]]:
        raise NotImplementedError

    @property
    def duration_s(self) -> float:
        raise NotImplementedError


class LabelTracker:
    """Transforma "teclas de rótulo seguradas por frame" em intervalos."""

    def __init__(self, key_to_gesture: Mapping[KeyName, str]) -> None:
        raise NotImplementedError

    def update(self, keys: frozenset[KeyName], t: float) -> None:
        raise NotImplementedError

    def close(self, t: float) -> tuple[LabelInterval, ...]:
        raise NotImplementedError
