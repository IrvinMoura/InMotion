"""Adaptador de arquivo das gravações: acumula frames e lê/escreve ``.npz`` (ADR 0008)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from inmotion.recording import LabelInterval, Recording, RecordingMeta
from inmotion.types import PoseFrame


class Recorder:
    """Acumula frames em memória e salva um ``.npz`` comprimido."""

    def __init__(self, meta: RecordingMeta, image_size: tuple[int, int]) -> None:
        raise NotImplementedError

    def add(self, t: float, pose: PoseFrame | None) -> None:
        raise NotImplementedError

    def save(self, path: Path, labels: Sequence[LabelInterval]) -> Path:
        raise NotImplementedError


def load_recording(path: Path) -> Recording:
    """Lê um ``.npz``; versão incompatível ou arquivo inválido → RecordingFormatError."""
    raise NotImplementedError
