"""Adaptador do MediaPipe Tasks ``PoseLandmarker`` (modo VIDEO). Único módulo com ``mediapipe``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from inmotion.types import PoseFrame

Variant = Literal["lite", "full", "heavy"]


@dataclass(frozen=True)
class PoseConfig:
    """Configuração do estimador (stub da T1: ainda sem validação)."""

    model_dir: Path = Path("models")
    variant: Variant = "full"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    @property
    def model_path(self) -> Path:
        raise NotImplementedError("spec 003 T3")


class ModelNotFoundError(FileNotFoundError):
    """Modelo ``.task`` ausente; a mensagem traz o caminho e o comando de download."""


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = ()


class MediaPipePoseEstimator:
    """Satisfaz o port ``PoseEstimator`` (stub da T1)."""

    def __init__(
        self,
        config: PoseConfig,
        landmarker_factory: Callable[[PoseConfig], Any] | None = None,
    ) -> None:
        raise NotImplementedError("spec 003 T3")

    def estimate(self, image_bgr: NDArray[np.uint8], t: float) -> PoseFrame | None:
        raise NotImplementedError("spec 003 T3")

    def close(self) -> None:
        raise NotImplementedError("spec 003 T3")

    def __enter__(self) -> MediaPipePoseEstimator:
        raise NotImplementedError("spec 003 T3")

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError("spec 003 T3")
