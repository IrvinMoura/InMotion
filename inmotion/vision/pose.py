"""Adaptador do MediaPipe Tasks ``PoseLandmarker`` em modo VIDEO (ADR 0002).

Único módulo do projeto que importa ``mediapipe``. Traduz o ``PoseLandmarkerResult`` para
``PoseFrame`` (ADR 0005) e garante timestamps em ms estritamente crescentes.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from inmotion.types import N_LANDMARKS, PoseFrame
from inmotion.vision.color import bgr_to_rgb

logger = logging.getLogger(__name__)

Variant = Literal["lite", "full", "heavy"]
_VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
_CONFIDENCE_FIELDS: tuple[str, ...] = (
    "min_detection_confidence",
    "min_presence_confidence",
    "min_tracking_confidence",
)

# Intervalo mínimo (em segundos de ``t``) entre avisos de timestamp corrigido.
_WARN_INTERVAL_S = 1.0


@dataclass(frozen=True)
class PoseConfig:
    """Configuração do estimador. Confianças em [0, 1]; variante lite/full/heavy."""

    model_dir: Path = Path("models")
    variant: Variant = "full"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.variant not in _VARIANTS:
            raise ValueError(f"variant deve ser um de {_VARIANTS}, veio {self.variant!r}")
        for name in _CONFIDENCE_FIELDS:
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} deve estar em [0, 1], veio {value}")

    @property
    def model_path(self) -> Path:
        """Caminho do modelo: ``<model_dir>/pose_landmarker_<variant>.task``."""
        return self.model_dir / f"pose_landmarker_{self.variant}.task"


class ModelNotFoundError(FileNotFoundError):
    """Modelo ``.task`` ausente; a mensagem traz o caminho e o comando de download."""


# Pares de índices do esqueleto (topologia de 33 pontos do BlazePose), usados no overlay (009).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
)  # fmt: skip


def _default_factory(config: PoseConfig) -> Any:
    """Cria o ``PoseLandmarker`` real em modo VIDEO (import tardio do mediapipe)."""
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.min_detection_confidence,
        min_pose_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def _to_array(points: Any, attrs: tuple[str, ...]) -> NDArray[np.float64]:
    """Converte uma lista de landmarks do mediapipe em array (N, len(attrs))."""
    return np.array([[getattr(p, a) for a in attrs] for p in points], dtype=np.float64)


class MediaPipePoseEstimator:
    """Satisfaz o port ``PoseEstimator``: BGR -> A1 -> PoseLandmarker (VIDEO) -> PoseFrame."""

    def __init__(
        self,
        config: PoseConfig,
        landmarker_factory: Callable[[PoseConfig], Any] | None = None,
    ) -> None:
        if landmarker_factory is None:
            if not config.model_path.is_file():
                raise ModelNotFoundError(
                    f"Modelo não encontrado: {config.model_path}. Baixe com: "
                    f"python tools/download_model.py --variant {config.variant}"
                )
            landmarker_factory = _default_factory
        self._config = config
        self._landmarker: Any = landmarker_factory(config)
        self._last_ts_ms: int | None = None
        self._last_warn_t: float | None = None

    def _next_timestamp(self, t: float) -> int:
        """RF2: ``ts_ms = round(t·1000)``; se não crescer, usa ``último + 1`` (e avisa)."""
        ts_ms = round(t * 1000)
        last = self._last_ts_ms
        if last is not None and ts_ms <= last:
            ts_ms = last + 1
            if self._last_warn_t is None or t - self._last_warn_t >= _WARN_INTERVAL_S:
                self._last_warn_t = t
                logger.warning("Timestamp não crescente (t=%.3f s); usando %d ms", t, ts_ms)
        self._last_ts_ms = ts_ms
        return ts_ms

    def estimate(self, image_bgr: NDArray[np.uint8], t: float) -> PoseFrame | None:
        """Estima a pose da primeira pessoa; ``None`` se ninguém for detectado."""
        import mediapipe as mp

        rgb = bgr_to_rgb(image_bgr)
        height, width = rgb.shape[:2]
        ts_ms = self._next_timestamp(t)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
        )
        if not result.pose_landmarks:
            return None
        landmarks = _to_array(result.pose_landmarks[0], ("x", "y", "z", "visibility", "presence"))
        if landmarks.shape[0] != N_LANDMARKS:
            return None
        world: NDArray[np.float64] | None = None
        if result.pose_world_landmarks:
            world = _to_array(result.pose_world_landmarks[0], ("x", "y", "z", "visibility"))
        return PoseFrame(t=t, image_size=(width, height), landmarks=landmarks, world=world)

    def close(self) -> None:
        """Libera o landmarker (idempotente)."""
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            landmarker.close()

    def __enter__(self) -> MediaPipePoseEstimator:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
