"""Ports do pipeline (Protocols). Núcleo puro."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from inmotion.types import FramePacket, PoseFrame


class Clock(Protocol):
    """Fonte de tempo injetável: segundos, monotônico."""

    def now(self) -> float: ...


class FrameSource(Protocol):
    """Origem de frames (câmera ou replay)."""

    def read(self, timeout_s: float) -> FramePacket | None:
        """Devolve um frame NOVO ou None se o tempo esgotar."""
        ...

    def close(self) -> None: ...


class PoseEstimator(Protocol):
    """Estimador de pose: imagem BGR -> PoseFrame, ou None se não achar ninguém."""

    def estimate(self, image_bgr: NDArray[np.uint8], t: float) -> PoseFrame | None: ...

    def close(self) -> None: ...
