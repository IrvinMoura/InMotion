"""Tipos de dados do pipeline: pose, landmark e pacote de frame.

Formato da pose descrito no ADR 0005 (arrays numpy imutáveis). Núcleo puro.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

N_LANDMARKS = 33


@dataclass(frozen=True)
class Landmark:
    """Um ponto da pose: x, y normalizados em [0, 1], z relativo e visibilidade."""

    x: float
    y: float
    z: float
    visibility: float


def _readonly_float64(name: str, value: object, cols: int) -> NDArray[np.float64]:
    """Copia para float64 com forma (33, cols) e marca como somente leitura."""
    try:
        arr = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} deve ser numérico") from exc
    if arr.shape != (N_LANDMARKS, cols):
        raise ValueError(f"{name} deve ter forma ({N_LANDMARKS}, {cols}), veio {arr.shape}")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class PoseFrame:
    """Uma pose num instante. Arrays validados no ``__post_init__`` e somente leitura.

    ``landmarks`` tem forma (33, 5): x, y normalizados; z; visibility; presence.
    ``world`` é opcional, forma (33, 4): x, y, z em metros; visibility.
    """

    t: float
    image_size: tuple[int, int]  # (W, H) em pixels
    landmarks: NDArray[np.float64]
    world: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.t):
            raise ValueError(f"t deve ser finito, veio {self.t}")
        size = self.image_size
        if len(size) != 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size
        ):
            raise ValueError(f"image_size deve ser (W, H) de inteiros > 0, veio {size}")
        object.__setattr__(self, "landmarks", _readonly_float64("landmarks", self.landmarks, 5))
        if self.world is not None:
            object.__setattr__(self, "world", _readonly_float64("world", self.world, 4))

    def landmark(self, index: int) -> Landmark:
        """Devolve o landmark ``index`` (sem a presence). Fora de [0, 33): IndexError."""
        if not 0 <= index < N_LANDMARKS:
            raise IndexError(f"índice de landmark fora de [0, {N_LANDMARKS}): {index}")
        row = self.landmarks[index]
        return Landmark(float(row[0]), float(row[1]), float(row[2]), float(row[3]))

    def pixels(self) -> NDArray[np.float64]:
        """A5: p = diag(W, H)·n, aplicado às colunas (x, y). Forma (33, 2)."""
        w, h = self.image_size
        scale = np.array([w, h], dtype=np.float64)
        out: NDArray[np.float64] = self.landmarks[:, :2] * scale
        return out


@dataclass(frozen=True)
class FramePacket:
    """Frame da câmera: instante, contador crescente e imagem BGR (H, W, 3) uint8."""

    t: float
    index: int
    image: NDArray[np.uint8]

    def __post_init__(self) -> None:
        if not math.isfinite(self.t):
            raise ValueError(f"t deve ser finito, veio {self.t}")
        if self.index < 0:
            raise ValueError(f"index deve ser >= 0, veio {self.index}")
        img = self.image
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"image deve ser uint8 (H, W, 3), veio {img.dtype} {img.shape}")
        # View somente leitura: não copia o frame (custo por loop).
        view = img.view()
        view.setflags(write=False)
        object.__setattr__(self, "image", view)
