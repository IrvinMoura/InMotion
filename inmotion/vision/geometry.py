"""Geometria homogênea 2D (núcleo puro): translação, rotação, escala, espelho e pixels.

Convenção: pontos são linhas ``(N, 2)``; a matriz age em coluna, ``p' = M·[x, y, 1]ᵀ``.
"""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Matrix3: TypeAlias = NDArray[np.float64]  # (3, 3) homogênea
Points2: TypeAlias = NDArray[np.float64]  # (N, 2)


def translation_matrix(dx: float, dy: float) -> Matrix3:
    """Translação T(dx, dy): ``[[1,0,dx],[0,1,dy],[0,0,1]]``. T(a)·T(b) = T(a+b)."""
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix(theta: float) -> Matrix3:
    """Rotação anti-horária R(θ) em radianos. R(α)·R(β) = R(α+β); R ortogonal, det 1."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_matrix(k: float) -> Matrix3:
    """Escala uniforme S(k) = diag(k, k, 1), com k > 0 e finito."""
    if not math.isfinite(k) or k <= 0:
        raise ValueError(f"k deve ser finito e > 0, recebido {k}")
    return np.diag([k, k, 1.0]).astype(np.float64)


def mirror_matrix(width: float) -> Matrix3:
    """A4: espelho horizontal F, ``x ↦ W − x``; F = [[-1,0,W],[0,1,0],[0,0,1]], F² = I."""
    if not math.isfinite(width) or width <= 0:
        raise ValueError(f"width deve ser finito e > 0, recebido {width}")
    return np.array([[-1.0, 0.0, width], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_h(m: Matrix3, points: Points2) -> Points2:
    """Aplica ``m`` em coordenadas homogêneas e volta para 2D (divide por w)."""
    if m.shape != (3, 3):
        raise ValueError(f"m deve ter forma (3, 3), recebido {m.shape}")
    pts = _check_points(points)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = homogeneous @ m.T
    result: Points2 = out[:, :2] / out[:, 2:3]
    return result


def to_pixels(normalized: Points2, width: int, height: int) -> Points2:
    """A5: normalizado [0,1]² → pixels, ``p = diag(W, H, 1)·n``."""
    _check_size(width, height)
    return apply_h(np.diag([float(width), float(height), 1.0]), normalized)


def from_pixels(pixels: Points2, width: int, height: int) -> Points2:
    """Inversa de A5: pixels → normalizado, ``n = diag(1/W, 1/H, 1)·p``."""
    _check_size(width, height)
    return apply_h(np.diag([1.0 / width, 1.0 / height, 1.0]), pixels)


def _check_points(points: Points2) -> Points2:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points deve ter forma (N, 2), recebido {arr.shape}")
    return arr


def _check_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"width e height devem ser > 0, recebido {width}x{height}")
