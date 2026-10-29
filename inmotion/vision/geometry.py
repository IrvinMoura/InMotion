"""Geometria homogênea 2D (núcleo puro): translação, rotação, escala, espelho e pixels.

Convenção: pontos são linhas ``(N, 2)``; a matriz age em coluna, ``p' = M·[x, y, 1]ᵀ``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from inmotion.types import PoseFrame
from inmotion.vision.landmarks import PoseLandmark

Matrix3: TypeAlias = NDArray[np.float64]  # (3, 3) homogênea
Points2: TypeAlias = NDArray[np.float64]  # (N, 2)


def translation_matrix(dx: float, dy: float) -> Matrix3:
    """A6 (componente T): translação ``[[1,0,dx],[0,1,dy],[0,0,1]]`` de ``M = S·R·T``.

    T(a)·T(b) = T(a+b). ``dx`` e ``dy`` devem ser finitos.
    """
    if not (math.isfinite(dx) and math.isfinite(dy)):
        raise ValueError(f"dx e dy devem ser finitos, recebido ({dx}, {dy})")
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_matrix(theta: float) -> Matrix3:
    """A6 (componente R): rotação anti-horária de ``M = S·R·T``, ``θ`` em radianos.

    ``R = [[cos θ, −sin θ, 0],[sin θ, cos θ, 0],[0,0,1]]``; R(α)·R(β) = R(α+β);
    R ortogonal, det 1. ``theta`` deve ser finito.
    """
    if not math.isfinite(theta):
        raise ValueError(f"theta deve ser finito, recebido {theta}")
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def scale_matrix(k: float) -> Matrix3:
    """A6 (componente S): escala uniforme ``S = diag(k, k, 1)`` de ``M = S·R·T``, k > 0 e finito."""
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


@dataclass(frozen=True)
class GeometryConfig:
    """Parâmetros da geometria: gate de visibilidade e largura mínima dos ombros (A6)."""

    visibility_threshold: float = 0.6  # [0, 1]
    min_shoulder_px: float = 10.0  # abaixo disso a normalização é degenerada

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.visibility_threshold) and 0.0 <= self.visibility_threshold <= 1.0
        ):
            raise ValueError(
                f"visibility_threshold deve estar em [0, 1], recebido {self.visibility_threshold}"
            )
        if not math.isfinite(self.min_shoulder_px) or self.min_shoulder_px <= 0:
            raise ValueError(
                f"min_shoulder_px deve ser finito e > 0, recebido {self.min_shoulder_px}"
            )


@dataclass(frozen=True)
class NormalizedSkeleton:
    """Esqueleto após ``M``: pontos (33, 2), a matriz ``M = S·R·T`` e a escala ``s`` em px."""

    points: Points2
    matrix: Matrix3
    shoulder_px: float


def normalize_skeleton(pose: PoseFrame, config: GeometryConfig) -> NormalizedSkeleton | None:
    """A6: normaliza o esqueleto em pixels (A5) com ``M = S·R·T``.

    - ``T`` leva o centro do quadril ``c`` à origem: ``T = T(−c)``.
    - ``R = R(−φ)`` com ``φ = atan2(v_y, v_x)``, ``v`` = ombro esquerdo − ombro direito
      (em px); após ``M``, ``v`` fica paralelo ao eixo x (G2).
    - ``S = S(1/s)`` com ``s = ‖v‖``; após ``M``, ``v ↦ (1, 0)``.

    Como ``M`` é uma semelhança (T, R, S) aplicada em pixels, que são isotrópicos, o
    resultado é invariante a translação, rotação no plano e escala da pessoa (G1).
    Devolve ``None`` se ombros ou quadris estiverem abaixo do limiar de visibilidade,
    se ``s < min_shoulder_px`` ou se algum ponto usado não for finito.
    """
    shoulders = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)
    hips = (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)
    if not is_visible(pose, (*shoulders, *hips), config.visibility_threshold):
        return None
    px = pose.pixels()
    left_sh, right_sh = px[PoseLandmark.LEFT_SHOULDER], px[PoseLandmark.RIGHT_SHOULDER]
    hip_center = (px[PoseLandmark.LEFT_HIP] + px[PoseLandmark.RIGHT_HIP]) / 2.0
    v = left_sh - right_sh
    s = float(np.hypot(v[0], v[1]))
    if not (np.isfinite(s) and np.all(np.isfinite(hip_center))) or s < config.min_shoulder_px:
        return None
    phi = math.atan2(float(v[1]), float(v[0]))
    m = (
        scale_matrix(1.0 / s)
        @ rotation_matrix(-phi)
        @ translation_matrix(-float(hip_center[0]), -float(hip_center[1]))
    )
    return NormalizedSkeleton(points=apply_h(m, px), matrix=m, shoulder_px=s)


def is_visible(pose: PoseFrame, indices: Sequence[int], threshold: float) -> bool:
    """Gate de visibilidade: todos os ``indices`` têm ``visibility >= threshold``."""
    vis = pose.landmarks[list(indices), 3]
    return bool(np.all(vis >= threshold))


def _check_points(points: Points2) -> Points2:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"points deve ter forma (N, 2), recebido {arr.shape}")
    return arr


def _check_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"width e height devem ser > 0, recebido {width}x{height}")
