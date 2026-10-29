"""Kernels de imagem (núcleo puro): gaussiano (A2) e Sobel (A3)."""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray


def gaussian_kernel_1d(sigma: float, radius: int | None = None) -> NDArray[np.float64]:
    """A2: ``g(i) ∝ e^{−i²/2σ²}``, ``i ∈ [−r, r]``, ``r = ⌈3σ⌉`` por padrão; soma 1."""
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma deve ser finito e > 0, recebido {sigma}")
    if radius is None:
        radius = math.ceil(3.0 * sigma)
    if radius < 0:
        raise ValueError(f"radius deve ser >= 0, recebido {radius}")
    i = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(i**2) / (2.0 * sigma**2))
    result: NDArray[np.float64] = g / g.sum()
    return result


def gaussian_kernel_2d(sigma: float, radius: int | None = None) -> NDArray[np.float64]:
    """A2: kernel separável ``K = g·gᵀ`` (soma 1)."""
    g = gaussian_kernel_1d(sigma, radius)
    result: NDArray[np.float64] = np.outer(g, g).astype(np.float64)
    return result


def _read_only(a: NDArray[np.float64]) -> NDArray[np.float64]:
    a.setflags(write=False)
    return a


# A3: gradiente de Sobel (derivada [-1, 0, 1] x suavização [1, 2, 1]).
SOBEL_X: Final[NDArray[np.float64]] = _read_only(
    np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
)
SOBEL_Y: Final[NDArray[np.float64]] = _read_only(SOBEL_X.T.copy())
