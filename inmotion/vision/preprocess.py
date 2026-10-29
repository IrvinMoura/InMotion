"""Pré-processamento de imagem (adaptador): suavização gaussiana (A2) e Sobel de debug (A3)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.vision.kernels import SOBEL_X, SOBEL_Y, gaussian_kernel_1d


@dataclass(frozen=True)
class PreprocessConfig:
    """``blur_sigma`` em pixels (``None`` desliga a suavização); ``sobel_debug`` liga a vista."""

    blur_sigma: float | None = 1.0
    sobel_debug: bool = False

    def __post_init__(self) -> None:
        sigma = self.blur_sigma
        if sigma is not None and (not math.isfinite(sigma) or sigma <= 0):
            raise ValueError(f"blur_sigma deve ser None ou finito e > 0, recebido {sigma}")


class Preprocessor:
    """Aplica o blur gaussiano e gera a vista de gradiente de Sobel."""

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config
        sigma = config.blur_sigma
        self._gauss: NDArray[np.float64] | None = (
            None if sigma is None else gaussian_kernel_1d(sigma)
        )

    def apply(self, image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """A2: ``I' = K * I com K = g·gᵀ`` via ``cv2.sepFilter2D``; sem sigma devolve a imagem."""
        if self._gauss is None:
            return image_bgr
        return np.asarray(cv2.sepFilter2D(image_bgr, -1, self._gauss, self._gauss), dtype=np.uint8)

    def sobel_view(self, image_bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """A3: ``‖∇I‖ = √(Gx² + Gy²)`` normalizado para uint8 de 3 canais."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.filter2D(gray, -1, SOBEL_X.astype(np.float32))
        gy = cv2.filter2D(gray, -1, SOBEL_Y.astype(np.float32))
        magnitude = np.hypot(gx, gy)
        scaled: NDArray[np.uint8]
        peak = float(magnitude.max())
        if peak <= 0.0:
            scaled = np.zeros(gray.shape, dtype=np.uint8)
        else:
            scaled = np.rint(magnitude * (255.0 / peak)).astype(np.uint8)
        return np.asarray(cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
