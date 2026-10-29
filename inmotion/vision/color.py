"""Conversão de cor (núcleo puro): BGR (OpenCV) para RGB (MediaPipe)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bgr_to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """A1: RGB = P·BGR por pixel, com ``P = [[0,0,1],[0,1,0],[1,0,0]]`` e ``P² = I``.

    ``P`` é a matriz de permutação que troca os canais B e R; como ``P² = I``, aplicar a
    conversão duas vezes devolve a imagem original. Implementada como inversão do eixo
    dos canais (equivalente a ``image @ P`` sem o custo do produto). Devolve uma cópia
    contígua (exigência do ``mp.Image``). Entrada: ``uint8`` de forma ``(H, W, 3)``.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image deve ser uint8 (H, W, 3), veio {image.dtype} {image.shape}")
    return np.ascontiguousarray(image[..., ::-1])
