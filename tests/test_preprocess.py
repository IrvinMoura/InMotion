"""Testes do Preprocessor (A2 gaussiano separável, A3 Sobel de debug)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from inmotion.vision.kernels import SOBEL_X, SOBEL_Y, gaussian_kernel_2d
from inmotion.vision.preprocess import PreprocessConfig, Preprocessor
from tests.test_kernels import convolve_valid


def _image(h: int = 48, w: int = 64, seed: int = 0) -> NDArray[np.uint8]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_blur_keeps_shape_and_dtype() -> None:
    img = _image()
    out = Preprocessor(PreprocessConfig(blur_sigma=1.5)).apply(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_blur_disabled_is_identity() -> None:
    img = _image()
    out = Preprocessor(PreprocessConfig(blur_sigma=None)).apply(img)
    assert np.array_equal(out, img)


def test_blur_matches_numpy_reference_on_interior() -> None:
    """A2: sepFilter2D(g, g) == convolução 2D por K = g·gᵀ (patch interno, até arredondar)."""
    sigma = 1.2
    img = _image()
    out = Preprocessor(PreprocessConfig(blur_sigma=sigma)).apply(img)
    kernel = gaussian_kernel_2d(sigma)
    r = kernel.shape[0] // 2
    for ch in range(3):
        ref = convolve_valid(img[:, :, ch].astype(np.float64), kernel)
        np.testing.assert_allclose(out[r:-r, r:-r, ch], ref, atol=0.51)


def test_blur_constant_image_is_unchanged() -> None:
    img = np.full((20, 30, 3), 77, dtype=np.uint8)
    assert np.array_equal(Preprocessor(PreprocessConfig(blur_sigma=2.0)).apply(img), img)


def test_sobel_view_shape() -> None:
    img = _image()
    view = Preprocessor(PreprocessConfig()).sobel_view(img)
    assert view.shape == img.shape
    assert view.dtype == np.uint8


def test_sobel_view_constant_is_zero_and_edge_is_bright() -> None:
    pre = Preprocessor(PreprocessConfig())
    flat = np.full((20, 30, 3), 90, dtype=np.uint8)
    assert not pre.sobel_view(flat).any()
    edge = np.zeros((20, 30, 3), dtype=np.uint8)
    edge[:, 15:] = 255
    view = pre.sobel_view(edge)
    assert view[10, 14, 0] == 255
    assert view[10, 5, 0] == 0


def test_sobel_view_matches_numpy_reference() -> None:
    """A3: ‖∇I‖ do patch interno bate com a convolução NumPy (normalizada pelo máximo)."""
    img = _image(seed=3)
    view = Preprocessor(PreprocessConfig()).sobel_view(img)
    gray = np.round(0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0])
    gx = convolve_valid(gray, SOBEL_X)
    gy = convolve_valid(gray, SOBEL_Y)
    mag = np.hypot(gx, gy)
    out = view[1:-1, 1:-1, 0].astype(np.float64)
    # o máximo global (com bordas) pode diferir do interno: compara a menos de escala
    scale = out.max() / mag.max()
    np.testing.assert_allclose(out, mag * scale, atol=3.0)
    assert np.array_equal(view[:, :, 0], view[:, :, 1])
    assert np.array_equal(view[:, :, 0], view[:, :, 2])


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="blur_sigma"):
        PreprocessConfig(blur_sigma=0.0)
    with pytest.raises(ValueError, match="blur_sigma"):
        PreprocessConfig(blur_sigma=float("nan"))
