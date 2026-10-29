"""Testes dos kernels gaussiano (A2) e de Sobel (A3); propriedades G5 e G6."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from inmotion.vision.kernels import (
    SOBEL_X,
    SOBEL_Y,
    gaussian_kernel_1d,
    gaussian_kernel_2d,
)

sigmas = st.floats(min_value=0.3, max_value=6.0, allow_nan=False, allow_infinity=False)


def convolve_valid(image: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convolução 2D de referência (modo valid, kernel espelhado), só para testes."""
    kh, kw = kernel.shape
    flipped = kernel[::-1, ::-1]
    h, w = image.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = float(np.sum(image[i : i + kh, j : j + kw] * flipped))
    return out


@given(sigma=sigmas)
def test_gaussian_kernel_sums_to_one_and_is_separable(sigma: float) -> None:
    g = gaussian_kernel_1d(sigma)
    k = gaussian_kernel_2d(sigma)
    assert math.isclose(float(g.sum()), 1.0, abs_tol=1e-12)
    assert math.isclose(float(k.sum()), 1.0, abs_tol=1e-12)
    assert g.shape == (2 * math.ceil(3 * sigma) + 1,)
    np.testing.assert_allclose(k, np.outer(g, g), atol=1e-15)


@given(sigma=sigmas, radius=st.integers(min_value=0, max_value=12))
def test_gaussian_kernel_symmetric(sigma: float, radius: int) -> None:
    g = gaussian_kernel_1d(sigma, radius)
    k = gaussian_kernel_2d(sigma, radius)
    assert g.shape == (2 * radius + 1,)
    np.testing.assert_array_equal(g, g[::-1])
    np.testing.assert_allclose(k, k.T, atol=1e-15)
    np.testing.assert_array_equal(k, k[::-1, ::-1])
    assert int(np.argmax(g)) == radius


@given(c=st.floats(min_value=-255, max_value=255, allow_nan=False))
def test_sobel_on_constant_image_is_zero(c: float) -> None:
    image = np.full((7, 9), c)
    np.testing.assert_allclose(convolve_valid(image, SOBEL_X), 0.0, atol=1e-9)
    np.testing.assert_allclose(convolve_valid(image, SOBEL_Y), 0.0, atol=1e-9)


def test_sobel_values_and_read_only() -> None:
    np.testing.assert_array_equal(SOBEL_X, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    np.testing.assert_array_equal(SOBEL_Y, SOBEL_X.T)
    assert not SOBEL_X.flags.writeable
    assert not SOBEL_Y.flags.writeable


def test_sobel_detects_vertical_edge() -> None:
    image = np.zeros((5, 6))
    image[:, 3:] = 1.0
    assert np.all(
        convolve_valid(image, SOBEL_X)[:, 1] == -4.0
    )  # convolução inverte o kernel: borda 0→1 dá -4
    np.testing.assert_allclose(convolve_valid(image, SOBEL_Y), 0.0)


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_gaussian_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        gaussian_kernel_1d(sigma)
    with pytest.raises(ValueError, match="sigma"):
        gaussian_kernel_2d(sigma)


def test_gaussian_rejects_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius"):
        gaussian_kernel_1d(1.0, -1)
    with pytest.raises(ValueError, match="radius"):
        gaussian_kernel_2d(1.0, -1)
