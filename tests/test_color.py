"""Testes da conversão BGR para RGB (spec 003, CA1)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from inmotion.vision.color import bgr_to_rgb

images = arrays(
    dtype=np.uint8,
    shape=st.tuples(st.integers(1, 6), st.integers(1, 6), st.just(3)),
)


@given(images)
def test_bgr_to_rgb_is_involution(img: np.ndarray) -> None:  # type: ignore[type-arg]
    """CA1 / A1: P² = I, logo aplicar duas vezes devolve a imagem."""
    np.testing.assert_array_equal(bgr_to_rgb(bgr_to_rgb(img)), img)


@given(images)
def test_bgr_to_rgb_matches_channel_reversal(img: np.ndarray) -> None:  # type: ignore[type-arg]
    """CA1: igual a inverter o eixo dos canais."""
    np.testing.assert_array_equal(bgr_to_rgb(img), img[..., ::-1])


def test_bgr_to_rgb_swaps_channels_and_is_contiguous() -> None:
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[..., 0] = 10  # B
    img[..., 2] = 30  # R
    out = bgr_to_rgb(img)
    assert out[0, 0].tolist() == [30, 0, 10]
    assert out.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
    ],
)
def test_bgr_to_rgb_rejects_invalid_input(bad: np.ndarray) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="uint8"):
        bgr_to_rgb(bad)
