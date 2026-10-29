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


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@given(images)
def test_bgr_to_rgb_is_involution(img: np.ndarray) -> None:  # type: ignore[type-arg]
    """CA1 / A1: P² = I, logo aplicar duas vezes devolve a imagem."""
    np.testing.assert_array_equal(bgr_to_rgb(bgr_to_rgb(img)), img)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@given(images)
def test_bgr_to_rgb_matches_channel_reversal(img: np.ndarray) -> None:  # type: ignore[type-arg]
    """CA1: igual a inverter o eixo dos canais."""
    np.testing.assert_array_equal(bgr_to_rgb(img), img[..., ::-1])
