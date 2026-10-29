"""Testes das matrizes homogêneas (A4, A5) e das propriedades G4, G7."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.vision.geometry import (
    apply_h,
    from_pixels,
    mirror_matrix,
    rotation_matrix,
    scale_matrix,
    to_pixels,
    translation_matrix,
)

PixelFn = Callable[[np.ndarray, int, int], np.ndarray]

coord = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
angle = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
pos = st.floats(min_value=1e-2, max_value=1e2, allow_nan=False, allow_infinity=False)
points = st.lists(st.tuples(coord, coord), min_size=1, max_size=8).map(
    lambda xs: np.array(xs, dtype=np.float64)
)


@given(width=st.floats(min_value=1.0, max_value=4096.0), pts=points)
def test_mirror_is_involution(width: float, pts: np.ndarray) -> None:
    f = mirror_matrix(width)
    np.testing.assert_allclose(f @ f, np.eye(3), atol=1e-9)
    out = apply_h(f, pts)
    np.testing.assert_allclose(out[:, 0], width - pts[:, 0], atol=1e-9)
    np.testing.assert_allclose(out[:, 1], pts[:, 1], atol=1e-9)
    np.testing.assert_allclose(apply_h(f, out), pts, atol=1e-9)


@given(
    w=st.integers(min_value=1, max_value=4096),
    h=st.integers(min_value=1, max_value=4096),
    pts=points,
)
def test_to_pixels_roundtrip(w: int, h: int, pts: np.ndarray) -> None:
    px = to_pixels(pts, w, h)
    np.testing.assert_allclose(px, pts * np.array([w, h]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(from_pixels(px, w, h), pts, rtol=1e-9, atol=1e-9)


@given(a=coord, b=coord, c=coord, d=coord, al=angle, be=angle, k=pos)
def test_matrix_composition(
    a: float, b: float, c: float, d: float, al: float, be: float, k: float
) -> None:
    np.testing.assert_allclose(
        translation_matrix(a, b) @ translation_matrix(c, d),
        translation_matrix(a + c, b + d),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rotation_matrix(al) @ rotation_matrix(be), rotation_matrix(al + be), atol=1e-9
    )
    r = rotation_matrix(al)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert math.isclose(float(np.linalg.det(r)), 1.0, abs_tol=1e-12)
    np.testing.assert_allclose(scale_matrix(k) @ scale_matrix(1 / k), np.eye(3), atol=1e-9)


def test_rotation_is_counterclockwise() -> None:
    out = apply_h(rotation_matrix(math.pi / 2), np.array([[1.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.0, 1.0]], atol=1e-12)


def test_translation_and_scale_apply() -> None:
    pts = np.array([[1.0, 2.0]])
    np.testing.assert_allclose(apply_h(translation_matrix(3, -1), pts), [[4.0, 1.0]])
    np.testing.assert_allclose(apply_h(scale_matrix(2), pts), [[2.0, 4.0]])


@pytest.mark.parametrize("k", [0.0, -1.0, math.nan, math.inf])
def test_scale_rejects_invalid(k: float) -> None:
    with pytest.raises(ValueError, match="k deve"):
        scale_matrix(k)


@pytest.mark.parametrize("w", [0.0, -3.0])
def test_mirror_rejects_invalid_width(w: float) -> None:
    with pytest.raises(ValueError, match="width deve"):
        mirror_matrix(w)


@pytest.mark.parametrize("fn", [to_pixels, from_pixels])
@pytest.mark.parametrize(("w", "h"), [(0, 10), (10, 0), (-1, 10)])
def test_pixels_reject_invalid_size(fn: PixelFn, w: int, h: int) -> None:
    with pytest.raises(ValueError, match="devem ser > 0"):
        fn(np.zeros((2, 2)), w, h)


@pytest.mark.parametrize("fn", [to_pixels, from_pixels])
def test_pixels_reject_bad_shape(fn: PixelFn) -> None:
    with pytest.raises(ValueError, match="forma"):
        fn(np.zeros((2, 3)), 10, 10)


def test_apply_h_rejects_bad_matrix_and_points() -> None:
    with pytest.raises(ValueError, match="forma"):
        apply_h(np.eye(2), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="forma"):
        apply_h(np.eye(3), np.zeros((1, 3)))
