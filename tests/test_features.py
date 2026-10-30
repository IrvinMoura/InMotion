"""Testes das features dos gestos de corpo (spec 008): alcance frontal e base R7."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.gestures.features import forward_reach
from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import PoseLandmark, Side
from tests.helpers import make_body

GEOMETRY = GeometryConfig()
TOL = 1e-9

coord = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
vec3 = st.tuples(coord, coord, coord)


def _with_arm_world(
    shoulder: tuple[float, float, float],
    elbow: tuple[float, float, float],
    wrist: tuple[float, float, float],
) -> PoseFrame:
    """Pose com o braço direito nos pontos dados (world); o quadril fica bem abaixo de tudo."""
    base = make_body()
    assert base.world is not None
    world = np.array(base.world)
    world[PoseLandmark.RIGHT_SHOULDER, :3] = shoulder
    world[PoseLandmark.RIGHT_ELBOW, :3] = elbow
    world[PoseLandmark.RIGHT_WRIST, :3] = wrist
    world[PoseLandmark.LEFT_HIP, :3] = (0.1, 10.0, 0.0)  # gate do pulso sempre aberto
    world[PoseLandmark.RIGHT_HIP, :3] = (-0.1, 10.0, 0.0)
    return PoseFrame(t=0.0, image_size=base.image_size, landmarks=base.landmarks, world=world)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@given(s=vec3, e=vec3, w=vec3)
def test_forward_reach_bounded_and_one_iff_straight_to_camera(
    s: tuple[float, float, float],
    e: tuple[float, float, float],
    w: tuple[float, float, float],
) -> None:
    """|r| ≤ 1 sempre; r = 1 quando o braço é reto ao longo de −z (a câmera)."""
    r = forward_reach(_with_arm_world(s, e, w), Side.RIGHT, GEOMETRY)
    if r is not None:
        assert abs(r) <= 1.0 + TOL
    # Braço reto apontando para a câmera: ombro, cotovelo e pulso colineares ao longo de −z.
    d = 0.5
    straight = _with_arm_world(s, (s[0], s[1], s[2] - d / 2), (s[0], s[1], s[2] - d))
    r_straight = forward_reach(straight, Side.RIGHT, GEOMETRY)
    assert r_straight is not None
    assert math.isclose(r_straight, 1.0, abs_tol=TOL)
