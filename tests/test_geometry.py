"""Testes das matrizes homogêneas (A4, A5) e das propriedades G4, G7."""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from inmotion.vision.geometry import (
    GeometryConfig,
    NormalizedSkeleton,
    apply_h,
    distance,
    from_pixels,
    joint_angle,
    mirror_matrix,
    normalize_skeleton,
    rotation_matrix,
    scale_matrix,
    to_pixels,
    translation_matrix,
    trunk_lean,
)
from inmotion.vision.landmarks import PoseLandmark
from tests.fakes import make_pose

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


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_translation_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="dx e dy"):
        translation_matrix(bad, 0.0)
    with pytest.raises(ValueError, match="dx e dy"):
        translation_matrix(0.0, bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_rotation_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValueError, match="theta"):
        rotation_matrix(bad)


# ---------------------------------------------------------------- T3: A6 normalize_skeleton

CFG = GeometryConfig()
L_SH, R_SH = int(PoseLandmark.LEFT_SHOULDER), int(PoseLandmark.RIGHT_SHOULDER)
L_HIP, R_HIP = int(PoseLandmark.LEFT_HIP), int(PoseLandmark.RIGHT_HIP)
BIG = (4000, 4000)  # imagem grande: a semelhança aleatória não precisa caber nela


def _torso(
    ls: tuple[float, float],
    rs: tuple[float, float],
    lh: tuple[float, float],
    rh: tuple[float, float],
) -> dict[int, tuple[float, float]]:
    return {L_SH: ls, R_SH: rs, L_HIP: lh, R_HIP: rh}


def test_normalized_hips_at_origin_shoulders_unit_horizontal() -> None:
    """G2: centro do quadril na origem e (ombro esq − ombro dir) = (1, 0)."""
    pose = make_pose(_torso((400, 150), (240, 170), (380, 330), (260, 350)))
    sk = normalize_skeleton(pose, CFG)
    assert sk is not None
    pts = sk.points
    assert pts.shape == (33, 2)
    np.testing.assert_allclose((pts[L_HIP] + pts[R_HIP]) / 2, [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pts[L_SH] - pts[R_SH], [1.0, 0.0], atol=1e-12)
    assert sk.shoulder_px == pytest.approx(math.hypot(160, -20))
    np.testing.assert_allclose(apply_h(sk.matrix, pose.pixels()), pts, atol=1e-12)


def test_degenerate_shoulders_return_none() -> None:
    """CA3: ombros coincidentes ou estreitos, ou ombro/quadril invisível -> None."""
    same = make_pose(_torso((300, 150), (300, 150), (380, 330), (260, 350)))
    narrow = make_pose(_torso((305, 150), (300, 150), (380, 330), (260, 350)))
    hidden = make_pose(
        _torso((400, 150), (240, 170), (380, 330), (260, 350)), visibility={R_HIP: 0.2}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert normalize_skeleton(same, CFG) is None
        assert normalize_skeleton(narrow, CFG) is None
        assert normalize_skeleton(hidden, CFG) is None


def test_geometry_config_validation() -> None:
    with pytest.raises(ValueError, match="visibility_threshold"):
        GeometryConfig(visibility_threshold=1.5)
    with pytest.raises(ValueError, match="min_shoulder_px"):
        GeometryConfig(min_shoulder_px=0.0)


_torso_xy = st.floats(min_value=100.0, max_value=300.0, allow_nan=False, allow_infinity=False)


@given(
    st.tuples(
        _torso_xy, _torso_xy, _torso_xy, _torso_xy, _torso_xy, _torso_xy, _torso_xy, _torso_xy
    ),
    st.floats(min_value=-math.pi, max_value=math.pi),
    st.floats(min_value=0.2, max_value=5.0),
    st.floats(min_value=-500.0, max_value=500.0),
    st.floats(min_value=-500.0, max_value=500.0),
)
def test_normalize_is_similarity_invariant(
    c: tuple[float, ...], theta: float, k: float, dx: float, dy: float
) -> None:
    """G1: normalize(Sim(p)) == normalize(p) para Sim = T(d)·S(k)·R(θ) (sem reflexão)."""
    ls, rs = (c[0], c[1]), (c[2], c[3])
    # ombros largos o bastante, antes e depois da semelhança (min_shoulder_px = 10)
    assume(min(1.0, k) * math.hypot(ls[0] - rs[0], ls[1] - rs[1]) >= 12.0)
    base = _torso(ls, rs, (c[4], c[5]), (c[6], c[7]))
    pose = make_pose(base, size=BIG)
    sim = translation_matrix(dx, dy) @ scale_matrix(k) @ rotation_matrix(theta)
    moved = apply_h(sim, pose.pixels())
    pose2 = make_pose({i: (float(x), float(y)) for i, (x, y) in enumerate(moved)}, size=BIG)
    a, b = normalize_skeleton(pose, CFG), normalize_skeleton(pose2, CFG)
    assert a is not None
    assert b is not None
    np.testing.assert_allclose(b.points, a.points, atol=1e-9 * max(1.0, k) * 10)
    assert b.shoulder_px == pytest.approx(k * a.shoulder_px, rel=1e-9)


# ---------------------------------------------------------------- T4: A7-A9


def test_joint_angle_known_values() -> None:
    """A7: 90°, 180° e 0° (2D e 3D)."""
    b = np.array([0.0, 0.0])
    assert joint_angle(np.array([1.0, 0.0]), b, np.array([0.0, 1.0])) == pytest.approx(math.pi / 2)
    assert joint_angle(np.array([1.0, 0.0]), b, np.array([-2.0, 0.0])) == pytest.approx(math.pi)
    assert joint_angle(np.array([1.0, 0.0]), b, np.array([3.0, 0.0])) == pytest.approx(
        0.0, abs=1e-7
    )
    b3 = np.array([1.0, 1.0, 1.0])
    a3, c3 = b3 + np.array([1.0, 0.0, 0.0]), b3 + np.array([0.0, 0.0, 5.0])
    assert joint_angle(a3, b3, c3) == pytest.approx(math.pi / 2)


def test_joint_angle_zero_vector_raises() -> None:
    """Vetor nulo (a == b ou c == b) é indefinido: ValueError."""
    p = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="nulo"):
        joint_angle(p, p, np.array([3.0, 4.0]))
    with pytest.raises(ValueError, match="nulo"):
        joint_angle(np.array([3.0, 4.0]), p, p)


def test_joint_angle_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="forma"):
        joint_angle(np.zeros(2), np.zeros(3), np.zeros(2))


_vec3 = st.tuples(coord, coord, coord).map(lambda t: np.array(t, dtype=np.float64))


@given(_vec3, _vec3, _vec3)
def test_joint_angle_symmetric_and_bounded(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
    """A7: θ(a,b,c) = θ(c,b,a) e θ ∈ [0, π]."""
    assume(np.linalg.norm(a - b) > 1e-3 and np.linalg.norm(c - b) > 1e-3)
    theta = joint_angle(a, b, c)
    assert 0.0 <= theta <= math.pi
    assert theta == pytest.approx(joint_angle(c, b, a), abs=1e-12)


def test_distance_known_values() -> None:
    """A8: norma euclidiana, simétrica."""
    a, b = np.array([0.0, 0.0, 0.0]), np.array([1.0, 2.0, 2.0])
    assert distance(a, b) == pytest.approx(3.0)
    assert distance(b, a) == pytest.approx(3.0)
    assert distance(a, a) == 0.0


def _lean_skeleton(shoulder_mid_x: float) -> NormalizedSkeleton:
    """Pessoa de frente com os ombros deslocados em x (px) em relação ao quadril."""
    pose = make_pose(
        _torso(
            (shoulder_mid_x + 80, 150),
            (shoulder_mid_x - 80, 150),
            (380, 350),
            (260, 350),
        )
    )
    sk = normalize_skeleton(pose, CFG)
    assert sk is not None
    return sk


def test_trunk_lean_sign() -> None:
    """A9: ereto = 0; ombros à direita da imagem > 0; à esquerda < 0; simétrico."""
    upright = trunk_lean(_lean_skeleton(320))
    right = trunk_lean(_lean_skeleton(420))
    left = trunk_lean(_lean_skeleton(220))
    assert upright == pytest.approx(0.0, abs=1e-12)
    assert right > 0.0 > left
    assert right == pytest.approx(-left)
    assert right == pytest.approx(math.atan2(100, 200))


def test_trunk_lean_includes_rigid_tilt() -> None:
    """Rotação rígida do tronco no plano da imagem entra na inclinação (A9 em px)."""
    base = make_pose(_torso((400, 150), (240, 150), (380, 350), (260, 350)))
    tilt = rotation_matrix(0.3)
    moved = apply_h(tilt, base.pixels())
    pose = make_pose({i: (float(x), float(y)) for i, (x, y) in enumerate(moved)}, size=(640, 480))
    sk = normalize_skeleton(pose, CFG)
    sk0 = normalize_skeleton(base, CFG)
    assert sk is not None
    assert sk0 is not None
    # R(θ) leva (0, −1) a (sin θ, −cos θ): o tronco inclina +θ para a direita da imagem
    assert trunk_lean(sk) - trunk_lean(sk0) == pytest.approx(0.3, abs=1e-9)
