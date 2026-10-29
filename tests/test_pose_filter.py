"""Testes do PoseFilter (spec 006, RF7, CA6)."""

from __future__ import annotations

import numpy as np

from inmotion.filters.ema import Ema, EmaParams
from inmotion.filters.pose_filter import PoseFilter
from inmotion.types import PoseFrame
from tests.fakes import make_pose

DT = 1.0 / 30.0


def _filter() -> PoseFilter:
    return PoseFilter(lambda: Ema(EmaParams(alpha=0.5)))


def _world(value: float) -> np.ndarray:
    w = np.full((33, 4), value, dtype=np.float64)
    w[:, 3] = 0.9
    return w


def _need(pose: PoseFrame | None) -> PoseFrame:
    assert pose is not None
    return pose


def _w(pose: PoseFrame) -> np.ndarray:
    assert pose.world is not None
    return pose.world


def _rand_pose(seed: int, t: float, world: bool = True) -> PoseFrame:
    rng = np.random.default_rng(seed)
    lm = rng.uniform(0.0, 1.0, size=(33, 5))
    w = rng.uniform(-1.0, 1.0, size=(33, 4)) if world else None
    return PoseFrame(t=t, image_size=(640, 480), landmarks=lm, world=w)


def test_none_passthrough() -> None:
    pf = _filter()
    assert pf(None) is None
    first = _rand_pose(1, 0.0)
    pf(first)
    assert pf(None) is None
    # None não tocou no estado: o 2º frame ainda é a média com o 1º.
    second = _rand_pose(2, DT)
    out = _need(pf(second))
    np.testing.assert_allclose(
        out.landmarks[:, :3], 0.5 * second.landmarks[:, :3] + 0.5 * first.landmarks[:, :3]
    )


def test_preserves_visibility_and_presence() -> None:
    pf = _filter()
    for n in range(3):
        pose = _rand_pose(n, n * DT)
        out = _need(pf(pose))
        assert out is not pose
        assert out.t == pose.t
        assert out.image_size == pose.image_size
        assert np.array_equal(out.landmarks[:, 3:5], pose.landmarks[:, 3:5])
        assert np.array_equal(_w(out)[:, 3], _w(pose)[:, 3])


def test_constant_pose_is_unchanged() -> None:
    """F3: entrada constante sai constante."""
    pf = _filter()
    base = make_pose({0: (100.0, 200.0)}, world=_world(0.3))
    for n in range(5):
        pose = PoseFrame(
            t=n * DT, image_size=base.image_size, landmarks=base.landmarks, world=base.world
        )
        out = _need(pf(pose))
        np.testing.assert_allclose(out.landmarks, base.landmarks)
        np.testing.assert_allclose(_w(out), _w(base))


def test_smooths_second_frame_is_average() -> None:
    pf = _filter()
    a = _rand_pose(3, 0.0)
    b = _rand_pose(4, DT)
    out_a = _need(pf(a))
    out_b = _need(pf(b))
    np.testing.assert_allclose(out_a.landmarks[:, :3], a.landmarks[:, :3])
    np.testing.assert_allclose(
        out_b.landmarks[:, :3], 0.5 * (a.landmarks[:, :3] + b.landmarks[:, :3])
    )
    np.testing.assert_allclose(_w(out_b)[:, :3], 0.5 * (_w(a)[:, :3] + _w(b)[:, :3]))


def test_world_reset_when_pose_without_world() -> None:
    """Pose sem world zera o filtro de world; o world seguinte reinicia (não mistura sessões)."""
    pf = _filter()
    pf(_rand_pose(5, 0.0))
    assert _need(pf(_rand_pose(6, DT, world=False))).world is None
    again = _rand_pose(7, 2 * DT)
    out = _need(pf(again))
    np.testing.assert_allclose(_w(out)[:, :3], _w(again)[:, :3])
    # landmarks continuam suavizados normalmente.
    assert not np.allclose(out.landmarks[:, :3], again.landmarks[:, :3])


def test_reset_restarts_both_filters() -> None:
    pf = _filter()
    pf(_rand_pose(8, 0.0))
    pf.reset()
    p = _rand_pose(9, DT)
    out = _need(pf(p))
    np.testing.assert_allclose(out.landmarks[:, :3], p.landmarks[:, :3])
    np.testing.assert_allclose(_w(out)[:, :3], _w(p)[:, :3])
