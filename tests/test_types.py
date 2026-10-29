"""Testes de inmotion.types e inmotion.vision.landmarks (spec 002 T2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numpy.typing import NDArray

from inmotion.types import N_LANDMARKS, FramePacket, Landmark, PoseFrame
from inmotion.vision.landmarks import PoseLandmark, Side


def _lm(seed: int = 0) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    return rng.random((N_LANDMARKS, 5))


def _frame(**kw: object) -> PoseFrame:
    args: dict[str, object] = {"t": 1.0, "image_size": (640, 480), "landmarks": _lm()}
    args.update(kw)
    return PoseFrame(**args)  # type: ignore[arg-type]


def test_pose_frame_valid() -> None:
    f = _frame(world=np.zeros((N_LANDMARKS, 4)))
    assert f.landmarks.shape == (33, 5)
    assert f.landmarks.dtype == np.float64
    assert f.world is not None
    assert f.world.shape == (33, 4)


@pytest.mark.parametrize("shape", [(33, 4), (32, 5), (5,), (33, 5, 1)])
def test_pose_frame_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="forma"):
        _frame(landmarks=np.zeros(shape))


def test_pose_frame_rejects_wrong_world_shape() -> None:
    with pytest.raises(ValueError, match="forma"):
        _frame(world=np.zeros((33, 5)))


@pytest.mark.parametrize("t", [math.nan, math.inf, -math.inf])
def test_pose_frame_rejects_non_finite_time(t: float) -> None:
    with pytest.raises(ValueError, match="finito"):
        _frame(t=t)


@pytest.mark.parametrize("size", [(0, 480), (640, 0), (-1, 480), (640.0, 480), (640,)])
def test_pose_frame_rejects_bad_image_size(size: object) -> None:
    with pytest.raises(ValueError, match="image_size"):
        _frame(image_size=size)


def test_pose_frame_arrays_are_read_only() -> None:
    f = _frame(world=np.zeros((N_LANDMARKS, 4)))
    assert not f.landmarks.flags.writeable
    assert f.world is not None
    assert not f.world.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        f.landmarks[0, 0] = 1.0


def test_pose_frame_copies_input_and_casts_to_float64() -> None:
    src = np.ones((N_LANDMARKS, 5), dtype=np.float32)
    f = _frame(landmarks=src)
    assert f.landmarks.dtype == np.float64
    src[0, 0] = 9.0
    assert f.landmarks[0, 0] == 1.0
    assert src.flags.writeable


def test_landmark_returns_row_without_presence() -> None:
    arr = _lm()
    f = _frame(landmarks=arr)
    assert f.landmark(7) == Landmark(arr[7, 0], arr[7, 1], arr[7, 2], arr[7, 3])


@pytest.mark.parametrize("index", [-1, 33, 100])
def test_landmark_index_out_of_range(index: int) -> None:
    with pytest.raises(IndexError):
        _frame().landmark(index)


@given(
    w=st.integers(1, 4000),
    h=st.integers(1, 4000),
    xs=st.lists(st.floats(0, 1), min_size=33, max_size=33),
    ys=st.lists(st.floats(0, 1), min_size=33, max_size=33),
)
def test_pixels_scales_by_image_size(w: int, h: int, xs: list[float], ys: list[float]) -> None:
    arr = _lm()
    arr[:, 0] = xs
    arr[:, 1] = ys
    px = _frame(image_size=(w, h), landmarks=arr).pixels()
    assert px.shape == (33, 2)
    np.testing.assert_allclose(px[:, 0], arr[:, 0] * w)
    np.testing.assert_allclose(px[:, 1], arr[:, 1] * h)


def test_pixels_uses_width_for_x_and_height_for_y() -> None:
    arr = np.zeros((N_LANDMARKS, 5))
    arr[:, 0] = 0.5
    arr[:, 1] = 0.5
    px = _frame(image_size=(200, 100), landmarks=arr).pixels()
    assert px[0].tolist() == [100.0, 50.0]


def _img(h: int = 4, w: int = 6) -> NDArray[np.uint8]:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_frame_packet_valid_and_read_only_without_copy() -> None:
    img = _img()
    p = FramePacket(t=0.5, index=3, image=img)
    assert p.image.shape == (4, 6, 3)
    assert not p.image.flags.writeable
    assert np.shares_memory(p.image, img)
    assert img.flags.writeable


@pytest.mark.parametrize("t", [math.nan, math.inf])
def test_frame_packet_rejects_non_finite_time(t: float) -> None:
    with pytest.raises(ValueError, match="finito"):
        FramePacket(t=t, index=0, image=_img())


def test_frame_packet_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="index"):
        FramePacket(t=0.0, index=-1, image=_img())


def test_frame_packet_rejects_bad_image() -> None:
    with pytest.raises(ValueError, match="image deve ser"):
        FramePacket(t=0.0, index=0, image=np.zeros((4, 6, 3), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="image deve ser"):
        FramePacket(t=0.0, index=0, image=np.zeros((4, 6), dtype=np.uint8))
    with pytest.raises(ValueError, match="image deve ser"):
        FramePacket(t=0.0, index=0, image=np.zeros((4, 6, 4), dtype=np.uint8))


def test_pose_landmark_has_33_members_with_indices_0_to_32() -> None:
    assert len(PoseLandmark) == 33
    assert [int(m) for m in PoseLandmark] == list(range(33))
    assert len(PoseLandmark) == N_LANDMARKS


def test_pose_landmark_known_indices() -> None:
    assert PoseLandmark.NOSE == 0
    assert (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER) == (11, 12)
    assert (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST) == (15, 16)
    assert (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP) == (23, 24)
    assert (PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE) == (27, 28)
    assert PoseLandmark.RIGHT_FOOT_INDEX == 32


def test_side_has_left_and_right() -> None:
    assert {s.name for s in Side} == {"LEFT", "RIGHT"}
