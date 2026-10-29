"""Testes dos dublês compartilhados do pipeline (spec 002 T1)."""

import numpy as np
import pytest

from inmotion.ports import Clock, FrameSource, PoseEstimator
from inmotion.types import FramePacket, PoseFrame
from tests.fakes import (
    FakeCapture,
    FakeClock,
    FakeFrameSource,
    FakePoseEstimator,
    make_pose,
)


class ScriptedEstimatorError(Exception):
    """Falha roteirizada do estimador (não herda de RuntimeError de propósito)."""


def test_fake_clock_advances_only_when_told() -> None:
    clock: Clock = FakeClock(t0=1.0)
    assert clock.now() == 1.0
    assert isinstance(clock, FakeClock)
    clock.advance(0.5)
    assert clock.now() == 1.5
    with pytest.raises(ValueError, match="dt"):
        clock.advance(-0.1)


def test_fake_frame_source_follows_script_then_returns_none() -> None:
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    source: FrameSource = FakeFrameSource([FramePacket(0.0, 0, img), None])
    first = source.read(timeout_s=0.1)
    assert first is not None
    assert first.index == 0
    assert source.read(timeout_s=0.1) is None
    assert source.read(timeout_s=0.1) is None
    source.close()
    assert isinstance(source, FakeFrameSource)
    assert source.closed


def test_fake_pose_estimator_follows_script_and_raises() -> None:
    pose = make_pose({}, size=(640, 480))
    estimator = FakePoseEstimator([pose, None, ScriptedEstimatorError("falha")])
    port: PoseEstimator = estimator
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    assert port.estimate(img, 0.0) is pose
    assert port.estimate(img, 0.1) is None
    with pytest.raises(ScriptedEstimatorError):
        port.estimate(img, 0.2)
    assert port.estimate(img, 0.3) is None  # roteiro esgotado
    assert estimator.calls == [0.0, 0.1, 0.2, 0.3]


def test_make_pose_converts_pixels_to_normalized() -> None:
    pose = make_pose({11: (320.0, 120.0), 12: (160.0, 240.0)}, size=(640, 480), t=2.0)
    assert isinstance(pose, PoseFrame)
    assert pose.t == 2.0
    np.testing.assert_allclose(pose.pixels()[11], [320.0, 120.0])
    np.testing.assert_allclose(pose.pixels()[12], [160.0, 240.0])
    assert pose.landmark(0).visibility == 1.0
    assert pose.world is None


def test_make_pose_visibility_override_and_world() -> None:
    world = np.zeros((33, 4))
    pose = make_pose({15: (10.0, 10.0)}, visibility={15: 0.3}, world=world)
    assert pose.landmark(15).visibility == pytest.approx(0.3)
    assert pose.landmark(16).visibility == 1.0
    assert pose.world is not None


def test_fake_capture_records_settings_and_counts_frames() -> None:
    cap = FakeCapture(width=8, height=6, max_frames=2)
    assert cap.isOpened()
    assert cap.set(3, 640.0)
    assert cap.get(3) == 640.0
    ok, frame = cap.read()
    assert ok
    assert frame.shape == (6, 8, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0, 0] == 0
    ok, frame = cap.read()
    assert frame[0, 0, 0] == 1
    ok, _ = cap.read()
    assert not ok
    assert cap.reads == 3
    cap.release()
    assert not cap.isOpened()
