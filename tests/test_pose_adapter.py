"""Testes do adaptador MediaPipe com landmarker falso (spec 003, CA2 a CA4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inmotion.vision.pose import (
    MediaPipePoseEstimator,
    ModelNotFoundError,
    PoseConfig,
)

NOT_IMPLEMENTED = pytest.mark.xfail(strict=True, raises=NotImplementedError)


def _point(i: int) -> SimpleNamespace:
    """Landmark imitando ``NormalizedLandmark``/``Landmark`` do mediapipe Tasks."""
    return SimpleNamespace(
        x=i / 100.0,
        y=i / 50.0,
        z=-i / 200.0,
        visibility=1.0 - i / 100.0,
        presence=0.5 + i / 100.0,
    )


def fake_result(people: int = 1) -> SimpleNamespace:
    """Imita ``PoseLandmarkerResult``: listas (uma por pessoa) de 33 landmarks."""
    return SimpleNamespace(
        pose_landmarks=[[_point(i) for i in range(33)] for _ in range(people)],
        pose_world_landmarks=[[_point(i) for i in range(33)] for _ in range(people)],
        segmentation_masks=None,
    )


class FakeLandmarker:
    """Dublê de ``PoseLandmarker`` no modo VIDEO: registra timestamps e devolve o resultado."""

    def __init__(self, result: SimpleNamespace | None = None) -> None:
        self.result = result if result is not None else fake_result()
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image: Any, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        self.closed = True


def make_estimator(tmp_path: Path, landmarker: FakeLandmarker) -> MediaPipePoseEstimator:
    return MediaPipePoseEstimator(PoseConfig(model_dir=tmp_path), lambda _cfg: landmarker)


@NOT_IMPLEMENTED
def test_result_maps_to_pose_frame(tmp_path: Path) -> None:
    est = make_estimator(tmp_path, FakeLandmarker())
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = est.estimate(img, 1.5)
    assert frame is not None
    assert frame.t == 1.5
    assert frame.image_size == (64, 48)
    assert frame.landmarks.shape == (33, 5)
    assert frame.world is not None
    assert frame.world.shape == (33, 4)
    expected = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in map(_point, range(33))])
    np.testing.assert_allclose(frame.landmarks, expected)
    np.testing.assert_allclose(frame.world, expected[:, :4])


@NOT_IMPLEMENTED
def test_no_pose_returns_none(tmp_path: Path) -> None:
    empty = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)
    est = make_estimator(tmp_path, FakeLandmarker(empty))
    assert est.estimate(np.zeros((8, 8, 3), dtype=np.uint8), 0.0) is None


@NOT_IMPLEMENTED
def test_timestamps_strictly_increasing(tmp_path: Path) -> None:
    fake = FakeLandmarker()
    est = make_estimator(tmp_path, fake)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for t in [0.0, 0.033, 0.033, 0.020, 0.066]:
        est.estimate(img, t)
    assert fake.timestamps == [0, 33, 34, 35, 66]


@NOT_IMPLEMENTED
def test_missing_model_raises_with_instructions(tmp_path: Path) -> None:
    config = PoseConfig(model_dir=tmp_path / "vazio", variant="lite")
    with pytest.raises(ModelNotFoundError) as info:
        MediaPipePoseEstimator(config)
    message = str(info.value)
    assert "pose_landmarker_lite.task" in message
    assert "tools/download_model.py" in message
    assert "--variant lite" in message
