"""Spec 011 T4 — ferramenta de gravação com câmera e estimador falsos (RF1, RF2)."""

from pathlib import Path

import numpy as np
import pytest

from inmotion.recorder import load_recording
from inmotion.recording import SCHEMA_VERSION, RecordingMeta
from inmotion.types import FramePacket
from tests.fakes import FakeClock, FakeFrameSource, FakeHotkeys, FakePoseEstimator, make_pose
from tools.record import record_session

SIZE = (64, 48)  # (W, H)
NONE: frozenset[str] = frozenset()
F9 = frozenset({"f9"})
KEY_1 = frozenset({"1"})


class EstimatorBoomError(Exception):
    """Falha simulada do estimador (não herda de RuntimeError)."""


def _meta() -> RecordingMeta:
    return RecordingMeta(
        schema_version=SCHEMA_VERSION,
        created="",
        person="p1",
        distance_m=2.0,
        lighting="clara",
        camera="fake",
        config_hash="abc",
    )


def _frames(n: int) -> FakeFrameSource:
    image = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    return FakeFrameSource([FramePacket(t=i * 0.1, index=i, image=image) for i in range(n)])


def test_record_with_fakes(tmp_path: Path) -> None:
    source = _frames(6)
    pose = make_pose({0: (10.0, 10.0)}, size=SIZE)
    estimator = FakePoseEstimator([pose, None, pose, pose, pose])
    # Um sondagem por frame lido; "1" seguro nos frames 1..3; F9 no frame 5.
    hotkeys = FakeHotkeys([NONE, KEY_1, KEY_1, KEY_1, NONE, F9])
    out = tmp_path / "sessao.npz"

    result = record_session(source, estimator, hotkeys, FakeClock(), _meta(), out)

    assert result == out
    rec = load_recording(out)
    assert rec.t.shape == (5,)
    assert rec.image_size == SIZE
    assert rec.present.tolist() == [True, False, True, True, True]
    assert np.isnan(rec.landmarks[1]).all()
    assert not np.isnan(rec.landmarks[0]).any()
    assert [(lab.gesture, lab.start, lab.end) for lab in rec.labels] == [
        ("punch_right", pytest.approx(0.1), pytest.approx(0.4))
    ]
    assert rec.meta.created != ""
    assert rec.meta.person == "p1"
    assert source.closed
    assert estimator.closed


def test_estimator_exception_still_saves(tmp_path: Path) -> None:
    pose = make_pose({0: (10.0, 10.0)}, size=SIZE)
    estimator = FakePoseEstimator([pose, pose, EstimatorBoomError("falha")])
    source = _frames(6)
    out = tmp_path / "parcial.npz"

    with pytest.raises(EstimatorBoomError):
        record_session(source, estimator, FakeHotkeys([]), FakeClock(), _meta(), out)

    rec = load_recording(out)
    assert rec.t.shape == (2,)
    assert source.closed
    assert estimator.closed


def test_max_frames_stops_recording(tmp_path: Path) -> None:
    out = tmp_path / "curta.npz"
    record_session(
        _frames(10), FakePoseEstimator([]), FakeHotkeys([]), FakeClock(), _meta(), out, max_frames=3
    )
    assert load_recording(out).t.shape == (3,)


def test_missing_model_returns_2_before_opening_camera(tmp_path: Path) -> None:
    from tools.record import main

    code = main(["--person", "p1", "--out", str(tmp_path / "x.npz"), "--model-dir", str(tmp_path)])

    assert code == 2
    assert not (tmp_path / "x.npz").exists()


def test_config_hash_is_stable_and_sensitive() -> None:
    from inmotion.capture.camera import CameraConfig
    from tools.record import DEFAULT_LABEL_KEYS, compute_config_hash

    base = compute_config_hash(CameraConfig(), "full", DEFAULT_LABEL_KEYS)
    assert base == compute_config_hash(CameraConfig(), "full", dict(DEFAULT_LABEL_KEYS))
    assert base != compute_config_hash(CameraConfig(index=2), "full", DEFAULT_LABEL_KEYS)
    assert base != compute_config_hash(CameraConfig(), "lite", DEFAULT_LABEL_KEYS)
