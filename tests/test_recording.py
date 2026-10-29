"""Testes do formato de gravação (spec 011 T2)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from inmotion.recording import (
    SCHEMA_VERSION,
    LabelInterval,
    LabelTracker,
    Recording,
    RecordingMeta,
)


def make_meta() -> RecordingMeta:
    return RecordingMeta(
        schema_version=SCHEMA_VERSION,
        created="2026-10-29T12:00:00",
        person="p1",
        distance_m=2.0,
        lighting="clara",
        camera="cam0",
        config_hash="abc123",
    )


def make_recording(n: int = 5, **overrides: Any) -> Recording:
    kwargs: dict[str, Any] = {
        "meta": make_meta(),
        "image_size": (640, 480),
        "t": np.arange(n, dtype=np.float64) / 30.0,
        "landmarks": np.full((n, 33, 5), 0.5),
        "world": np.full((n, 33, 4), 0.1),
        "present": np.ones(n, dtype=np.bool_),
        "labels": (),
    }
    kwargs.update(overrides)
    return Recording(**kwargs)


def test_recording_validates_shapes_and_time() -> None:
    rec = make_recording(5)
    assert rec.duration_s == pytest.approx(4 / 30.0)
    for arr in (rec.t, rec.landmarks, rec.world, rec.present):
        assert not arr.flags.writeable

    with pytest.raises(ValueError, match="landmarks"):
        make_recording(5, landmarks=np.zeros((5, 33, 4)))
    with pytest.raises(ValueError, match="world"):
        make_recording(5, world=np.zeros((4, 33, 4)))
    with pytest.raises(ValueError, match="present"):
        make_recording(5, present=np.ones(4, dtype=np.bool_))
    with pytest.raises(ValueError, match="t deve"):
        make_recording(5, t=np.zeros((5, 1)))
    with pytest.raises(ValueError, match="crescente"):
        make_recording(5, t=np.array([0.0, 0.1, 0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="finito"):
        make_recording(5, t=np.array([0.0, 0.1, math.nan, 0.3, 0.4]))
    with pytest.raises(ValueError, match="rótulo"):
        make_recording(5, labels=(LabelInterval("attack", 1.0, 1.0),))
    with pytest.raises(ValueError, match="image_size"):
        make_recording(5, image_size=(0, 480))


def test_label_tracker_builds_intervals() -> None:
    tracker = LabelTracker({"1": "attack", "2": "jump"})
    none: frozenset[str] = frozenset()
    tracker.update(none, 1.0)
    tracker.update(frozenset({"1"}), 2.0)
    tracker.update(frozenset({"1"}), 2.5)
    tracker.update(frozenset({"1"}), 3.5)
    tracker.update(none, 3.6)  # soltou em 3,6
    tracker.update(frozenset({"2", "x"}), 4.0)  # "x" não é tecla de rótulo
    # "2" ainda segurada no close: fecha em 5,0
    assert tracker.close(5.0) == (
        LabelInterval("attack", 2.0, 3.6),
        LabelInterval("jump", 4.0, 5.0),
    )


def test_poses_yield_none_when_absent() -> None:
    present = np.array([True, False, True])
    landmarks = np.full((3, 33, 5), 0.5)
    landmarks[1] = np.nan
    rec = make_recording(3, present=present, landmarks=landmarks)
    out = list(rec.poses())
    assert [t for t, _ in out] == pytest.approx([0.0, 1 / 30, 2 / 30])
    assert out[1][1] is None
    first = out[0][1]
    assert first is not None
    assert first.t == 0.0
    assert first.image_size == (640, 480)
    assert first.landmarks.shape == (33, 5)
    assert first.world is not None
    assert first.world.shape == (33, 4)
