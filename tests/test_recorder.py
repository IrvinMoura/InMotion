"""Testes de gravação/leitura do ``.npz`` (spec 011 T3; CA1, CA6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inmotion.recorder import Recorder, load_recording
from inmotion.recording import (
    SCHEMA_VERSION,
    LabelInterval,
    RecordingFormatError,
    RecordingMeta,
)
from tests.fakes import make_pose

_TDD = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="spec 011 T1")


def make_meta(schema_version: int = SCHEMA_VERSION) -> RecordingMeta:
    return RecordingMeta(
        schema_version=schema_version,
        created="2026-10-29T12:00:00",
        person="p1",
        distance_m=None,
        lighting="escura",
        camera="cam0",
        config_hash="abc123",
    )


@_TDD
def test_save_load_roundtrip(tmp_path: Path) -> None:
    meta = make_meta()
    rec = Recorder(meta, (640, 480))
    world = np.full((33, 4), 0.25)
    rec.add(0.0, make_pose({0: (320.0, 100.0)}, world=world))
    rec.add(0.04, None)
    rec.add(0.08, make_pose({0: (330.0, 110.0)}))  # sem world
    labels = [LabelInterval("attack", 0.0, 0.05)]

    path = rec.save(tmp_path / "sessao.npz", labels)
    loaded = load_recording(path)

    assert loaded.meta == meta
    assert loaded.image_size == (640, 480)
    assert loaded.labels == tuple(labels)
    np.testing.assert_array_equal(loaded.t, [0.0, 0.04, 0.08])
    np.testing.assert_array_equal(loaded.present, [True, False, True])
    assert np.isnan(loaded.landmarks[1]).all()
    assert np.isnan(loaded.world[1]).all()
    assert np.isnan(loaded.world[2]).all()  # pose sem world → NaN
    np.testing.assert_array_equal(loaded.world[0], world)
    assert loaded.landmarks[0, 0, 0] == pytest.approx(0.5)
    assert loaded.landmarks[2, 0, 1] == pytest.approx(110.0 / 480.0)


@_TDD
def test_incompatible_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "futuro.npz"
    np.savez_compressed(path, meta_json=np.array(json.dumps({"schema_version": 99})))
    with pytest.raises(RecordingFormatError, match="99"):
        load_recording(path)
