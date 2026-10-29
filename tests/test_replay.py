"""Replay determinístico (spec 011 T5; CA2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from inmotion.fps import FpsMeter
from inmotion.gestures.legacy import LegacyJump
from inmotion.input.backend import MOUSE_BUTTONS, InputBackend
from inmotion.pipeline import Pipeline, StepResult
from inmotion.recorder import Recorder, load_recording
from inmotion.recording import Recording, RecordingMeta
from inmotion.replay import RecordingInputBackend, run_replay
from inmotion.safety import SafetyConfig, SafetyManager, TrackedInput
from inmotion.types import PoseFrame
from tests.fakes import FakeInputBackend, make_pose
from tools.replay import build_legacy_pipeline, main

SIZE = (640, 480)
HIPS = (23, 24)
WRISTS = (15, 16)


def hip_pose(y: float) -> PoseFrame:
    # punhos baixos (abaixo dos quadris): nenhum soco dispara, só o pulo
    points = dict.fromkeys(HIPS, (320.0, y))
    points.update(dict.fromkeys(WRISTS, (320.0, 470.0)))
    return make_pose(points, size=SIZE)


# Base em y=400; pulo em y=300 (sobe > rise·H); volta; perde a pose; volta; pula e termina
# com a tecla apertada (o replay tem de soltar no fim).
FRAMES: list[tuple[float, PoseFrame | None]] = [
    (0.00, hip_pose(400)),
    (0.05, hip_pose(400)),
    (0.10, hip_pose(300)),
    (0.15, hip_pose(300)),
    (0.20, hip_pose(400)),
    (0.25, None),
    (0.30, hip_pose(400)),
    (0.35, hip_pose(300)),
]


def make_meta() -> RecordingMeta:
    return RecordingMeta(
        schema_version=1,
        created="2026-10-29T12:00:00",
        person="p1",
        distance_m=None,
        lighting="clara",
        camera="cam0",
        config_hash="x",
    )


def record(path: Path) -> Path:
    rec = Recorder(make_meta(), SIZE)
    for t, pose in FRAMES:
        rec.add(t, pose)
    return rec.save(path, [])


def synthetic_recording() -> Recording:
    n = len(FRAMES)
    landmarks = np.full((n, 33, 5), np.nan)
    present = np.zeros(n, dtype=bool)
    for i, (_, pose) in enumerate(FRAMES):
        if pose is not None:
            landmarks[i] = pose.landmarks
            present[i] = True
    return Recording(
        meta=make_meta(),
        image_size=SIZE,
        t=np.array([t for t, _ in FRAMES]),
        landmarks=landmarks,
        world=np.full((n, 33, 4), np.nan),
        present=present,
        labels=(),
    )


def test_replay_is_deterministic() -> None:
    rec = synthetic_recording()
    first = run_replay(rec, build_legacy_pipeline)
    second = run_replay(rec, build_legacy_pipeline)
    assert first.inputs == second.inputs
    assert first == second
    assert any(kind == "press" for _, kind, _ in first.inputs)
    assert first.events == ()


def test_replay_matches_live_pipeline() -> None:
    rec = synthetic_recording()
    result = run_replay(rec, build_legacy_pipeline)

    fake = FakeInputBackend()
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig())
    pipeline = Pipeline(safety, [LegacyJump("space")], FpsMeter())
    with safety.guard():
        for t, pose in rec.poses():
            pipeline.step(pose, t)
    replayed = [(kind, key) for _, kind, key in result.inputs]
    assert replayed == fake.events
    assert ("press", "space") in replayed


def test_replay_timestamps_come_from_recording() -> None:
    result = run_replay(synthetic_recording(), build_legacy_pipeline)
    assert {t for t, _, _ in result.inputs} <= {t for t, _ in FRAMES}
    assert result.inputs[0] == (0.10, "press", "space")


def test_replay_releases_everything_at_end() -> None:
    result = run_replay(synthetic_recording(), build_legacy_pipeline)
    # release_all() solta P ∪ botões do mouse (sempre), ordenado; tudo em t = último frame
    tail = [r for r in result.inputs if r[0] == 0.35]
    assert tail[0] == (0.35, "press", "space")
    assert [r[2] for r in tail[1:]] == sorted(["space", *MOUSE_BUTTONS])
    assert all(kind == "release" for _, kind, _ in tail[1:])
    spaces = [kind for _, kind, key in result.inputs if key == "space"]
    assert spaces.count("press") == spaces.count("release")


class BoomError(Exception):
    pass


def test_replay_releases_on_exception() -> None:
    captured: list[RecordingInputBackend] = []

    def build(backend: InputBackend) -> tuple[Pipeline, SafetyManager]:
        assert isinstance(backend, RecordingInputBackend)
        captured.append(backend)
        inner, safety = build_legacy_pipeline(backend)

        class Exploding(Pipeline):
            def step(self, pose: PoseFrame | None, t: float) -> StepResult:
                if t >= 0.15:
                    raise BoomError
                return inner.step(pose, t)

        return Exploding(safety, [], FpsMeter()), safety

    with pytest.raises(BoomError):
        run_replay(synthetic_recording(), build)
    space = [(kind, key) for _, kind, key in captured[0].records if key == "space"]
    assert space == [("press", "space"), ("release", "space")]


def test_snapshots_only_when_requested() -> None:
    rec = synthetic_recording()
    assert run_replay(rec, build_legacy_pipeline).snapshots is None
    snaps = run_replay(rec, build_legacy_pipeline, keep_snapshots=True).snapshots
    assert snaps is not None
    assert len(snaps) == len(FRAMES)


def test_roundtrip_record_load_replay(tmp_path: Path) -> None:
    loaded = load_recording(record(tmp_path / "s.npz"))
    direct = run_replay(synthetic_recording(), build_legacy_pipeline)
    assert run_replay(loaded, build_legacy_pipeline).inputs == direct.inputs


def test_tool_main_returns_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = record(tmp_path / "s.npz")
    with caplog.at_level("INFO", logger="replay"):
        assert main(["--recording", str(path)]) == 0
    assert "frames: 8" in caplog.text
    assert "space: 2 press, 2 release" in caplog.text


def test_tool_main_missing_file_returns_one(tmp_path: Path) -> None:
    assert main(["--recording", str(tmp_path / "nao_existe.npz")]) == 1
