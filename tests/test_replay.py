"""Replay determinístico (spec 011 T5; CA2)."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest

from inmotion.app import DEFAULT_PROFILE
from inmotion.fps import FpsMeter
from inmotion.input.backend import MOUSE_BUTTONS, InputBackend
from inmotion.pipeline import Pipeline, StepResult
from inmotion.recorder import Recorder, load_recording
from inmotion.recording import Recording, RecordingMeta
from inmotion.replay import RecordingInputBackend, run_replay
from inmotion.safety import SafetyConfig, SafetyManager
from inmotion.types import PoseFrame
from tests.fakes import FakeInputBackend
from tests.helpers import make_body
from tools.replay import build_pipeline, main

SIZE = (640, 480)
DT = 1 / 30
# A fixture tem ~1,2 s, mais que a persistência padrão (min_track_s = 1 s) só no fim: desligada.
build = partial(build_pipeline, config=SafetyConfig(min_track_s=0.0))


def body(y: float) -> PoseFrame:
    return make_body(hip_y_px=y)  # braços pendurados: nenhum soco dispara, só o pulo


# Base R7 em y=300 (20 quadros, K = 15); pulo (sobe 50 px = 0,5 largura de ombro); volta;
# perde a pose; volta; pula e termina com o tap apertado (o replay tem de soltar no fim).
YS: list[float | None] = [
    *([300.0] * 20),
    *([250.0] * 3),
    *([300.0] * 5),
    None,
    *([300.0] * 4),
    *([250.0] * 3),
]
FRAMES: list[tuple[float, PoseFrame | None]] = [
    (i * DT, None if y is None else body(y)) for i, y in enumerate(YS)
]
T_FIRST_PRESS = 21 * DT  # 2.º quadro acima de theta_on (debounce N = 2)
T_LAST = (len(FRAMES) - 1) * DT


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
    first = run_replay(rec, build)
    second = run_replay(rec, build)
    assert first.inputs == second.inputs
    assert first == second
    assert any(kind == "press" for _, kind, _ in first.inputs)
    assert first.events == ()


def test_replay_matches_live_pipeline() -> None:
    rec = synthetic_recording()
    result = run_replay(rec, build)

    fake = FakeInputBackend()
    pipeline, safety = build(fake)
    with safety.guard():
        for t, pose in rec.poses():
            pipeline.step(pose, t)
    replayed = [(kind, key) for _, kind, key in result.inputs]
    assert replayed == fake.events
    assert ("press", "space") in replayed


def test_replay_timestamps_come_from_recording() -> None:
    result = run_replay(synthetic_recording(), build)
    assert {t for t, _ in FRAMES} >= {t for t, _, _ in result.inputs}
    assert result.inputs[0] == (T_FIRST_PRESS, "press", "space")
    assert result.inputs[1] == (24 * DT, "release", "space")  # tap de 80 ms


def test_replay_releases_everything_at_end() -> None:
    result = run_replay(synthetic_recording(), build)
    # release_all() solta P ∪ botões do mouse (sempre), ordenado; tudo em t = último frame
    tail = [r for r in result.inputs if r[0] == T_LAST]
    assert [r[2] for r in tail] == sorted(["space", *MOUSE_BUTTONS])
    assert all(kind == "release" for _, kind, _ in tail)
    spaces = [kind for _, kind, key in result.inputs if key == "space"]
    assert spaces == ["press", "release", "press", "release"]


class BoomError(Exception):
    pass


def test_replay_releases_on_exception() -> None:
    captured: list[RecordingInputBackend] = []

    def exploding_build(backend: InputBackend) -> tuple[Pipeline, SafetyManager]:
        assert isinstance(backend, RecordingInputBackend)
        captured.append(backend)
        inner, safety = build(backend)

        class Exploding(Pipeline):
            def step(self, pose: PoseFrame | None, t: float) -> StepResult:
                if t >= T_LAST:
                    raise BoomError
                return inner.step(pose, t)

        return Exploding(safety, [], FpsMeter(), profile=DEFAULT_PROFILE), safety

    with pytest.raises(BoomError):
        run_replay(synthetic_recording(), exploding_build)
    space = [(kind, key) for _, kind, key in captured[0].records if key == "space"]
    assert space == [("press", "space"), ("release", "space")] * 2


def test_snapshots_only_when_requested() -> None:
    rec = synthetic_recording()
    assert run_replay(rec, build).snapshots is None
    snaps = run_replay(rec, build, keep_snapshots=True).snapshots
    assert snaps is not None
    assert len(snaps) == len(FRAMES)


def test_roundtrip_record_load_replay(tmp_path: Path) -> None:
    loaded = load_recording(record(tmp_path / "s.npz"))
    direct = run_replay(synthetic_recording(), build)
    assert run_replay(loaded, build).inputs == direct.inputs


def test_tool_main_returns_zero(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = record(tmp_path / "s.npz")
    with caplog.at_level("INFO", logger="replay"):
        assert main(["--recording", str(path)]) == 0
    assert f"frames: {len(FRAMES)}" in caplog.text
    # o tool usa a config padrão do app: só os quadros após min_track_s = 1 s valem; o pulo
    # da fixture (20 quadros de base) vem antes, então não aperta nada
    assert "space:" not in caplog.text


def test_tool_main_missing_file_returns_one(tmp_path: Path) -> None:
    assert main(["--recording", str(tmp_path / "nao_existe.npz")]) == 1
