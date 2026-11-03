"""Fluxos de demonstração e relatórios executados sem câmera ou injeção de input."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from inmotion.config.schema import AppConfig, FilterConfig
from inmotion.filters.ema import EmaParams
from inmotion.gestures.base import EventKind
from inmotion.recorder import load_recording
from inmotion.replay import run_replay
from inmotion.safety import SafetyConfig
from tests.fakes import FakeInputBackend
from tests.helpers import ARM_STRAIGHT_FORWARD, make_body
from tools.replay import build_pipeline
from tools.replay import main as replay_main
from tools.simulate import FIXTURES, SCENARIOS, main, render_frame, show


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_headless_simulation_reports_and_releases_all_keys(tmp_path: Path, scenario: str) -> None:
    output = tmp_path / f"{scenario}.json"
    assert main(["--scenario", scenario, "--headless", "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["frames"] > 100
    assert report["events"]
    pressed: set[str] = set()
    for event in report["inputs"]:
        if event["kind"] == "press":
            pressed.add(event["key"])
        else:
            pressed.discard(event["key"])
    assert pressed == set()
    assert any(e["kind"] == "press" and e["key"] == "space" for e in report["inputs"])


def test_replay_cli_accepts_profile_overrides_and_exports_events(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"
    profile = tmp_path / "custom.yaml"
    profile.write_text("bindings:\n  jump: {input: enter, kind: trigger}\n", encoding="utf-8")
    assert (
        replay_main(
            [
                "--recording",
                str(FIXTURES / "crouch_2m.npz"),
                "--game",
                "minecraft",
                "--profile",
                str(profile),
                "--set",
                "filters.kind=none",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert any(e["key"] == "enter" and e["kind"] == "press" for e in report["inputs"])
    assert any(e["gesture"] == "jump" and e["kind"] == "pulse" for e in report["events"])


def test_replay_applies_configured_filter_and_handedness() -> None:
    config = replace(
        AppConfig(),
        handedness="left",
        safety=SafetyConfig(min_track_s=0),
        filters=FilterConfig(kind="ema", ema=EmaParams(alpha=0.25)),
    )
    fake = FakeInputBackend()
    pipeline, _ = build_pipeline(fake, app_config=config)
    for i in range(5):
        t = i / 30
        result = pipeline.step(make_body(t=t, left_arm=ARM_STRAIGHT_FORWARD), t)
    assert result.pressed == frozenset({"mouse_left"})
    p0 = make_body(t=1.0, hip_y_px=300)
    p1 = make_body(t=1.1, hip_y_px=250)
    pipeline.step(p0, 1.0)
    result = pipeline.step(p1, 1.1)
    assert result.pose is not None
    np.testing.assert_allclose(
        result.pose.landmarks[23:25, 1],
        0.75 * p0.landmarks[23:25, 1] + 0.25 * p1.landmarks[23:25, 1],
    )


def test_simulator_renders_active_commands_without_modifying_snapshot() -> None:
    recording = load_recording(FIXTURES / "march_2m.npz")
    replay = run_replay(recording, build_pipeline, keep_snapshots=True)
    assert replay.snapshots is not None
    result = next(s for s in replay.snapshots if "w" in s.pressed)
    frame = render_frame(recording, result, replay)
    assert frame.shape == (640, 1040, 3)
    assert frame.dtype == np.uint8
    assert "w" in result.pressed
    assert any(e.kind is EventKind.START for e in replay.events)


def test_simulator_playback_controls_and_window_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.simulate as simulator

    recording = load_recording(FIXTURES / "march_2m.npz")
    replay = run_replay(recording, build_pipeline, keep_snapshots=True)
    keys = iter([ord(" "), ord("n"), ord("r"), 27])
    drawn: list[np.ndarray] = []
    closed: list[bool] = []
    monkeypatch.setattr(simulator.cv2, "imshow", lambda name, frame: drawn.append(frame))
    monkeypatch.setattr(simulator.cv2, "waitKey", lambda ms: next(keys))
    monkeypatch.setattr(simulator.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(simulator.cv2, "destroyAllWindows", lambda: closed.append(True))
    show(recording, replay, 1.0)
    assert len(drawn) == 4
    assert closed == [True]


@pytest.mark.parametrize("speed", ["0", "-1", "nan", "inf"])
def test_simulator_rejects_invalid_speed(speed: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--headless", "--speed", speed])
    assert exc.value.code == 2


def test_simulator_missing_recording_returns_error(tmp_path: Path) -> None:
    assert main(["--headless", "--recording", str(tmp_path / "missing.npz")]) == 1
