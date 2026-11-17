"""Spec 002 T10 — CLI ``python -m inmotion``."""

import os
import subprocess
import sys
from pathlib import Path

EXPECTED_ARGS = [
    "--camera",
    "--camera-backend",
    "--log-level",
    "--log-file",
    "--no-overlay",
    "--model-dir",
    "--model-variant",
    "--config",
    "--game",
    "--set",
    "--profile",
]


def test_help_lists_arguments() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "inmotion", "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    for arg in EXPECTED_ARGS:
        assert arg in proc.stdout


def test_missing_model_exits_with_message(tmp_path: Path) -> None:
    """Sem modelo: código 2 e o comando de download; a câmera nem chega a ser aberta."""
    proc = subprocess.run(
        [sys.executable, "-m", "inmotion", "--model-dir", str(tmp_path), "--no-overlay"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode == 2
    assert "download_model.py" in output
    assert str(tmp_path) in output
    assert "câmera" not in output.lower()  # nenhuma tentativa de abrir a câmera


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "inmotion", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=60,
        check=False,
    )


def test_game_flag_loads_profile() -> None:
    from inmotion.__main__ import build_parser, resolve_config

    args = build_parser().parse_args(["--game", "minecraft"])
    config, profile = resolve_config(args)
    assert profile is not None
    assert profile.name == "minecraft"
    assert profile.bindings["jump"].input == "space"
    assert config.camera.index == 1


def test_flags_and_set_become_overrides_with_set_winning() -> None:
    from inmotion.__main__ import build_parser, resolve_config

    args = build_parser().parse_args(
        ["--camera", "4", "--model-variant", "lite", "--no-overlay", "--set", "camera.index=2"]
    )
    config, _ = resolve_config(args)
    assert config.camera.index == 2  # --set vem depois dos atalhos
    assert config.pose.variant == "lite"
    assert config.overlay.enabled is False


def test_unknown_game_exits_with_message() -> None:
    proc = _run_cli("--game", "jogo_inexistente", "--no-overlay")
    assert proc.returncode == 2
    assert "jogo_inexistente" in proc.stdout + proc.stderr


def test_bad_set_exits_with_path_and_message() -> None:
    proc = _run_cli("--set", "filters.ema.alpha=7")
    assert proc.returncode == 2
    assert "filters.ema.alpha" in proc.stdout + proc.stderr


def test_valid_profile_reaches_model_check(tmp_path: Path) -> None:
    """O perfil do Minecraft passa na validação de teclas e só então falta o modelo."""
    proc = _run_cli("--game", "minecraft", "--model-dir", str(tmp_path), "--no-overlay")
    assert proc.returncode == 2
    assert "download_model.py" in proc.stdout + proc.stderr


def test_user_profile_layer_is_available_from_cli(tmp_path: Path) -> None:
    from inmotion.__main__ import build_parser, resolve_config

    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text("handedness: left\ncamera:\n  index: 3\n", encoding="utf-8")
    config, _ = resolve_config(
        build_parser().parse_args(
            [
                "--game",
                "minecraft",
                "--profile",
                str(profile_path),
                "--set",
                "camera.index=2",
            ]
        )
    )
    assert config.handedness == "left"
    assert config.camera.index == 2


def test_unknown_hotkey_fails_before_model_or_camera() -> None:
    proc = _run_cli("--set", "safety.pause_key=banana")
    assert proc.returncode == 2
    assert "banana" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_binding_cannot_trigger_pause_key() -> None:
    proc = _run_cli("--game", "minecraft", "--set", "bindings.jump.input=f8")
    assert proc.returncode == 2
    assert "bindings.jump.input" in proc.stderr


def test_set_handedness_left_swaps_arms() -> None:
    """CA9: ``--set handedness=left``: o braço esquerdo gera ``mouse_left`` e o direito
    ``mouse_right``; sem a opção é o contrário (destro)."""
    from inmotion.__main__ import build_parser, resolve_config
    from inmotion.fps import FpsMeter
    from inmotion.gestures.factory import build_body_gestures
    from inmotion.input.backend import MOUSE_BUTTONS
    from inmotion.pipeline import Pipeline
    from inmotion.safety import SafetyConfig, SafetyManager, TrackedInput
    from inmotion.vision.landmarks import Side
    from tests.fakes import FakeInputBackend
    from tests.helpers import ARM_STRAIGHT_FORWARD, make_body

    def pressed_by(handedness_args: list[str], **arm: object) -> set[str]:
        args = build_parser().parse_args(["--game", "minecraft", *handedness_args])
        config, profile = resolve_config(args)
        assert profile is not None
        fake = FakeInputBackend()
        safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(min_track_s=0.0))
        gestures = build_body_gestures(
            profile.bindings, Side(config.handedness), config.gestures, config.geometry
        )
        pipeline = Pipeline(safety, gestures, FpsMeter(window=30), profile=profile)
        for i in range(10):
            pipeline.step(make_body(t=i / 30, **arm), i / 30)  # type: ignore[arg-type]
        return set(fake.pressed)

    left = ["--set", "handedness=left"]
    assert pressed_by(left, left_arm=ARM_STRAIGHT_FORWARD) == {"mouse_left"}
    assert pressed_by(left, right_arm=ARM_STRAIGHT_FORWARD) == {"mouse_right"}
    assert pressed_by([], right_arm=ARM_STRAIGHT_FORWARD) == {"mouse_left"}
    assert pressed_by([], left_arm=ARM_STRAIGHT_FORWARD) == {"mouse_right"}
