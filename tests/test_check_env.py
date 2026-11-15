"""Spec 015 — testes do verificador de ambiente (``tools/check_env.py``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from inmotion.config.schema import CameraConfig
from tools.check_env import EnvHooks, main


def _write_model(model_dir: Path, variant: str = "full", data: bytes = b"ok") -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"pose_landmarker_{variant}.task"
    path.write_bytes(data)
    return path


def _ok_hooks(
    *,
    version_info: tuple[int, int, int] = (3, 10, 11),
    platform: str = "win32",
    missing_module: str | None = None,
    camera_error: Exception | None = None,
    resolved_camera: int = 1,
) -> EnvHooks:
    def _import(name: str) -> object:
        if missing_module is not None and name == missing_module:
            raise ImportError(f"No module named {name!r}")
        return object()

    def _probe(cfg: CameraConfig) -> tuple[int, tuple[int, int, float]]:
        if camera_error is not None:
            raise camera_error
        return resolved_camera, (cfg.width, cfg.height, float(cfg.fps))

    return EnvHooks(
        version_info=version_info,
        platform=platform,
        import_module=_import,
        probe_camera=_probe,
    )


def test_check_env_passes_when_all_requirements_satisfied(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_model(tmp_path)
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc = main(["--model-dir", str(tmp_path)], hooks=_ok_hooks())
    assert rc == 0
    assert "[OK]" in caplog.text
    assert "FALHOU" not in caplog.text


def test_check_env_fails_on_wrong_python_version(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_model(tmp_path)
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc = main(["--model-dir", str(tmp_path)], hooks=_ok_hooks(version_info=(3, 12, 0)))
    assert rc == 1
    assert "esperado 3.10.x" in caplog.text


def test_check_env_fails_on_missing_dependency(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_model(tmp_path)
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc = main(["--model-dir", str(tmp_path)], hooks=_ok_hooks(missing_module="mediapipe"))
    assert rc == 1
    assert "mediapipe" in caplog.text


def test_check_env_fails_on_missing_or_empty_model(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc_missing = main(["--model-dir", str(tmp_path), "--skip-camera"], hooks=_ok_hooks())
    assert rc_missing == 1
    assert "download_model.py" in caplog.text

    _write_model(tmp_path, data=b"")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc_empty = main(["--model-dir", str(tmp_path), "--skip-camera"], hooks=_ok_hooks())
    assert rc_empty == 1
    assert "vazio" in caplog.text


def test_check_env_camera_error_and_skip_camera(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_model(tmp_path)
    hooks = _ok_hooks(camera_error=OSError("sem webcam"))
    with caplog.at_level(logging.INFO, logger="check_env"):
        assert main(["--model-dir", str(tmp_path)], hooks=hooks) == 1
    assert "sem webcam" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="check_env"):
        assert main(["--model-dir", str(tmp_path), "--skip-camera"], hooks=hooks) == 0
    assert "pulada (--skip-camera)" in caplog.text


def test_check_env_reports_camera_fallback_index(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_model(tmp_path)
    with caplog.at_level(logging.INFO, logger="check_env"):
        rc = main(
            ["--model-dir", str(tmp_path), "--camera", "1"],
            hooks=_ok_hooks(resolved_camera=0),
        )
    assert rc == 0
    assert "[fallback de 1]" in caplog.text
