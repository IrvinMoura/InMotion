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
