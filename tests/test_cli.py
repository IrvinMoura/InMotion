"""Spec 002 T10 — CLI ``python -m inmotion``."""

import subprocess
import sys

EXPECTED_ARGS = ["--camera", "--camera-backend", "--log-level", "--log-file", "--no-overlay"]


def test_help_lists_arguments() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "inmotion", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0
    for arg in EXPECTED_ARGS:
        assert arg in proc.stdout
