"""Valida o workflow de CI (spec 015, RF7/CA2) lendo o YAML como texto, sem PyYAML."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

DOD_COMMANDS = [
    "ruff check .",
    "ruff format --check .",
    "mypy",
    'pytest -m "not hardware and not slow"',
]


def _text() -> str:
    """Lê o workflow sem os comentários."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("#"))


def _run_commands() -> list[str]:
    """Extrai o valor de cada `run:` de uma linha."""
    return [m.strip() for m in re.findall(r"^\s*-?\s*run:\s*(.+)$", _text(), re.MULTILINE)]


def test_workflow_file_exists() -> None:
    assert WORKFLOW.is_file()


def test_runs_on_windows_with_python_310() -> None:
    text = _text()
    assert re.search(r"^\s*runs-on:\s*windows-latest\s*$", text, re.MULTILINE)
    assert re.search(r"""python-version:\s*["']3\.10["']""", text)


def test_triggers_on_push_and_pull_request() -> None:
    text = _text()
    assert re.search(r"^\s*push:", text, re.MULTILINE)
    assert re.search(r"^\s*pull_request:", text, re.MULTILINE)


def test_pip_cache_covers_requirement_files() -> None:
    text = _text()
    assert re.search(r"cache:\s*pip", text)
    assert "requirements.txt" in text
    assert "requirements-dev.txt" in text


def test_actions_are_pinned_and_permissions_read_only() -> None:
    text = _text()
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses
    assert all(re.search(r"@v\d+", u) for u in uses)
    assert re.search(r"^\s*contents:\s*read\s*$", text, re.MULTILINE)


@pytest.mark.parametrize("command", DOD_COMMANDS)
def test_workflow_runs_dod_commands(command: str) -> None:
    assert command in _run_commands()


def test_installs_dev_requirements() -> None:
    assert "pip install -r requirements-dev.txt" in _run_commands()


def test_no_step_runs_only_hardware() -> None:
    for command in _run_commands():
        if command.startswith("pytest"):
            assert "-m hardware" not in command
            assert "not hardware and not slow" in command
