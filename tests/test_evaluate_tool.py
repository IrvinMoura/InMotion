"""Spec 011 T7 — ``tools/evaluate.py``: roteiro, relatório e gráficos (CA8)."""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from inmotion.metrics import StepExpectation
from inmotion.recording import LabelInterval
from tests.test_recording import make_recording
from tools.evaluate import detect_script, expectations, main
from tools.guided import GuidedStep, crouch_steps, default_steps

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"


def labels_like(
    steps: Sequence[GuidedStep], start: float = 10.0, gap: float = 2.0
) -> tuple[LabelInterval, ...]:
    """Rótulos com a duração exata de cada passo pontuado, separados por ``gap``."""
    out: list[LabelInterval] = []
    t = start
    for s in steps:
        if s.expected is None:
            continue
        out.append(LabelInterval(s.label, t, t + s.duration_s))
        t += s.duration_s + gap
    return tuple(out)


def test_expectations_skip_unscored_steps() -> None:
    exp = expectations(default_steps("right"))
    assert "setup" not in exp
    assert exp["punch_dominant"] == StepExpectation("attack", 3)
    assert exp["rest"] == StepExpectation(None, 0)


@pytest.mark.parametrize(("steps", "name"), [(default_steps, "full"), (crouch_steps, "crouch")])
def test_detect_script_uses_step_durations(
    steps: Callable[[str], tuple[GuidedStep, ...]], name: str
) -> None:
    """O passo "crouch" existe nos dois roteiros com durações diferentes: a duração desempata."""
    rec = make_recording(5, labels=labels_like(steps("right")))
    assert detect_script(rec, "right") == name


def test_detect_script_accepts_clipped_edges_only() -> None:
    labels = list(labels_like(crouch_steps("right")))
    first, last = labels[0], labels[-1]
    labels[0] = LabelInterval(first.gesture, first.end - 3.0, first.end)  # recorte no início
    labels[-1] = LabelInterval(last.gesture, last.start, last.start + 2.0)  # e no fim
    assert detect_script(make_recording(5, labels=tuple(labels)), "right") == "crouch"
    mid = labels[2]
    labels[2] = LabelInterval(mid.gesture, mid.start, mid.start + 2.0)  # meio encurtado
    assert detect_script(make_recording(5, labels=tuple(labels)), "right") is None


def test_unknown_labels_are_not_a_script() -> None:
    rec = make_recording(5, labels=(LabelInterval("dancar", 0.0, 0.1),))
    assert detect_script(rec, "right") is None
    assert detect_script(make_recording(5), "right") is None


@pytest.mark.slow
def test_report_and_plots_created(tmp_path: Path) -> None:
    """Smoke: uma sessão real recortada gera a tabela, os CSVs e (com matplotlib) gráficos."""
    shutil.copy(FIXTURES / "crouch_2m.npz", tmp_path / "crouch_2m.npz")
    out = tmp_path / "out"
    assert main(["--recordings", str(tmp_path / "*.npz"), "--out", str(out)]) == 0
    report = (out / "metrics.md").read_text(encoding="utf-8")
    assert "| crouch |" in report
    assert "crouch_2m | crouch | 2 m" in report
    assert "tremor parado" in report  # o recorte tem o passo "rest"
    assert (out / "metrics.csv").read_text(encoding="utf-8").startswith("session,")
    assert (out / "filters.csv").exists()
    if importlib.util.find_spec("matplotlib") is not None:
        assert (out / "confusion.png").stat().st_size > 0


def test_main_without_recordings_returns_one(tmp_path: Path) -> None:
    assert main(["--recordings", str(tmp_path / "*.npz"), "--out", str(tmp_path)]) == 1
