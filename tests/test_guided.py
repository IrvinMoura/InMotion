"""Sessão guiada de validação (ferramenta visual, sem input real)."""

from __future__ import annotations

import numpy as np
import pytest

from tools.guided import (
    GuidedStep,
    Phase,
    Schedule,
    ascii_text,
    default_steps,
    render_frame,
    score_steps,
    step_labels,
)

STEPS = (
    GuidedStep("rest", "FIQUE PARADO", 4.0, gesture=None, expected=0),
    GuidedStep("attack", "SOCO DIREITO 3x", 6.0, gesture="attack", expected=3),
    GuidedStep("setup", "POSICIONE-SE", 2.0, gesture=None, expected=None),
)
KEYS = {"attack": "mouse_left", "jump": "space"}


def test_schedule_alternates_prep_and_run_then_done() -> None:
    sched = Schedule(STEPS, prep_s=2.0)
    assert sched.total_s == pytest.approx(2 + 4 + 2 + 6 + 2 + 2)
    assert sched.at(0.5) == (0, Phase.PREP, pytest.approx(1.5))
    assert sched.at(2.0) == (0, Phase.RUN, pytest.approx(4.0))
    assert sched.at(5.9)[:2] == (0, Phase.RUN)
    assert sched.at(6.5)[:2] == (1, Phase.PREP)
    assert sched.at(8.0)[:2] == (1, Phase.RUN)
    assert sched.at(17.0)[:2] == (2, Phase.RUN)
    assert sched.at(18.0)[:2] == (2, Phase.DONE)
    assert sched.run_window(1) == pytest.approx((8.0, 14.0))


def test_score_counts_expected_presses_and_flags_extras() -> None:
    sched = Schedule(STEPS, prep_s=2.0)
    presses = [
        (3.0, "w"),  # tecla indevida no "parado"
        (8.5, "mouse_left"),
        (10.0, "mouse_left"),
        (14.3, "mouse_left"),  # dentro da tolerância de latência após o fim
        (12.0, "space"),  # extra no passo do soco
        (16.5, "w"),  # passo sem pontuação: ignorado
    ]
    rest, attack, setup = score_steps(sched, presses, KEYS, grace_s=0.5)
    assert (rest.got, rest.extras, rest.ok) == (0, ("w",), False)
    assert (attack.got, attack.extras, attack.ok) == (3, ("space",), False)
    assert setup.ok is None  # não pontuado


def test_score_ok_when_exact() -> None:
    sched = Schedule(STEPS, prep_s=2.0)
    presses = [(9.0, "mouse_left"), (10.0, "mouse_left"), (11.0, "mouse_left")]
    rest, attack, _ = score_steps(sched, presses, KEYS, grace_s=0.5)
    assert rest.ok is True
    assert attack.ok is True


def test_step_labels_cover_run_windows_of_gesture_steps() -> None:
    labels = step_labels(Schedule(STEPS, prep_s=2.0))
    assert [(lb.gesture, lb.start, lb.end) for lb in labels] == [
        ("rest", 2.0, 6.0),
        ("attack", 8.0, 14.0),
    ]


def test_default_steps_follow_handedness() -> None:
    right = default_steps("right")
    left = default_steps("left")
    attack_r = next(s for s in right if s.gesture == "attack")
    attack_l = next(s for s in left if s.gesture == "attack")
    assert "DIREITO" in attack_r.title
    assert "ESQUERDO" in attack_l.title
    assert {s.gesture for s in right} >= {"attack", "use", "jump", "crouch", "walk", None}


def test_ascii_text_strips_accents_for_hershey() -> None:
    assert ascii_text("Inclinação à frente") == "Inclinacao a frente"


def test_render_frame_draws_big_canvas() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = render_frame(
        frame,
        title="FIQUE PARADO",
        countdown_s=3.2,
        phase=Phase.RUN,
        state_text="ATIVO",
        bars=[("attack", 0.4, 0.70, 0.55, False), ("walk", None, 0.30, 0.20, True)],
        last_keys=["W"],
        size=(1280, 720),
    )
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    assert out.any()
