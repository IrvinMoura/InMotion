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
    assert {s.gesture for s in right} >= {"attack", "use", "jump", "crouch", None}
    # spec 017: walk fora do perfil do Minecraft; inclinar não pode apertar nada
    lean = next(s for s in right if s.label == "lean")
    assert lean.gesture is None
    assert lean.expected == 0


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


def test_step_count_so_far_counts_only_current_step_key() -> None:
    from tools.guided import step_count

    sched = Schedule(STEPS, prep_s=2.0)
    presses = [(3.0, "mouse_left"), (8.5, "mouse_left"), (9.0, "space"), (10.0, "mouse_left")]
    assert step_count(sched, 1, presses, KEYS) == 2  # só mouse_left a partir de 8,0 s
    assert step_count(sched, 0, presses, KEYS) is None  # passo "parado" não tem gesto


def test_gesture_totals_count_presses_per_gesture() -> None:
    from tools.guided import gesture_totals

    presses = [(1.0, "mouse_left"), (2.0, "mouse_left"), (3.0, "space"), (4.0, "w")]
    assert gesture_totals(presses, KEYS) == {"attack": 2, "jump": 1}


def test_render_frame_shows_counter() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    base = render_frame(frame, "3 PULOS", 3.0, Phase.RUN, "ATIVO", [], [])
    with_counter = render_frame(frame, "3 PULOS", 3.0, Phase.RUN, "ATIVO", [], [], counter=(2, 3))
    assert with_counter.sum() > base.sum()


def test_key_held_into_a_step_counts_as_extra() -> None:
    """Bug: `w` apertado antes do "parado" e segurado durante ele passava como OK."""
    sched = Schedule(STEPS, prep_s=2.0)
    presses = [(1.0, "w")]  # no aviso do passo 0, nunca solto
    releases: list[tuple[float, str]] = []
    rest, _, _ = score_steps(sched, presses, KEYS, grace_s=0.5, releases=releases)
    assert rest.ok is False
    assert rest.extras == ("w",)
    # solto antes do passo começar: não conta
    rest2, _, _ = score_steps(sched, presses, KEYS, grace_s=0.5, releases=[(1.5, "w")])
    assert rest2.ok is True


def test_crouch_script_for_spec_017_t7() -> None:
    """Roteiro da T7: pulos (agachadinha), 3 agachamentos e 3 inclinações sem tecla."""
    from tools.guided import SCRIPTS

    steps = SCRIPTS["crouch"]("right")
    by_label = {s.label: s for s in steps}
    assert by_label["jump"].gesture == "jump"
    assert by_label["jump"].expected == 3
    assert by_label["crouch"].gesture == "crouch"
    assert by_label["crouch"].expected == 3
    # inclinar para frente não pode agachar nem apertar nada (RF3b; walk fora do perfil)
    assert by_label["lean"].gesture is None
    assert by_label["lean"].expected == 0
    assert steps[0].expected is None  # posicionar-se não pontua
    assert SCRIPTS["full"]("left") == default_steps("left")


def test_cli_accepts_script_choice() -> None:
    from tools.guided import _build_parser

    assert _build_parser().parse_args([]).script == "full"
    assert _build_parser().parse_args(["--script", "crouch"]).script == "crouch"


def test_cli_set_overrides_are_repeatable() -> None:
    from tools.guided import _build_parser

    args = _build_parser().parse_args(["--set", "handedness=left", "--set", "camera.index=0"])
    assert args.set == ["handedness=left", "camera.index=0"]
    assert _build_parser().parse_args([]).set == []


def test_march_script_for_spec_018_t0() -> None:
    """Coleta da 018 T0: marcha lenta e rápida (sem nota até existir o gesto), pausas
    paradas, pulos e agachamentos pontuados."""
    from tools.guided import SCRIPTS

    steps = SCRIPTS["march"]("right")
    by_label = {s.label: s for s in steps}
    assert len(by_label) == len(steps)  # rótulos únicos (viram LabelInterval)
    # desde a T8 a marcha é pontuada (ver test_march_script_scores_walk_and_sprint_now)
    assert by_label["march_slow"].expected == 1
    assert by_label["march_fast"].expected == 1
    assert by_label["jump"].expected == 3
    assert by_label["crouch"].expected == 2
    assert all(s.expected == 0 for s in steps if s.label.startswith("rest"))


def test_unscored_data_steps_are_labelled_too() -> None:
    """Bug da 018 T0: a marcha (sem nota) ficou sem rótulo na gravação. Todo passo com
    gesto ou nota vira rótulo; só o posicionar-se (sem gesto e sem nota) fica de fora."""
    steps = (
        GuidedStep("setup", "POSICIONE-SE", 2.0, gesture=None, expected=None),
        GuidedStep("march_slow", "MARCHE", 4.0, gesture="walk", expected=None),
        GuidedStep("rest", "PARADO", 2.0, gesture=None, expected=0),
    )
    labels = step_labels(Schedule(steps, prep_s=1.0))
    assert [lb.gesture for lb in labels] == ["march_slow", "rest"]


def test_at_least_and_allowed_keys_for_march_steps() -> None:
    """018 T8: marcha conta "pelo menos 1" e, na rápida, o `w` junto com o `ctrl` é permitido;
    `ctrl` na marcha lenta continua sendo extra (reprova)."""
    steps = (
        GuidedStep("march_slow", "MARCHE", 4.0, gesture="walk", expected=1, at_least=True),
        GuidedStep(
            "march_fast",
            "CORRA",
            4.0,
            gesture="sprint",
            expected=1,
            at_least=True,
            allow=("walk",),
        ),
    )
    keys = {"walk": "w", "sprint": "ctrl"}
    sched = Schedule(steps, prep_s=1.0)  # janelas: [1, 5) e [6, 10)
    presses = [(1.5, "w"), (3.0, "w"), (6.5, "w"), (7.0, "ctrl")]
    slow, fast = score_steps(sched, presses, keys, grace_s=0.5)
    assert (slow.got, slow.ok) == (2, True)
    assert (fast.got, fast.extras, fast.ok) == (1, (), True)
    slow_bad, _ = score_steps(sched, [(1.5, "w"), (2.0, "ctrl")], keys, grace_s=0.5)
    assert slow_bad.ok is False


def test_march_script_scores_walk_and_sprint_now() -> None:
    from tools.guided import SCRIPTS

    by_label = {s.label: s for s in SCRIPTS["march"]("right")}
    assert by_label["march_slow"].gesture == "walk"
    assert by_label["march_slow"].at_least
    assert by_label["march_fast"].gesture == "sprint"
    assert by_label["march_fast"].allow == ("walk",)
