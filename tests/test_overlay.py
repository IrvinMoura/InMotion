"""Spec 002 T8 — overlay mínimo (adaptador)."""

import numpy as np
import pytest

from inmotion.overlay import handedness_label, render
from inmotion.pipeline import StepResult
from inmotion.safety import SafetyState
from tests.fakes import make_pose

SIZE = (160, 120)  # (W, H)


def blank() -> np.ndarray:
    return np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)


def result_with_pose() -> StepResult:
    pose = make_pose({0: (80, 60)}, size=SIZE, t=1.0)
    return StepResult(t=1.0, fps=29.7, safety_state=SafetyState.ACTIVE, pose=pose)


def test_render_keeps_shape_and_dtype() -> None:
    img = blank()
    out = render(img, result_with_pose())
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out is not img


def test_render_without_pose() -> None:
    res = StepResult(t=0.0, fps=None, safety_state=SafetyState.LOST, pose=None)
    out = render(blank(), res)
    assert out.shape == (SIZE[1], SIZE[0], 3)
    assert out.any()  # o texto de estado foi desenhado


def test_render_draws_something_with_pose() -> None:
    out = render(blank(), result_with_pose())
    assert out[60, 80].any()  # ponto do nariz


@pytest.mark.parametrize("state", list(SafetyState))
def test_render_all_states(state: SafetyState) -> None:
    res = StepResult(t=0.0, fps=30.0, safety_state=state, pose=None)
    assert render(blank(), res).any()


def test_render_does_not_modify_input_or_result() -> None:
    img = blank()
    img.setflags(write=False)  # imagem somente leitura (view do FramePacket) também funciona
    res = result_with_pose()
    assert res.pose is not None
    landmarks_before = res.pose.landmarks.copy()
    render(img, res)
    assert not img.any()
    assert np.array_equal(res.pose.landmarks, landmarks_before)
    assert res.fps == 29.7


def test_render_draws_exit_zone_with_progress() -> None:
    from dataclasses import replace

    from inmotion.exit_zone import ExitZoneConfig, ExitZoneState

    cfg = ExitZoneConfig(x0=0.0, y0=0.5, x1=0.5, y1=1.0)
    base = StepResult(t=0.0, fps=None, safety_state=SafetyState.ACTIVE, pose=None)
    idle = render(blank(), replace(base, exit_zone=ExitZoneState(cfg, False, 0.0, False)))
    half = render(blank(), replace(base, exit_zone=ExitZoneState(cfg, True, 0.5, False)))
    zone = (slice(60, 120), slice(0, 80))  # [y0·H, y1·H) × [x0·W, x1·W)
    assert idle[zone].any()  # contorno do bloco
    assert half[zone].sum() > idle[zone].sum()  # barra de progresso preenchida


def test_handedness_label_names_the_mode_and_the_attacking_arm() -> None:
    """Spec 017 RF6: o texto diz Destro/Canhoto e qual braço ataca (ASCII: Hershey)."""
    assert handedness_label("right") == "Destro - ataca: braco direito"
    assert handedness_label("left") == "Canhoto - ataca: braco esquerdo"
    for text in (handedness_label("right"), handedness_label("left")):
        assert text.isascii()


def test_shows_handedness_label() -> None:
    """Sem lateralidade nada é desenhado na linha nova; com ela, cada modo tem o seu texto."""
    res = StepResult(t=0.0, fps=30.0, safety_state=SafetyState.ACTIVE, pose=None)
    row = (slice(52, 76), slice(0, SIZE[0]))  # abaixo do FPS (y = 20) e do estado (y = 42)
    plain = render(blank(), res)
    left = render(blank(), res, "left")
    right = render(blank(), res, "right")
    assert not plain[row].any()
    assert left[row].any()
    assert right[row].any()
    assert not np.array_equal(left[row], right[row])
    assert np.array_equal(plain[:52], left[:52])  # o resto do quadro não muda
