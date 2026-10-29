"""Spec 002 T8 — overlay mínimo (adaptador)."""

import numpy as np
import pytest

from inmotion.overlay import render
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
