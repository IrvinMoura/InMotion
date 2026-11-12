"""Câmera pela cabeça (spec 014): ResponseCurve (T2, CA1: C1-C4), head_pose e HeadLook."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.gestures.head_look import (
    HeadLook,
    HeadLookParams,
    HeadPose,
    ResponseCurve,
    head_pose,
)
from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import PoseLandmark
from tests.helpers import make_body, with_face

_pos = st.floats(min_value=1e-3, max_value=1e4, allow_nan=False)
_curves = st.builds(
    ResponseCurve,
    gain=_pos,
    dead_zone=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    gamma=st.floats(min_value=1.0, max_value=3.0, allow_nan=False),
    v_max=_pos,
)
_xs = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False)


@given(_curves, _xs, _xs)
def test_response_curve_properties(curve: ResponseCurve, x: float, y: float) -> None:
    v = curve(x)
    assert curve(-x) == -v  # C1
    if abs(x) <= curve.dead_zone:
        assert v == 0.0  # C2
    assert abs(v) <= curve.v_max  # C4
    lo, hi = sorted((abs(x), abs(y)))
    assert abs(curve(lo)) <= abs(curve(hi))  # C3


@given(_curves, st.floats(min_value=1e-12, max_value=1e-6))
def test_continuity_at_dead_zone(curve: ResponseCurve, eps: float) -> None:
    d = curve.dead_zone
    assert curve(d) == 0.0
    assert abs(curve(d + eps)) <= curve.gain * eps**curve.gamma + 1e-9


def test_known_values() -> None:
    c = ResponseCurve(gain=1500.0, dead_zone=0.08, gamma=1.0, v_max=1200.0)
    assert c(0.18) == pytest.approx(150.0)
    assert c(-0.18) == pytest.approx(-150.0)
    assert c(0.05) == 0.0
    assert c(5.0) == 1200.0
    assert ResponseCurve()(0.08 + 0.1) == pytest.approx(1500.0 * 0.1**1.5)


def test_non_finite_inputs() -> None:
    c = ResponseCurve()
    assert c(math.nan) == 0.0
    assert c(math.inf) == c.v_max
    assert c(-math.inf) == -c.v_max


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gain": 0.0},
        {"gain": -1.0},
        {"dead_zone": -0.01},
        {"gamma": 0.99},
        {"v_max": 0.0},
        {"gain": math.nan},
        {"dead_zone": math.inf},
        {"gamma": math.inf},
        {"v_max": math.inf},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=r"deve|devem"):
        ResponseCurve(**kwargs)


# ---------------------------------------------------------------------------
# Spec 014 T1: orientação da cabeça (RF1, C6) e HeadLook (RF2, C5, RF5, RF6).
# ---------------------------------------------------------------------------

GEOMETRY = GeometryConfig()
MIN_EAR_PX = HeadLookParams().min_ear_px
# Curva linear sem zona morta nem limite prático: v = 100·x px/s, fácil de integrar à mão.
LINEAR = ResponseCurve(gain=100.0, dead_zone=0.0, gamma=1.0, v_max=1e6)
STILL = HeadPose(0.0, 0.0)


def _face(
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    center_px: tuple[float, float] = (320.0, 120.0),
    ear_px: float = 40.0,
) -> PoseFrame:
    return with_face(make_body(), yaw=yaw, pitch=pitch, center_px=center_px, ear_px=ear_px)


def _settled(params: HeadLookParams, hz: float = 30.0) -> tuple[HeadLook, float]:
    """HeadLook com o neutro em (0, 0) já medido; devolve também o último ``t`` usado."""
    look = HeadLook(params)
    t = 0.0
    while t < params.neutral_s + 1e-9:
        assert look.update(STILL, t) == (0, 0)
        t += 1.0 / hz
    return look, t - 1.0 / hz


def _look(params: HeadLookParams, head: HeadPose, hz: float, seconds: float) -> tuple[int, int]:
    """Neutro em (0, 0), depois ``head`` por ``seconds`` a ``hz``: soma de (dx, dy)."""
    look, t = _settled(params, hz)
    total = [0, 0]
    for i in range(1, round(seconds * hz) + 1):
        dx, dy = look.update(head, t + i / hz)
        total[0] += dx
        total[1] += dy
    return total[0], total[1]


@given(
    st.floats(min_value=-0.8, max_value=0.8),
    st.floats(min_value=-0.5, max_value=0.5),
    st.floats(min_value=100.0, max_value=540.0),
    st.floats(min_value=80.0, max_value=400.0),
    st.floats(min_value=15.0, max_value=120.0),
)
def test_head_pose_scale_and_translation_invariant(
    yaw: float, pitch: float, cx: float, cy: float, ear_px: float
) -> None:
    """C6 (CA5): só o deslocamento relativo às orelhas conta, em qualquer posição e escala."""
    pose = _face(yaw=yaw, pitch=pitch, center_px=(cx, cy), ear_px=ear_px)
    head = head_pose(pose, GEOMETRY, MIN_EAR_PX)
    assert head is not None
    assert head.yaw == pytest.approx(yaw, abs=1e-9)
    assert head.pitch == pytest.approx(pitch, abs=1e-9)


def test_head_pose_sign_conventions() -> None:
    """Nariz à esquerda da imagem (não espelhada, ADR 0007) = cabeça para a direita da
    pessoa (``yaw > 0``); nariz para cima = ``pitch > 0`` (T0: decisão 3)."""
    base = _face()
    w, h = base.image_size
    lm = np.array(base.landmarks, copy=True)
    lm[PoseLandmark.NOSE, 0] -= 10.0 / w
    lm[PoseLandmark.NOSE, 1] -= 10.0 / h
    pose = PoseFrame(t=0.0, image_size=base.image_size, landmarks=lm, world=None)
    head = head_pose(pose, GEOMETRY, MIN_EAR_PX)
    assert head is not None
    assert head.yaw == pytest.approx(0.25)  # 10 px / 40 px entre as orelhas
    assert head.pitch == pytest.approx(0.25)


@pytest.mark.parametrize(
    "landmark",
    [
        PoseLandmark.NOSE,
        PoseLandmark.LEFT_EYE,
        PoseLandmark.RIGHT_EYE,
        PoseLandmark.LEFT_EAR,
        PoseLandmark.RIGHT_EAR,
    ],
)
def test_head_pose_none_when_face_hidden(landmark: PoseLandmark) -> None:
    base = _face(yaw=0.3)
    lm = np.array(base.landmarks, copy=True)
    lm[landmark, 3] = 0.0
    pose = PoseFrame(t=0.0, image_size=base.image_size, landmarks=lm, world=None)
    assert head_pose(pose, GEOMETRY, MIN_EAR_PX) is None


def test_head_pose_none_when_ears_too_close() -> None:
    """De perfil as orelhas se sobrepõem na imagem: a razão explodiria, então é ``None``."""
    assert head_pose(_face(ear_px=MIN_EAR_PX / 2.0), GEOMETRY, MIN_EAR_PX) is None


@pytest.mark.parametrize("hz", [30.0, 60.0, 47.0])
def test_fps_independent_displacement(hz: float) -> None:
    """C5 (CA2): 1 s com ``x`` constante dá ``v·1 s`` em qualquer FPS, ± 1 px."""
    params = HeadLookParams(yaw=LINEAR, pitch=LINEAR, pitch_enabled=True)
    dx, dy = _look(params, HeadPose(0.5, 0.3), hz, 1.0)
    assert abs(dx - 50) <= 1  # v = 100·0,5 px/s
    assert abs(dy - (-30)) <= 1  # cabeça para cima → ponteiro para cima (dy < 0)


def test_irregular_frame_times_keep_total_displacement() -> None:
    """C5 com ``Δt`` irregular: o resto fracionário não se perde."""
    look, t = _settled(HeadLookParams(yaw=LINEAR, pitch=LINEAR))
    start = t
    total = 0
    for i in range(200):
        t += 0.005 if i % 3 else 0.04
        total += look.update(HeadPose(0.2, 0.0), t)[0]
    assert abs(total - 100.0 * 0.2 * (t - start)) <= 1.0


def test_neutral_measured_after_reset() -> None:
    """RF2 (CA3): ``(0, 0)`` até ``neutral_s``; a postura do neutro não move; o ``reset``
    mede de novo."""
    params = HeadLookParams(yaw=LINEAR, pitch=LINEAR, pitch_enabled=True)
    look = HeadLook(params)
    tilted = HeadPose(0.4, -0.2)
    t = 0.0
    while t < 2.0 * params.neutral_s:
        assert look.update(tilted, t) == (0, 0)  # o neutro é a própria postura
        t += 1.0 / 30.0
    look.reset()
    t0 = t
    while t - t0 < params.neutral_s:
        assert look.update(STILL, t) == (0, 0)  # medindo de novo, nada se move
        t += 1.0 / 30.0
    moved = sum(look.update(tilted, t + i / 30.0)[0] for i in range(30))
    assert moved > 0  # agora o neutro é (0, 0) e a postura virada move


def test_lost_pose_zeroes_accumulator() -> None:
    """RF6: sem pose → ``(0, 0)`` e o resto zera; o tempo sem pose não vira ``Δt``."""
    look, t = _settled(HeadLookParams(yaw=LINEAR, pitch=LINEAR))
    assert look.update(HeadPose(0.29, 0.0), t + 1 / 30.0) == (0, 0)  # 0,97 px no resto
    assert look.update(None, t + 2 / 30.0) == (0, 0)
    assert look.update(HeadPose(0.29, 0.0), t + 2.0) == (0, 0)


def test_suppress_stops_both_axes() -> None:
    """RF5: agachar em curso para a câmera nos dois eixos e zera o resto."""
    look, t = _settled(HeadLookParams(yaw=LINEAR, pitch=LINEAR, pitch_enabled=True))
    for i in range(1, 61):
        assert look.update(HeadPose(1.0, 1.0), t + i / 30.0, suppress=True) == (0, 0)
    assert look.update(HeadPose(0.29, 0.0), t + 61 / 30.0) == (0, 0)  # sem Δt acumulado


def test_pitch_disabled_by_default() -> None:
    """T0 (decisão 2): o pitch fica desligado por padrão; só o yaw move."""
    assert HeadLookParams().pitch_enabled is False
    dx, dy = _look(HeadLookParams(yaw=LINEAR, pitch=LINEAR), HeadPose(0.5, 0.5), 30.0, 1.0)
    assert dx > 0
    assert dy == 0


def test_t0_defaults() -> None:
    """Parâmetros iniciais saídos da T0 (12/11/2026, decisão 4)."""
    params = HeadLookParams()
    assert params.yaw == ResponseCurve(gain=1200.0, dead_zone=0.15, gamma=1.5, v_max=800.0)
    assert params.neutral_s == 1.0


@pytest.mark.parametrize(
    "kwargs", [{"neutral_s": 0.0}, {"neutral_s": math.inf}, {"min_ear_px": 0.0}]
)
def test_invalid_head_look_params(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="deve"):
        HeadLookParams(**kwargs)


def test_neutral_is_the_median_of_the_window() -> None:
    """RF2: um pico isolado durante a medição não desloca o neutro (mediana)."""
    look = HeadLook(HeadLookParams(yaw=LINEAR, pitch=LINEAR))
    for i in range(31):
        look.update(HeadPose(0.9 if i == 5 else 0.1, 0.0), i / 30.0)
    assert look.neutral is not None
    assert look.neutral.yaw == pytest.approx(0.1)


def test_invert_pitch_flips_vertical() -> None:
    params = HeadLookParams(yaw=LINEAR, pitch=LINEAR, pitch_enabled=True, invert_pitch=True)
    assert _look(params, HeadPose(0.0, 0.3), 30.0, 1.0)[1] == pytest.approx(30, abs=1)
