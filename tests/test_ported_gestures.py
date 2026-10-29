"""Spec 002 T4 — gestos do protótipo portados para PoseFrame (CA3)."""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.gestures.legacy import LegacyJump, LegacyPunch
from inmotion.input.backend import MOUSE_BUTTONS
from inmotion.safety import TrackedInput
from inmotion.types import PoseFrame
from inmotion.vision.landmarks import PoseLandmark as L
from inmotion.vision.landmarks import Side
from tests.fakes import FakeInputBackend, make_pose

_SIDE_POINTS = {
    Side.RIGHT: (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST, L.RIGHT_HIP),
    Side.LEFT: (L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST, L.LEFT_HIP),
}


def make_input() -> tuple[TrackedInput, FakeInputBackend]:
    fake = FakeInputBackend()
    return TrackedInput(fake, MOUSE_BUTTONS), fake


def legacy_oracle(
    s: tuple[float, float],
    e: tuple[float, float],
    w: tuple[float, float],
    hip_y: float,
    threshold: float = 0.60,
) -> bool:
    """Fórmula de ``actions/atacar.py`` reescrita em pixels (oráculo do CA3)."""
    forearm = math.hypot(e[0] - w[0], e[1] - w[1])
    arm = math.hypot(s[0] - e[0], s[1] - e[1])
    if arm == 0:
        arm = 1
    return bool(w[1] < hip_y and forearm / arm < threshold)


coord = st.floats(min_value=1, max_value=1000, allow_nan=False)
point = st.tuples(coord, coord)


@given(
    size=st.tuples(st.integers(100, 1920), st.integers(100, 1080)),
    side=st.sampled_from(list(Side)),
    s=point,
    e=point,
    w=point,
    hip_y=coord,
)
@settings(max_examples=200)
def test_punch_ratio_matches_legacy_formula(
    size: tuple[int, int],
    side: Side,
    s: tuple[float, float],
    e: tuple[float, float],
    w: tuple[float, float],
    hip_y: float,
) -> None:
    width, height = size
    clip = lambda p: (min(p[0], width - 1.0), min(p[1], height - 1.0))  # noqa: E731
    s, e, w = clip(s), clip(e), clip(w)
    hip_y = min(hip_y, height - 1.0)
    si, ei, wi, hi = _SIDE_POINTS[side]
    pose = make_pose({si: s, ei: e, wi: w, hi: (width / 2, hip_y)}, size=size)
    out, fake = make_input()
    LegacyPunch(side, "mouse_left").update(pose, out)
    # Evita comparar exatamente em cima do limiar (erro de ponto flutuante da conversão).
    forearm = math.hypot(e[0] - w[0], e[1] - w[1])
    arm = math.hypot(s[0] - e[0], s[1] - e[1]) or 1.0
    if abs(forearm / arm - 0.60) < 1e-6 or abs(w[1] - hip_y) < 1e-6:
        return
    assert (("press", "mouse_left") in fake.events) == legacy_oracle(s, e, w, hip_y)


def test_punch_press_release_and_no_repeat() -> None:
    out, fake = make_input()
    punch = LegacyPunch(Side.RIGHT, "mouse_left")
    folded = make_pose(
        {
            L.RIGHT_SHOULDER: (300, 100),
            L.RIGHT_ELBOW: (300, 200),
            L.RIGHT_WRIST: (300, 350),
            L.RIGHT_HIP: (300, 400),
        }
    )
    short = make_pose(
        {
            L.RIGHT_SHOULDER: (300, 100),
            L.RIGHT_ELBOW: (300, 200),
            L.RIGHT_WRIST: (300, 230),
            L.RIGHT_HIP: (300, 400),
        }
    )
    punch.update(folded, out)  # razão 1,5 -> nada
    assert fake.events == []
    punch.update(short, out)  # razão 0,3 -> aperta
    punch.update(short, out)  # sem repetir
    assert fake.events == [("press", "mouse_left")]
    punch.update(folded, out)
    assert fake.events == [("press", "mouse_left"), ("release", "mouse_left")]


def test_punch_requires_wrist_above_hip() -> None:
    out, fake = make_input()
    pose = make_pose(
        {
            L.LEFT_SHOULDER: (300, 100),
            L.LEFT_ELBOW: (300, 200),
            L.LEFT_WRIST: (300, 230),
            L.LEFT_HIP: (300, 150),  # pulso abaixo do quadril
        }
    )
    LegacyPunch(Side.LEFT, "mouse_right").update(pose, out)
    assert fake.events == []


def test_punch_zero_upper_arm_uses_one() -> None:
    out, fake = make_input()
    pose = make_pose(
        {
            L.RIGHT_SHOULDER: (300, 100),
            L.RIGHT_ELBOW: (300, 100),
            L.RIGHT_WRIST: (300.2, 100),
            L.RIGHT_HIP: (300, 400),
        }
    )
    LegacyPunch(Side.RIGHT, "mouse_left").update(pose, out)
    assert fake.events == [("press", "mouse_left")]


def test_punch_reset_forgets_active_state() -> None:
    out, fake = make_input()
    punch = LegacyPunch(Side.RIGHT, "mouse_left")
    short = make_pose(
        {
            L.RIGHT_SHOULDER: (300, 100),
            L.RIGHT_ELBOW: (300, 200),
            L.RIGHT_WRIST: (300, 230),
            L.RIGHT_HIP: (300, 400),
        }
    )
    punch.update(short, out)
    out.release_all()
    punch.reset()
    punch.update(short, out)
    assert fake.events.count(("press", "mouse_left")) == 2


def test_punch_invalid_params() -> None:
    with pytest.raises(ValueError, match="ratio_threshold"):
        LegacyPunch(Side.RIGHT, "mouse_left", ratio_threshold=0.0)
    with pytest.raises(ValueError, match="key"):
        LegacyPunch(Side.RIGHT, "")


def hip_pose(y: float, size: tuple[int, int] = (640, 720), t: float = 0.0) -> PoseFrame:
    return make_pose({L.LEFT_HIP: (300, y), L.RIGHT_HIP: (340, y)}, size=size, t=t)


def test_jump_rise_press_and_release() -> None:
    out, fake = make_input()
    jump = LegacyJump()  # rise = 50/720, H = 720 -> 50 px
    jump.update(hip_pose(400), out)  # calibra
    jump.update(hip_pose(360), out)  # sobe 40 < 50
    assert fake.events == []
    jump.update(hip_pose(340), out)  # sobe 60 > 50
    jump.update(hip_pose(330), out)
    assert fake.events == [("press", "space")]
    jump.update(hip_pose(395), out)  # voltou
    assert fake.events == [("press", "space"), ("release", "space")]


def test_jump_threshold_scales_with_image_height() -> None:
    out, fake = make_input()
    jump = LegacyJump()
    size = (320, 360)  # H = 360 -> rise = 25 px
    jump.update(hip_pose(200, size), out)
    jump.update(hip_pose(180, size), out)  # sobe 20 < 25
    assert fake.events == []
    jump.update(hip_pose(170, size), out)  # sobe 30 > 25
    assert fake.events == [("press", "space")]


def test_jump_recalibrates_on_release_within_settle() -> None:
    out, fake = make_input()
    jump = LegacyJump()  # settle = 20 px em H = 720
    jump.update(hip_pose(400), out)
    jump.update(hip_pose(300), out)  # pulo
    jump.update(hip_pose(390), out)  # solta; |390-400| = 10 < 20 -> base = 390
    # Nova base 390: sobe de 390 para 345 (45 < 50) não pula; com a base antiga (400) sim.
    jump.update(hip_pose(345), out)
    assert fake.events == [("press", "space"), ("release", "space")]


def test_jump_does_not_recalibrate_outside_settle() -> None:
    out, fake = make_input()
    jump = LegacyJump()
    jump.update(hip_pose(400), out)
    jump.update(hip_pose(300), out)
    jump.update(hip_pose(370), out)  # solta; |370-400| = 30 >= 20 -> base continua 400
    jump.update(hip_pose(345), out)  # 55 acima da base antiga -> pula de novo
    assert fake.events == [
        ("press", "space"),
        ("release", "space"),
        ("press", "space"),
    ]


def test_jump_reset_recalibrates_on_next_frame() -> None:
    out, fake = make_input()
    jump = LegacyJump()
    jump.update(hip_pose(400), out)
    jump.update(hip_pose(300), out)
    out.release_all()
    jump.reset()
    jump.update(hip_pose(300), out)  # vira a nova base, não pula
    jump.update(hip_pose(300), out)
    space = [ev for ev in fake.events if ev[1] == "space"]
    assert space == [("press", "space"), ("release", "space")]  # só o release_all()


def test_jump_invalid_params() -> None:
    with pytest.raises(ValueError, match="rise"):
        LegacyJump(rise=0.0)
    with pytest.raises(ValueError, match="settle"):
        LegacyJump(settle=float("nan"))
    with pytest.raises(ValueError, match="key"):
        LegacyJump(key="")
