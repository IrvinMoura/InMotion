"""Testes das features dos gestos de corpo (spec 008): alcance frontal e base R7."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.gestures.features import (
    BaselineParams,
    BodyBaseline,
    FrameContext,
    build_context,
    crouch_depth,
    crouch_drop,
    forward_reach,
    jump_height,
    knee_lift,
    lean_forward,
)
from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import PoseLandmark, Side
from tests.helpers import ARM_HANGING, ARM_STRAIGHT_FORWARD, make_body

GEOMETRY = GeometryConfig()
TOL = 1e-9

coord = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
vec3 = st.tuples(coord, coord, coord)


def _with_arm_world(
    shoulder: tuple[float, float, float],
    elbow: tuple[float, float, float],
    wrist: tuple[float, float, float],
) -> PoseFrame:
    """Pose com o braço direito nos pontos dados (world); o quadril fica bem abaixo de tudo."""
    base = make_body()
    assert base.world is not None
    world = np.array(base.world)
    world[PoseLandmark.RIGHT_SHOULDER, :3] = shoulder
    world[PoseLandmark.RIGHT_ELBOW, :3] = elbow
    world[PoseLandmark.RIGHT_WRIST, :3] = wrist
    world[PoseLandmark.LEFT_HIP, :3] = (0.1, 10.0, 0.0)  # gate do pulso sempre aberto
    world[PoseLandmark.RIGHT_HIP, :3] = (-0.1, 10.0, 0.0)
    return PoseFrame(t=0.0, image_size=base.image_size, landmarks=base.landmarks, world=world)


@given(s=vec3, e=vec3, w=vec3)
def test_forward_reach_bounded_and_one_iff_straight_to_camera(
    s: tuple[float, float, float],
    e: tuple[float, float, float],
    w: tuple[float, float, float],
) -> None:
    """|r| ≤ 1 sempre; r = 1 quando o braço é reto ao longo de −z (a câmera)."""
    r = forward_reach(_with_arm_world(s, e, w), Side.RIGHT, GEOMETRY)
    if r is not None:
        assert abs(r) <= 1.0 + TOL
    # Braço reto apontando para a câmera: ombro, cotovelo e pulso colineares ao longo de −z.
    d = 0.5
    straight = _with_arm_world(s, (s[0], s[1], s[2] - d / 2), (s[0], s[1], s[2] - d))
    r_straight = forward_reach(straight, Side.RIGHT, GEOMETRY)
    assert r_straight is not None
    assert math.isclose(r_straight, 1.0, abs_tol=TOL)


@given(d=st.floats(min_value=0.1, max_value=1.0), off=st.floats(min_value=0.05, max_value=0.5))
def test_forward_reach_bent_arm_is_below_one(d: float, off: float) -> None:
    """Cotovelo fora do segmento: r = d / (2·sqrt((d/2)² + off²)) < 1."""
    pose = _with_arm_world((0.0, 0.0, 0.0), (0.0, off, -d / 2), (0.0, 0.0, -d))
    r = forward_reach(pose, Side.RIGHT, GEOMETRY)
    assert r is not None
    assert math.isclose(r, d / (2.0 * math.hypot(d / 2.0, off)), abs_tol=1e-9)
    assert r < 1.0


def test_forward_reach_none_without_world() -> None:
    assert forward_reach(make_body(with_world=False), Side.RIGHT, GEOMETRY) is None


def test_forward_reach_none_when_arm_invisible() -> None:
    low = make_body(right_arm=ARM_STRAIGHT_FORWARD)
    lm = np.array(low.landmarks)
    lm[PoseLandmark.RIGHT_WRIST, 3] = 0.1
    pose = PoseFrame(t=0.0, image_size=low.image_size, landmarks=lm, world=low.world)
    assert forward_reach(pose, Side.RIGHT, GEOMETRY) is None


def test_forward_reach_left_and_right_are_independent() -> None:
    pose = make_body(right_arm=ARM_STRAIGHT_FORWARD, left_arm=ARM_HANGING)
    assert forward_reach(pose, Side.RIGHT, GEOMETRY) == pytest.approx(1.0)
    assert forward_reach(pose, Side.LEFT, GEOMETRY) == pytest.approx(0.0)


def test_wrist_below_hip_gates_to_zero() -> None:
    """Pulso abaixo do quadril: 0,0 mesmo com o antebraço apontando para a câmera."""
    pose = _with_arm_world((0.0, 20.0, 0.0), (0.0, 20.0, -0.25), (0.0, 20.0, -0.5))
    assert forward_reach(pose, Side.RIGHT, GEOMETRY) == 0.0


def _ctx(pose: PoseFrame, baseline: BodyBaseline, resting: bool = True) -> FrameContext:
    return build_context(pose, pose.t, GEOMETRY, baseline, resting)


def test_baseline_initial_is_mean_of_k() -> None:
    """A base inicial é a média de K amostras (não um frame só); antes disso, None."""
    base = BodyBaseline(BaselineParams(init_frames=4, rate=0.02))
    assert [base.update(h, True) for h in (10.0, 20.0, 30.0)] == [None, None, None]
    assert base.update(40.0, True) == pytest.approx(25.0)
    assert base.value == pytest.approx(25.0)


def test_baseline_ema_in_rest() -> None:
    """R7: b = (1-λ)·b + λ·h."""
    base = BodyBaseline(BaselineParams(init_frames=1, rate=0.1))
    assert base.update(100.0, True) == 100.0
    assert base.update(200.0, True) == pytest.approx(110.0)


def test_baseline_frozen_when_not_resting() -> None:
    """J2: fora do repouso a base não anda, nem conta amostra antes de ter K."""
    base = BodyBaseline(BaselineParams(init_frames=2, rate=0.5))
    base.update(100.0, True)
    assert base.update(500.0, False) is None
    assert base.update(100.0, True) == 100.0
    for _ in range(100):
        assert base.update(900.0, False) == 100.0
    assert base.update(float("nan"), True) == 100.0  # amostra inválida é ignorada


def test_baseline_reset_forgets() -> None:
    base = BodyBaseline(BaselineParams(init_frames=1))
    base.update(5.0, True)
    base.reset()
    assert base.value is None
    assert base.update(7.0, True) == 7.0


def test_baseline_params_validation() -> None:
    with pytest.raises(ValueError, match="init_frames"):
        BaselineParams(init_frames=0)
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="rate"):
            BaselineParams(rate=bad)


@given(k=st.floats(min_value=0.5, max_value=3.0), rise=st.floats(min_value=-0.5, max_value=0.8))
def test_jump_height_scale_invariant(k: float, rise: float) -> None:
    """J1: j = (b - h)/s não muda com a escala; vale ``rise`` (em larguras de ombro)."""
    size = (4096, 4096)
    base = BodyBaseline(BaselineParams(init_frames=1))
    y0 = 1000.0 * k
    _ctx(make_body(scale=k, hip_y_px=y0, size=size), base)  # base = y0
    pose = make_body(scale=k, hip_y_px=y0 - rise * 100.0 * k, size=size)
    ctx = _ctx(pose, base, resting=False)
    j = jump_height(ctx)
    assert j is not None
    assert j == pytest.approx(rise, abs=1e-9)
    assert crouch_depth(ctx) == pytest.approx(-rise, abs=1e-9)


def test_context_without_skeleton_or_baseline() -> None:
    base = BodyBaseline(BaselineParams(init_frames=2))
    ctx = _ctx(make_body(visibility=0.0), base)
    assert ctx.skeleton is None
    assert ctx.hip_y_px is None
    assert ctx.shoulder_px is None
    assert jump_height(ctx) is None
    assert crouch_depth(ctx) is None
    ctx = _ctx(make_body(), base)  # 1 amostra de 2: ainda sem base
    assert ctx.baseline_px is None
    assert ctx.hip_y_px == pytest.approx(300.0)
    assert ctx.shoulder_px == pytest.approx(100.0)
    assert jump_height(ctx) is None


@given(lean=st.floats(min_value=-0.6, max_value=1.2))
def test_lean_forward_recovers_synthetic_lean(lean: float) -> None:
    """A9 em profundidade: o ângulo sintético do tronco é recuperado (positivo = frente)."""
    assert lean_forward(make_body(lean=lean), GEOMETRY) == pytest.approx(lean, abs=1e-9)


def test_lean_forward_none_cases() -> None:
    assert lean_forward(make_body(with_world=False), GEOMETRY) is None
    assert lean_forward(make_body(visibility=0.0), GEOMETRY) is None


# --- Spec 017: crouch_drop (agachar pelos ombros) ---------------------------------------------


def _shoulder_ctx(pose: PoseFrame, sbase: BodyBaseline, resting: bool = True) -> FrameContext:
    hip_base = BodyBaseline(BaselineParams(init_frames=1))
    return build_context(pose, pose.t, GEOMETRY, hip_base, resting, shoulder_baseline=sbase)


@given(
    k=st.floats(min_value=0.5, max_value=3.0),
    drop=st.floats(min_value=-0.5, max_value=2.0),
    x=st.floats(min_value=1000.0, max_value=3000.0),
    y=st.floats(min_value=1500.0, max_value=3000.0),
)
def test_crouch_drop_invariant_to_translation_and_scale(
    k: float, drop: float, x: float, y: float
) -> None:
    """CA5: ``c = (ȳ − b_o)/s`` vale ``drop`` (em larguras de ombro) para qualquer k, x, y."""
    size = (4096, 4096)
    sbase = BodyBaseline(BaselineParams(init_frames=1))
    _shoulder_ctx(make_body(scale=k, hip_x_px=x, hip_y_px=y, size=size), sbase)  # b_o
    dropped = make_body(scale=k, hip_x_px=x, hip_y_px=y, shoulder_dy_px=drop * 100.0 * k, size=size)
    c = crouch_drop(_shoulder_ctx(dropped, sbase, resting=False))
    assert c is not None
    assert c == pytest.approx(drop, abs=1e-6)


def test_crouch_drop_zero_at_baseline() -> None:
    """CA5: na base, ``c ≈ 0``; sem base (menos de K amostras) é ``None``."""
    sbase = BodyBaseline(BaselineParams(init_frames=2))
    pose = make_body()
    assert crouch_drop(_shoulder_ctx(pose, sbase)) is None  # 1 amostra de 2
    c = crouch_drop(_shoulder_ctx(pose, sbase))
    assert c is not None
    assert c == pytest.approx(0.0, abs=1e-9)


def test_shoulder_fields_without_hips() -> None:
    """RF3/RF4: ``ȳ`` e ``s`` dos ombros saem em pixels mesmo sem quadris (esqueleto ``None``)."""
    sbase = BodyBaseline(BaselineParams(init_frames=2))
    hip_base = BodyBaseline(BaselineParams(init_frames=1))
    pose = make_body(hips_visible=False, shoulder_dy_px=10.0)
    ctx = build_context(pose, 0.0, GEOMETRY, hip_base, True, shoulder_baseline=sbase)
    assert ctx.skeleton is None
    assert ctx.hip_y_px is None
    assert ctx.shoulder_px is None
    assert ctx.shoulder_y_px == pytest.approx(300.0 - 150.0 + 10.0)  # quadril - tronco + descida
    assert ctx.shoulder_width_px == pytest.approx(100.0)
    assert ctx.shoulder_baseline_px is None  # 1 amostra de 2
    ctx = build_context(pose, 0.0, GEOMETRY, hip_base, True, shoulder_baseline=sbase)
    assert ctx.shoulder_baseline_px == pytest.approx(160.0)
    assert hip_base.value is None  # a base do quadril não anda sem esqueleto


def test_shoulder_fields_with_hips_match_skeleton_scale() -> None:
    base = BodyBaseline(BaselineParams(init_frames=1))
    sbase = BodyBaseline(BaselineParams(init_frames=1))
    ctx = build_context(make_body(), 0.0, GEOMETRY, base, True, shoulder_baseline=sbase)
    assert ctx.shoulder_width_px == pytest.approx(ctx.shoulder_px)
    assert ctx.shoulder_y_px == pytest.approx(150.0)
    assert ctx.shoulder_baseline_px == pytest.approx(150.0)


def test_shoulder_fields_none_when_shoulders_unusable() -> None:
    sbase = BodyBaseline(BaselineParams(init_frames=1))
    base = BodyBaseline(BaselineParams(init_frames=1))
    ctx = build_context(
        make_body(visibility=0.0), 0.0, GEOMETRY, base, True, shoulder_baseline=sbase
    )
    assert ctx.shoulder_y_px is None
    assert ctx.shoulder_width_px is None
    assert sbase.value is None  # amostra inválida não vira base
    tiny = make_body(scale=0.05)  # s = 5 px < min_shoulder_px
    ctx = build_context(tiny, 0.0, GEOMETRY, base, True, shoulder_baseline=sbase)
    assert ctx.shoulder_y_px is None
    assert ctx.shoulder_width_px is None


def test_shoulder_baseline_is_optional_and_frozen_when_not_resting() -> None:
    base = BodyBaseline(BaselineParams(init_frames=1))
    ctx = build_context(make_body(), 0.0, GEOMETRY, base, True)  # sem a segunda base
    assert ctx.shoulder_baseline_px is None
    assert ctx.shoulder_y_px == pytest.approx(150.0)
    sbase = BodyBaseline(BaselineParams(init_frames=1, rate=0.5))
    build_context(make_body(), 0.0, GEOMETRY, base, True, shoulder_baseline=sbase)
    moved = make_body(shoulder_dy_px=40.0)
    ctx = build_context(moved, 0.1, GEOMETRY, base, False, shoulder_baseline=sbase)
    assert ctx.shoulder_baseline_px == pytest.approx(150.0)  # congelada fora de repouso


# --- Spec 017 T7: base R7 com portão de desvio (contaminação na descida do agachar) ---


def test_baseline_ignores_samples_far_from_it() -> None:
    """Descendo para agachar (c ainda < θ_on conta como repouso), a base não pode seguir
    o quadril: só anda com |h − b|/s < rest_band. Sem isso, ao levantar, j > θ_on do pulo."""
    base = BodyBaseline(BaselineParams(init_frames=1, rate=0.5, rest_band=0.15))
    base.update(300.0, resting=True, scale=100.0)
    assert base.value == pytest.approx(300.0)
    base.update(330.0, resting=True, scale=100.0)  # 0,30 s de desvio: ignorado
    assert base.value == pytest.approx(300.0)
    base.update(310.0, resting=True, scale=100.0)  # 0,10 s: anda (EMA)
    assert base.value == pytest.approx(305.0)


def test_baseline_without_scale_keeps_old_behaviour() -> None:
    base = BodyBaseline(BaselineParams(init_frames=1, rate=0.5))
    base.update(300.0, resting=True)
    base.update(400.0, resting=True)
    assert base.value == pytest.approx(350.0)


def test_rest_band_validation() -> None:
    assert BaselineParams().rest_band == pytest.approx(0.15)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="rest_band"):
            BaselineParams(rest_band=bad)


# --- Spec 018: elevação do joelho (RF1, M1) ---

NOT_YET = pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="spec 018 T1")


def _knee_ctx(
    lift_left: float = 0.0,
    lift_right: float = 0.0,
    *,
    scale: float = 1.0,
    dx: float = 0.0,
    dy: float = 0.0,
    hip_dy_after: float = 0.0,
) -> FrameContext:
    """15 frames de repouso (criam a base R7 dos joelhos) e um frame de teste."""
    baselines = (BodyBaseline(BaselineParams()), BodyBaseline(BaselineParams()))
    hip = BodyBaseline(BaselineParams())
    ctx: FrameContext | None = None
    for i in range(16):
        last = i == 15
        pose = make_body(
            t=i / 30.0,
            scale=scale,
            hip_x_px=320.0 + dx,
            hip_y_px=300.0 + dy + (hip_dy_after if last else 0.0),
            knee_lift_left=lift_left if last else 0.0,
            knee_lift_right=lift_right if last else 0.0,
        )
        ctx = build_context(pose, pose.t, GEOMETRY, hip, not last, None, knee_baselines=baselines)
    assert ctx is not None
    return ctx


@NOT_YET
@given(
    lift=st.floats(0.3, 1.2),
    scale=st.floats(0.5, 2.0),
    dx=st.floats(-60.0, 60.0),
    dy=st.floats(-60.0, 60.0),
)
def test_knee_lift_invariant_to_translation_and_scale(
    lift: float, scale: float, dx: float, dy: float
) -> None:
    """M1: transladar ou escalar o esqueleto não muda `k` (vale a elevação sintética)."""
    ctx = _knee_ctx(lift_left=lift, scale=scale, dx=dx, dy=dy)
    assert knee_lift(ctx, Side.LEFT) == pytest.approx(lift, abs=1e-6)
    assert knee_lift(ctx, Side.RIGHT) == pytest.approx(0.0, abs=1e-6)


@NOT_YET
def test_jump_keeps_knee_lift_near_zero() -> None:
    """M1: no pulo o quadril e os joelhos sobem juntos: `k` fica ≈ 0 nos dois lados."""
    ctx = _knee_ctx(hip_dy_after=-60.0)  # quadril 0,6 larguras de ombro acima da base
    assert knee_lift(ctx, Side.LEFT) == pytest.approx(0.0, abs=1e-6)
    assert knee_lift(ctx, Side.RIGHT) == pytest.approx(0.0, abs=1e-6)
