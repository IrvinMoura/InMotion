"""Spec 002 T5 / 008 T9 — Pipeline.step: segurança -> contexto -> gestos -> árbitro -> mapper."""

from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inmotion.config.schema import Binding, GameProfile
from inmotion.filters import FloatArray
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.base import Direction, GestureEvent, GestureKind, GestureParams, GestureState
from inmotion.gestures.body import BodyGesture, FeatureGesture
from inmotion.gestures.factory import BODY_GESTURE_NAMES, build_body_gestures
from inmotion.gestures.features import FrameContext
from inmotion.input.backend import MOUSE_BUTTONS
from inmotion.pipeline import Pipeline
from inmotion.safety import SafetyConfig, SafetyManager, SafetyState, TrackedInput
from inmotion.types import PoseFrame
from inmotion.vision.landmarks import PoseLandmark
from tests.fakes import FakeInputBackend, make_pose
from tests.helpers import ARM_HANGING, ARM_STRAIGHT_FORWARD, Arm, make_body

SPY_PARAMS = GestureParams(0.5, 0.25, Direction.ABOVE, debounce_frames=1)
SPY_PROFILE = GameProfile("teste", {"spy": Binding("space", GestureKind.HOLD)})


class SpyGesture(FeatureGesture):
    """Gesto de teste: sempre ativo; guarda o ``t`` e o x do landmark 0 de cada contexto."""

    def __init__(self) -> None:
        super().__init__("spy", GestureKind.HOLD, SPY_PARAMS, self._probe)
        self.updates: list[float] = []
        self.seen_x: list[float] = []
        self.resets = 0

    def _probe(self, ctx: FrameContext) -> float:
        self.updates.append(ctx.t)
        self.seen_x.append(float(ctx.pose.landmarks[0, 0]))
        return 1.0

    def reset(self) -> None:
        self.resets += 1
        super().reset()


def build(
    gestures: list[BodyGesture],
    profile: GameProfile = SPY_PROFILE,
    config: SafetyConfig | None = None,
    pose_filter: PoseFilter | None = None,
) -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    fake = FakeInputBackend()
    config = config or SafetyConfig(lost_timeout_s=0.2, min_track_s=0.0)
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), config)
    pipeline = Pipeline(safety, gestures, FpsMeter(window=30), pose_filter, profile=profile)
    return pipeline, safety, fake


def make_pipeline(spy: SpyGesture) -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    return build([spy])


def test_step_runs_gestures_only_when_active() -> None:
    spy = SpyGesture()
    pipeline, safety, fake = make_pipeline(spy)

    res = pipeline.step(make_pose({}, t=0.0), 0.0)
    assert res.safety_state is SafetyState.ACTIVE
    assert spy.updates == [0.0]

    safety.on_hotkeys(frozenset({"f8"}))  # pausa
    res = pipeline.step(make_pose({}, t=0.1), 0.1)
    assert res.safety_state is SafetyState.PAUSED
    assert spy.updates == [0.0]  # gesto não rodou
    assert fake.pressed == set()  # e a pausa soltou tudo


def test_step_without_pose_skips_gestures() -> None:
    spy = SpyGesture()
    pipeline, _, _ = make_pipeline(spy)
    pipeline.step(make_pose({}, t=0.0), 0.0)
    res = pipeline.step(None, 0.1)
    assert res.pose is None
    assert spy.updates == [0.0]
    assert res.safety_state is SafetyState.ACTIVE  # ainda dentro do timeout


def test_no_pose_tracking_ok_false_reaches_safety() -> None:
    spy = SpyGesture()
    pipeline, _, fake = make_pipeline(spy)
    assert pipeline.step(make_pose({}, t=0.0), 0.0).safety_state is SafetyState.ACTIVE
    assert ("press", "space") in fake.events
    # Sem pose por >= lost_timeout_s: SafetyManager recebe tracking_ok=False e solta tudo.
    assert pipeline.step(None, 0.1).safety_state is SafetyState.ACTIVE
    res = pipeline.step(None, 0.25)
    assert res.safety_state is SafetyState.LOST
    assert fake.pressed == set()


def test_gestures_reset_when_returning_to_active() -> None:
    spy = SpyGesture()
    pipeline, _, _ = make_pipeline(spy)
    pipeline.step(make_pose({}, t=0.0), 0.0)
    assert spy.resets == 0
    pipeline.step(None, 0.3)  # perde o tracking -> LOST
    pipeline.step(None, 0.4)
    assert spy.resets == 0
    res = pipeline.step(make_pose({}, t=0.5), 0.5)  # volta para ACTIVE
    assert res.safety_state is SafetyState.ACTIVE
    assert spy.resets == 1
    assert spy.updates == [0.0, 0.5]  # o reset vem antes do update
    pipeline.step(make_pose({}, t=0.6), 0.6)
    assert spy.resets == 1  # só na transição


def test_gestures_reset_when_resuming_from_pause() -> None:
    spy = SpyGesture()
    pipeline, safety, _ = make_pipeline(spy)
    pipeline.step(make_pose({}, t=0.0), 0.0)
    safety.on_hotkeys(frozenset({"f8"}))
    pipeline.step(make_pose({}, t=0.1), 0.1)
    safety.on_hotkeys(frozenset())
    safety.on_hotkeys(frozenset({"f8"}))  # retoma
    pipeline.step(make_pose({}, t=0.2), 0.2)
    assert spy.resets == 1


def test_watchdog_between_steps_resets_and_represses_held_gesture() -> None:
    spy = SpyGesture()
    pipeline, safety, fake = make_pipeline(spy)
    pipeline.step(make_pose({}, t=0.0), 0.0)
    assert fake.pressed == {"space"}
    safety.on_stall()
    assert not fake.pressed
    result = pipeline.step(make_pose({}, t=1.0), 1.0)
    assert spy.resets == 1
    assert result.pressed == frozenset({"space"})
    assert fake.events.count(("press", "space")) == 2


def test_pause_and_resume_between_steps_still_resets_gestures() -> None:
    spy = SpyGesture()
    pipeline, safety, fake = make_pipeline(spy)
    pipeline.step(make_pose({}, t=0.0), 0.0)
    safety.on_hotkeys(frozenset({"f8"}))
    safety.on_hotkeys(frozenset())
    safety.on_hotkeys(frozenset({"f8"}))
    pipeline.step(make_pose({}, t=0.1), 0.1)
    assert spy.resets == 1
    assert fake.pressed == {"space"}


def test_watchdog_requires_continuous_tracking_again() -> None:
    spy = SpyGesture()
    pipeline, safety, fake = build([spy], config=SafetyConfig(min_track_s=0.1))
    pipeline.step(make_pose({}, t=0.0), 0.0)
    pipeline.step(make_pose({}, t=0.11), 0.11)
    assert fake.pressed == {"space"}
    safety.on_stall()
    pipeline.step(make_pose({}, t=1.0), 1.0)
    assert fake.pressed == set()
    pipeline.step(make_pose({}, t=1.11), 1.11)
    assert fake.pressed == {"space"}


def test_step_result_carries_t_pose_and_fps() -> None:
    spy = SpyGesture()
    pipeline, _, _ = make_pipeline(spy)
    p0, p1 = make_pose({}, t=0.0), make_pose({}, t=0.05)
    r0 = pipeline.step(p0, 0.0)
    r1 = pipeline.step(p1, 0.05)
    assert r0.fps is None
    assert r0.pose is p0
    assert r0.t == 0.0
    assert r1.fps is not None
    assert abs(r1.fps - 20.0) < 1e-9
    assert r1.pose is p1


# ---------------------------------------------------------------------------
# Spec 006 T6 — pose_filter opcional, aplicado antes dos gestos.
# ---------------------------------------------------------------------------

FILTERED_X = 0.25


class ConstantTemporalFilter:
    """TemporalFilter de teste: troca toda amostra por ``FILTERED_X`` e conta os resets."""

    def __init__(self) -> None:
        self.resets = 0

    def update(self, x: FloatArray | float, t: float) -> FloatArray:
        return np.full_like(np.asarray(x, dtype=np.float64), FILTERED_X)

    def reset(self) -> None:
        self.resets += 1


def make_filtered_pipeline() -> tuple[Pipeline, SpyGesture, list[ConstantTemporalFilter]]:
    inner: list[ConstantTemporalFilter] = []

    def factory() -> ConstantTemporalFilter:
        inner.append(ConstantTemporalFilter())
        return inner[-1]

    gesture = SpyGesture()
    pipeline, _, _ = build([gesture], pose_filter=PoseFilter(factory))
    return pipeline, gesture, inner


def test_pose_filter_applied_before_gestures() -> None:
    pipeline, gesture, _ = make_filtered_pipeline()
    raw = make_pose({0: (64.0, 48.0)}, t=0.0)  # x normalizado = 0,1

    result = pipeline.step(raw, 0.0)

    assert gesture.seen_x == [FILTERED_X]  # o gesto recebeu a pose filtrada
    assert result.pose is not None
    assert float(result.pose.landmarks[0, 0]) == FILTERED_X  # e o resultado também
    assert float(raw.landmarks[0, 0]) == pytest.approx(0.1)  # a original não muda


def test_pose_filter_reset_on_lost() -> None:
    pipeline, _, inner = make_filtered_pipeline()
    pipeline.step(make_pose({}, t=0.0), 0.0)
    image_filter = inner[0]  # (o de world é resetado pelo PoseFilter quando a pose não tem world)
    assert image_filter.resets == 0

    pipeline.step(None, 0.1)  # perdeu a pose
    assert image_filter.resets == 1


# --- Bug do checklist 003 T8: pose "fantasma" ao sair do quadro deixa o pulo preso ---
# Ao sair do quadro, o MediaPipe ainda devolveu uma pose (pontos num canto, visibilidade
# baixa). Como a pose não era None, o app não entrou em LOST: o pulo viu os quadris "subirem"
# e, sem LOST, nada soltou a tecla nem recalibrou a base R7. Com os gestos da spec 008 o pulo
# é um tap (PULSE) sobre a base adaptativa; a asserção é a mesma: o espaço nunca é apertado.

DT = 1 / 30
SIZE = (640, 480)
HIP_BASE_Y = 300.0
HIP_UP_Y = 250.0  # 50 px acima da base = 0,5 largura de ombro (> theta_on do pulo, 0,25)
GHOST_VISIBILITY = 0.05
JUMP_PROFILE = GameProfile("teste", {"jump": Binding("space", GestureKind.TRIGGER)})


def _body(y: float, t: float, visibility: float = 1.0) -> PoseFrame:
    return make_body(t=t, hip_y_px=y, visibility=visibility)


def _jump_pipeline() -> tuple[Pipeline, FakeInputBackend]:
    pipeline, _, fake = build(build_body_gestures({"jump"}), JUMP_PROFILE)
    return pipeline, fake


def _run(pipeline: Pipeline, poses: list[PoseFrame]) -> None:
    for pose in poses:
        pipeline.step(pose, pose.t)


def test_ghost_pose_after_leaving_frame_does_not_hold_jump() -> None:
    pipeline, fake = _jump_pipeline()
    poses = [_body(HIP_BASE_Y, i * DT) for i in range(20)]  # base R7 (K = 15)
    poses += [_body(120.0, (20 + i) * DT, GHOST_VISIBILITY) for i in range(30)]  # 1 s
    _run(pipeline, poses)
    assert ("press", "space") not in fake.events
    assert "space" not in fake.pressed


def test_returning_at_new_distance_after_ghost_does_not_hold_jump() -> None:
    """Volta ao quadro com os quadris em outra altura (mais perto/longe): sem pulo."""
    pipeline, fake = _jump_pipeline()
    poses = [_body(HIP_BASE_Y, i * DT) for i in range(20)]
    poses += [_body(120.0, (20 + i) * DT, GHOST_VISIBILITY) for i in range(30)]
    poses += [_body(HIP_UP_Y, (50 + i) * DT) for i in range(40)]  # parado, só mais alto
    _run(pipeline, poses)
    assert ("press", "space") not in fake.events
    assert "space" not in fake.pressed


def test_real_jump_taps_space_through_the_mapper() -> None:
    """Controle: pulo de verdade (tudo visível) = um tap de espaço (TRIGGER -> mapper)."""
    pipeline, fake = _jump_pipeline()
    ys = [HIP_BASE_Y] * 20 + [HIP_UP_Y] * 4 + [HIP_BASE_Y] * 6
    _run(pipeline, [_body(y, i * DT) for i, y in enumerate(ys)])
    assert fake.events == [("press", "space"), ("release", "space")]


def test_tap_is_released_by_time_even_without_pose() -> None:
    """O tap vence por tempo mesmo sem pose válida (ainda dentro do timeout de LOST)."""
    pipeline, _, fake = build(build_body_gestures({"jump"}), JUMP_PROFILE)
    ys = [HIP_BASE_Y] * 20 + [HIP_UP_Y] * 3
    _run(pipeline, [_body(y, i * DT) for i, y in enumerate(ys)])
    assert fake.pressed == {"space"}  # tap em andamento
    pipeline.step(None, 23 * DT)  # 0,033 s depois: pose ausente, ainda ACTIVE
    pipeline.step(None, 25 * DT)  # 0,1 s >= tap_hold_s
    assert fake.pressed == set()


def test_pending_tap_is_dropped_on_pause() -> None:
    """Pausa solta tudo (release_all) e o mapper esquece o tap: nada é re-apertado."""
    pipeline, safety, fake = build(build_body_gestures({"jump"}), JUMP_PROFILE)
    ys = [HIP_BASE_Y] * 20 + [HIP_UP_Y] * 3
    _run(pipeline, [_body(y, i * DT) for i, y in enumerate(ys)])
    assert fake.pressed == {"space"}
    safety.on_hotkeys(frozenset({"f8"}))
    pipeline.step(_body(HIP_UP_Y, 23 * DT), 23 * DT)
    assert fake.pressed == set()
    n_events = len(fake.events)
    for i in range(24, 40):  # ainda pausado
        pipeline.step(_body(HIP_UP_Y, i * DT), i * DT)
    assert len(fake.events) == n_events


def test_hips_out_of_frame_counts_as_lost() -> None:
    """Opção A (custo aceito): sem quadris visíveis (muito perto) -> LOST e nada apertado."""
    spy = SpyGesture()
    pipeline, _, fake = make_pipeline(spy)
    res = None
    for i in range(10):
        body = make_body(t=i * DT)
        landmarks = body.landmarks.copy()
        landmarks[[PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP], 3] = 0.1
        res = pipeline.step(replace(body, landmarks=landmarks), i * DT)
    assert res is not None
    assert spy.updates == []  # gestos nunca rodaram com o tronco incompleto
    assert res.safety_state is SafetyState.LOST
    assert res.pose is not None  # o overlay ainda mostra os pontos
    assert fake.pressed == set()


# --- Spec 008 T9: árbitro e combo ---

BOTH_PROFILE = GameProfile(
    "teste",
    {
        "attack": Binding("mouse_left", GestureKind.HOLD),
        "use": Binding("mouse_right", GestureKind.HOLD),
    },
)


def _arms(right: bool, left: bool, i: int) -> PoseFrame:
    """Braço direito e/ou esquerdo esticado para a câmera; o outro pendurado."""
    return make_body(
        t=i * DT,
        right_arm=ARM_STRAIGHT_FORWARD if right else ARM_HANGING,
        left_arm=ARM_STRAIGHT_FORWARD if left else ARM_HANGING,
    )


def test_one_arm_clicks_attack() -> None:
    """Controle: com um braço só (o direito = ataque, destro), o clique acontece."""
    pipeline, _, fake = build(build_body_gestures({"attack", "use"}), BOTH_PROFILE)
    _run(pipeline, [_arms(True, False, i) for i in range(10)])
    assert fake.events == [("press", "mouse_left")]


def test_both_arms_suppress_clicks() -> None:
    """Dois braços esticados juntos: o combo ``both_arms`` só suprime, nenhum clique."""
    pipeline, _, fake = build(build_body_gestures({"attack", "use"}), BOTH_PROFILE)
    _run(pipeline, [_arms(True, True, i) for i in range(10)])
    assert fake.events == []


def test_second_arm_cancels_first_click_and_lowering_it_restores() -> None:
    pipeline, _, fake = build(build_body_gestures({"attack", "use"}), BOTH_PROFILE)
    frames = [_arms(True, False, i) for i in range(6)]
    frames += [_arms(True, True, i) for i in range(6, 12)]
    frames += [_arms(True, False, i) for i in range(12, 18)]
    _run(pipeline, frames)
    assert fake.events == [
        ("press", "mouse_left"),
        ("release", "mouse_left"),  # o segundo braço suprime o ataque
        ("press", "mouse_left"),  # e ao baixá-lo o ataque volta
    ]


def test_min_torso_visibility_validation() -> None:
    assert SafetyConfig().min_torso_visibility == 0.5
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="min_torso_visibility"):
            SafetyConfig(min_torso_visibility=bad)


# --- Persistência (checklist 003 T8): a pose só vale após min_track_s contínuos ---
# Em recordings/fantasma.npz (29/10/2026) o MediaPipe achou uma "pessoa" no travesseiro com
# visibilidade de até 0,99, mas piscando: sequências de 1 a 22 quadros (máx. 0,73 s),
# enquanto a pessoa real ficou 792 quadros seguidos. Sequência real de quadros com e sem
# pose fantasma, a 30 FPS, entre 16,1 s e 32,7 s da gravação:
GHOST_RUNS = (
    (True, 4), (False, 5), (True, 2), (False, 105), (True, 1), (False, 3), (True, 22),
    (False, 24), (True, 11), (False, 39), (True, 2), (False, 6), (True, 8), (False, 60),
    (True, 6), (False, 14), (True, 6), (False, 147), (True, 3), (False, 11), (True, 3),
    (False, 5), (True, 12),
)  # fmt: skip


def _persistent_pipeline(min_track_s: float = 1.0) -> tuple[Pipeline, SpyGesture, FakeInputBackend]:
    spy = SpyGesture()
    config = SafetyConfig(lost_timeout_s=0.2, min_track_s=min_track_s)
    pipeline, _, fake = build([spy], config=config)
    return pipeline, spy, fake


def test_flickering_ghost_from_recording_never_activates() -> None:
    pipeline, spy, fake = _persistent_pipeline()
    i = 0
    for has_pose, n in GHOST_RUNS:
        for _ in range(n):
            t = i * DT
            pose = make_body(t=t) if has_pose else None  # tudo visível
            pipeline.step(pose, t)
            i += 1
    assert spy.updates == []  # os gestos nunca rodaram com o fantasma
    assert fake.pressed == set()


def test_continuous_pose_activates_after_min_track_s() -> None:
    pipeline, spy, _ = _persistent_pipeline(min_track_s=1.0)
    for i in range(45):  # 1,5 s
        pipeline.step(make_body(t=i * DT), i * DT)
    assert spy.updates  # ligou
    assert min(spy.updates) == pytest.approx(1.0, abs=DT)  # e só depois de 1 s


def test_interruption_restarts_min_track_count() -> None:
    pipeline, spy, _ = _persistent_pipeline(min_track_s=1.0)
    for i in range(25):  # 0,83 s com pose
        pipeline.step(make_body(t=i * DT), i * DT)
    pipeline.step(None, 25 * DT)  # um quadro sem pose zera a contagem
    for i in range(26, 50):  # mais 0,8 s: ainda não chegou a 1 s contínuo
        pipeline.step(make_body(t=i * DT), i * DT)
    assert spy.updates == []


def test_min_track_s_default_and_validation() -> None:
    assert SafetyConfig().min_track_s == 1.0
    assert SafetyConfig(min_track_s=0.0).min_track_s == 0.0  # 0 desliga
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_track_s"):
            SafetyConfig(min_track_s=bad)


# --- Spec 008 T11: E3 de ponta a ponta com poses aleatórias ---

ALL_KEYS_PROFILE = GameProfile(
    "teste",
    {
        "attack": Binding("mouse_left", GestureKind.HOLD),
        "use": Binding("mouse_right", GestureKind.HOLD),
        "jump": Binding("space", GestureKind.TRIGGER),
        "crouch": Binding("shift", GestureKind.HOLD),
        "walk": Binding("w", GestureKind.HOLD),
        "sprint": Binding("ctrl", GestureKind.HOLD),
    },
)
PROFILE_KEYS = {b.input for b in ALL_KEYS_PROFILE.bindings.values()}


def _random_pose(seed: int, mode: int, t: float) -> PoseFrame | None:
    """0 = sem pose; 1 = ruído puro (visibilidade variável); 2 = corpo com braços/quadril/tronco
    aleatórios (o que de fato aperta teclas)."""
    rng = np.random.default_rng(seed)
    if mode == 0:
        return None
    if mode == 1:
        landmarks = rng.uniform(0.0, 1.0, (33, 5))
        landmarks[:, 3] = rng.uniform(0.3, 1.0, 33)
        world = rng.normal(0.0, 0.4, (33, 4))
        world[:, 3] = landmarks[:, 3]
        return PoseFrame(t=t, image_size=SIZE, landmarks=landmarks, world=world)

    def arm() -> Arm:
        pick = int(rng.integers(0, 3))
        if pick == 0:
            return ARM_STRAIGHT_FORWARD
        if pick == 1:
            return ARM_HANGING
        up, fore = rng.normal(size=3), rng.normal(size=3)
        return Arm(
            (float(up[0]), float(up[1]), float(up[2])),
            (float(fore[0]), float(fore[1]), float(fore[2])),
        )

    return make_body(
        t=t,
        hip_y_px=float(rng.choice([220.0, 300.0, 380.0])) + float(rng.normal(0.0, 5.0)),
        lean=float(rng.uniform(-0.6, 0.6)),
        knee_lift_left=float(rng.choice([0.0, 0.0, 0.8])),
        knee_lift_right=float(rng.choice([0.0, 0.0, 0.8])),
        left_arm=arm(),
        right_arm=arm(),
    )


@settings(max_examples=60, deadline=None)
@given(
    frames=st.lists(
        st.tuples(
            st.integers(0, 2**31 - 1),  # semente da pose
            st.sampled_from([0, 1, 2, 2, 2, 2, 2, 2]),  # modo
            st.integers(1, 8),  # quantos frames seguidos com a mesma pose
            st.sampled_from([DT, DT, DT, DT, 0.1, 0.3]),  # intervalo até cada frame
            st.sampled_from([False, False, False, False, False, False, False, True]),  # F8
        ),
        min_size=1,
        max_size=30,
    )
)
def test_random_poses_never_leave_keys_pressed(
    frames: list[tuple[int, int, int, float, bool]],
) -> None:
    """E3 de ponta a ponta: o Pipeline com poses aleatórias, perda de tracking e pausas nunca
    deixa tecla presa. Fora de ACTIVE nada fica apertado; só teclas do perfil são apertadas; e
    ao fim (``guard`` -> ``release_all``) o conjunto de pressionadas é vazio."""
    gestures = build_body_gestures(BODY_GESTURE_NAMES)
    pipeline, safety, fake = build(gestures, ALL_KEYS_PROFILE)
    t = 0.0
    with safety.guard():
        for seed, mode, repeat, dt, f8 in frames:
            for _ in range(repeat):
                t += dt
                safety.on_hotkeys(frozenset({"f8"}) if f8 else frozenset())
                result = pipeline.step(_random_pose(seed, mode, t), t)
                if result.safety_state is not SafetyState.ACTIVE:
                    assert fake.pressed == set()
                assert fake.pressed <= PROFILE_KEYS
    assert fake.pressed == set()
    assert safety.input.pressed == frozenset()


def test_random_poses_property_is_not_vacuous() -> None:
    """Controle: com os mesmos geradores, uma sequência de corpo aperta mesmo alguma tecla."""
    gestures = build_body_gestures(BODY_GESTURE_NAMES)
    pipeline, _, fake = build(gestures, ALL_KEYS_PROFILE)
    for i in range(10):
        pipeline.step(
            make_body(t=i * DT, right_arm=ARM_STRAIGHT_FORWARD), i * DT
        )  # soco direito mantido
    assert "mouse_left" in fake.pressed


class _ProbeGesture(FeatureGesture):
    """Gesto de teste: fica ativo a partir de ``t_on`` e guarda as duas bases R7 de cada quadro."""

    def __init__(self, t_on: float) -> None:
        super().__init__("spy", GestureKind.HOLD, SPY_PARAMS, self._probe)
        self._t_on = t_on
        self.bases: list[tuple[float | None, float | None]] = []

    def _probe(self, ctx: FrameContext) -> float:
        self.bases.append((ctx.baseline_px, ctx.shoulder_baseline_px))
        return 1.0 if ctx.t >= self._t_on else 0.0


def test_both_baselines_freeze_together() -> None:
    """Spec 017: a base R7 dos ombros anda em repouso e congela junto com a do quadril."""
    n_rest, n_active = 40, 30
    probe = _ProbeGesture(t_on=n_rest * DT)
    pipeline, _, _ = build([probe])
    for i in range(n_rest + n_active):
        moved = i >= n_rest - 5  # o corpo se mexe um pouco antes e durante o gesto
        pose = make_body(
            t=i * DT,
            hip_y_px=300.0 + (3.0 * (i - n_rest + 5) if moved else 0.0),
            shoulder_dy_px=2.0 * (i - n_rest + 5) if moved else 0.0,
        )
        pipeline.step(pose, i * DT)
    hip_base, shoulder_base = zip(*probe.bases, strict=True)
    assert (hip_base[0], shoulder_base[0]) == (None, None)  # ainda sem K amostras
    assert None not in (hip_base[20], shoulder_base[20])
    frozen = n_rest + 1  # o estado ACTIVE vale a partir do quadro seguinte à ativação
    assert len(set(hip_base[frozen:])) == 1
    assert len(set(shoulder_base[frozen:])) == 1
    assert hip_base[frozen - 1] != hip_base[n_rest - 6]  # e antes disso ainda andava
    assert shoulder_base[frozen - 1] != shoulder_base[n_rest - 6]


# --- Spec 017 RF4 / ADR 0011: com o agachar em curso, a pose vale só com os ombros -----------

CROUCH_PROFILE = GameProfile("teste", {"crouch": Binding("shift", GestureKind.HOLD)})
SQUAT_DY_PX = 120.0  # c = 1,2 larguras de ombro (> theta_on 1,0 do agachar)


def _stand(i: int) -> PoseFrame:
    return make_body(t=i * DT)


def _squat(i: int, *, hips: bool) -> PoseFrame:
    return make_body(t=i * DT, shoulder_dy_px=SQUAT_DY_PX, hips_visible=hips)


def _crouch_pipeline(
    config: SafetyConfig | None = None,
) -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    return build(build_body_gestures({"crouch"}), CROUCH_PROFILE, config)


def _shift_events(fake: FakeInputBackend) -> list[tuple[str, str]]:
    return [e for e in fake.events if e[1] == "shift"]


def test_crouch_keeps_tracking_without_hips() -> None:
    """Agachado (ACTIVE), o quadril sai do quadro por muito mais que ``lost_timeout_s``:
    a pose vale pelos ombros, o ``shift`` segue apertado e nada é solto."""
    pipeline, _, fake = _crouch_pipeline()
    poses = [_stand(i) for i in range(20)]
    poses += [_squat(i, hips=True) for i in range(20, 24)]  # N = 4 frames: START
    poses += [_squat(i, hips=False) for i in range(24, 50)]  # 0,87 s sem quadril
    for pose in poses:
        result = pipeline.step(pose, pose.t)
        assert result.safety_state is SafetyState.ACTIVE
    assert _shift_events(fake) == [("press", "shift")]
    assert fake.pressed == {"shift"}


def test_crouch_keeps_tracking_without_hips_while_arming() -> None:
    """Premissa 1: o quadril some já na descida (ARMING), antes de ``c`` passar de ``θ_on``
    pela quarta vez; o gesto ainda ativa."""
    pipeline, _, fake = _crouch_pipeline()
    poses = [_stand(i) for i in range(20)]
    poses += [_squat(20, hips=True)]  # 1º frame acima de θ_on: IDLE -> ARMING
    poses += [_squat(i, hips=False) for i in range(21, 45)]
    for pose in poses:
        assert pipeline.step(pose, pose.t).safety_state is SafetyState.ACTIVE
    assert fake.pressed == {"shift"}


def test_crouch_released_when_standing_up_without_hips() -> None:
    """Ao levantar com o quadril ainda fora, o END solta o ``shift``; só então a regra
    volta a exigir o quadril e a pose sem ele deixa de ser tracking."""
    pipeline, _, fake = _crouch_pipeline()
    poses = [_stand(i) for i in range(20)]
    poses += [_squat(i, hips=True) for i in range(20, 24)]
    poses += [_squat(i, hips=False) for i in range(24, 30)]
    poses += [make_body(t=i * DT, hips_visible=False) for i in range(30, 36)]  # em pé, sem quadril
    for pose in poses[:-3]:
        pipeline.step(pose, pose.t)
    assert _shift_events(fake) == [("press", "shift"), ("release", "shift")]
    last = poses[-1]
    for k in range(1, 12):  # sem quadril e sem agachar: rejeitada, LOST depois de 0,2 s
        result = pipeline.step(last, last.t + k * DT)
    assert result.safety_state is SafetyState.LOST


def test_no_hips_rejected_when_crouch_idle() -> None:
    """Regressão do fantasma: com o agachar em IDLE a pose sem quadris NÃO é tracking, mesmo
    com os ombros bem visíveis e baixos."""
    pipeline, _, fake = _crouch_pipeline()
    for i in range(20):
        pipeline.step(_stand(i), i * DT)
    states = [pipeline.step(_squat(i, hips=False), i * DT).safety_state for i in range(20, 40)]
    assert states[-1] is SafetyState.LOST  # 20 frames = 0,67 s > lost_timeout_s (0,2 s)
    assert _shift_events(fake) == []


class _CountingCrouch(FeatureGesture):
    """Gesto ``crouch`` de teste: conta os ``update`` (um por pose aceita pelo pipeline)."""

    def __init__(self) -> None:
        crouch = build_body_gestures({"crouch"})[0]
        super().__init__("crouch", GestureKind.HOLD, SPY_PARAMS, crouch._feature)
        self.updates = 0

    def update(self, ctx: FrameContext) -> GestureEvent | None:
        self.updates += 1
        return super().update(ctx)


def test_min_track_s_still_applies_to_shoulder_only_poses() -> None:
    """A persistência continua valendo: depois de um quadro sem pose, a pose só de ombros
    precisa de ``min_track_s`` de detecção contínua antes de chegar ao gesto."""
    config = SafetyConfig(lost_timeout_s=1.0, min_track_s=0.5)
    gesture = _CountingCrouch()
    pipeline, _, _ = build([gesture], CROUCH_PROFILE, config)
    poses = [_stand(i) for i in range(40)]
    poses += [_squat(i, hips=True) for i in range(40, 44)]  # SPY_PARAMS: θ_on 0,5, debounce 1
    for pose in poses:
        pipeline.step(pose, pose.t)
    assert gesture.snapshot().state is not GestureState.IDLE
    before = gesture.updates
    pipeline.step(None, 44 * DT)  # um quadro sem pose zera t₀
    for i in range(45, 45 + 14):  # 14 frames = 0,47 s < min_track_s
        pipeline.step(_squat(i, hips=False), i * DT)
    assert gesture.updates == before
    pipeline.step(_squat(60, hips=False), 60 * DT)  # 0,5 s contínuos: volta a valer
    assert gesture.updates == before + 1


# --- Spec 018: marcha no lugar no pipeline (RF3, RF4, RF6, CA6) ------------------------------

MARCH_PROFILE = GameProfile(
    "teste",
    {"walk": Binding("w", GestureKind.HOLD), "sprint": Binding("ctrl", GestureKind.HOLD)},
)
STEP_FRAMES = 3  # frames com o joelho erguido em cada passo
SLOW = 24  # marcha lenta: 0,8 s por passo (1,25 passos/s, mediana da T0)
FAST = 9  # marcha rápida: 0,3 s por passo (3,3 passos/s)


def _march_poses(start: int, n_steps: int, every: int, first_left: bool = True) -> list[PoseFrame]:
    """Passos alternados a cada ``every`` frames, com o joelho erguido por ``STEP_FRAMES``."""
    poses = []
    for n in range(n_steps):
        left = first_left == (n % 2 == 0)
        for j in range(every):
            lift = 0.8 if j < STEP_FRAMES else 0.0
            i = start + n * every + j
            poses.append(
                make_body(
                    t=i * DT,
                    knee_lift_left=lift if left else 0.0,
                    knee_lift_right=0.0 if left else lift,
                )
            )
    return poses


def _march_pipeline() -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    return build(build_body_gestures({"walk", "sprint"}), MARCH_PROFILE)


def _rest(n: int, start: int = 0) -> list[PoseFrame]:
    return [make_body(t=(start + i) * DT) for i in range(n)]


def test_slow_march_holds_w_without_ctrl_and_releases_on_stop() -> None:
    """RF3 + CA6: marcha lenta (1,25 passos/s) aperta `w` no 2º passo; ao parar de marchar, solta
    em `step_timeout_s` (1,2 s), sem perda de tracking e sem `ctrl`."""
    pipeline, _, fake = _march_pipeline()
    poses = _rest(20) + _march_poses(20, 6, SLOW) + _rest(60, 20 + 6 * SLOW)
    for pose in poses:
        pipeline.step(pose, pose.t)
        assert "ctrl" not in fake.pressed
    assert [e for e in fake.events if e[1] == "w"] == [("press", "w"), ("release", "w")]
    assert fake.pressed == set()


def test_w_released_within_step_timeout_after_last_step() -> None:
    pipeline, _, fake = _march_pipeline()
    last_step_i = 20 + 5 * SLOW  # 6 passos: o último começa neste frame
    release_t = None
    for pose in _rest(20) + _march_poses(20, 6, SLOW) + _rest(60, 20 + 6 * SLOW):
        pipeline.step(pose, pose.t)
        if release_t is None and ("release", "w") in fake.events:
            release_t = pose.t
    assert release_t is not None
    elapsed = release_t - last_step_i * DT
    assert 1.2 <= elapsed <= 1.2 + 2 * DT


def test_fast_march_adds_ctrl_and_both_release_at_the_end() -> None:
    """RF4: marcha rápida soma `ctrl` ao `w`; os dois soltam ao parar."""
    pipeline, _, fake = _march_pipeline()
    both = False
    for pose in _rest(20) + _march_poses(20, 12, FAST) + _rest(60, 20 + 12 * FAST):
        pipeline.step(pose, pose.t)
        both = both or fake.pressed == {"w", "ctrl"}
    assert both
    assert fake.pressed == set()
    assert fake.events.index(("press", "w")) < fake.events.index(("press", "ctrl"))


def test_standing_still_never_presses() -> None:
    pipeline, _, fake = _march_pipeline()
    for pose in _rest(200):
        pipeline.step(pose, pose.t)
    assert fake.events == []


def test_tracking_loss_releases_w_and_march_state_is_cleared() -> None:
    """CA6/RF6: perda de tracking com `walk` ativo solta `w` em até `lost_timeout_s`; ao voltar,
    o estado velho da marcha não reaparece (precisa de novos passos alternados)."""
    pipeline, _, fake = _march_pipeline()
    poses = _rest(20) + _march_poses(20, 4, SLOW)
    for pose in poses:
        pipeline.step(pose, pose.t)
    assert fake.pressed == {"w"}
    t_lost = poses[-1].t
    released_at = None
    for k in range(1, 12):
        t = t_lost + k * DT
        pipeline.step(None, t)
        if released_at is None and fake.pressed == set():
            released_at = t
    assert released_at is not None
    assert released_at - t_lost <= 0.2 + DT
    start = 20 + 4 * SLOW + 12
    for pose in _rest(40, start) + _march_poses(start + 40, 1, SLOW):  # um só passo ao voltar
        pipeline.step(pose, pose.t)
    assert fake.pressed == set()  # um passo isolado não anda
    for pose in _march_poses(start + 40 + SLOW, 1, SLOW, first_left=False):
        pipeline.step(pose, pose.t)
    assert fake.pressed == {"w"}  # o 2º passo alternado anda


def test_pause_releases_w_and_resume_does_not_resume_walking() -> None:
    pipeline, safety, fake = _march_pipeline()
    poses = _rest(20) + _march_poses(20, 4, SLOW)
    for pose in poses:
        pipeline.step(pose, pose.t)
    assert fake.pressed == {"w"}
    t0 = poses[-1].t + DT
    safety.on_hotkeys(frozenset({"f8"}))
    pipeline.step(make_body(t=t0), t0)
    assert fake.pressed == set()
    safety.on_hotkeys(frozenset({"f8"}))  # retoma
    for i in range(1, 11):
        pipeline.step(make_body(t=t0 + i * DT), t0 + i * DT)
    assert fake.pressed == set()


def _bouncing_march(start: int, n_steps: int, every: int) -> list[PoseFrame]:
    """Marcha rápida em que o quadril quica 33 px (j = 0,33 > θ_on do pulo) a cada passo, com o
    joelho erguido (k = 0,8), como na T0."""
    poses = []
    for n in range(n_steps):
        left = n % 2 == 0
        for j in range(every):
            lifted = j < STEP_FRAMES
            lift = 0.8 if lifted else 0.0
            i = start + n * every + j
            poses.append(
                make_body(
                    t=i * DT,
                    hip_y_px=267.0 if lifted else 300.0,
                    knee_lift_left=lift if left else 0.0,
                    knee_lift_right=0.0 if left else lift,
                )
            )
    return poses


def test_fast_march_bounce_does_not_press_space_but_real_jump_does() -> None:
    """RF6b + CA7: o quique da marcha rápida não aperta `space`; um pulo real depois, sim."""
    profile = GameProfile(
        "teste",
        {
            "walk": Binding("w", GestureKind.HOLD),
            "sprint": Binding("ctrl", GestureKind.HOLD),
            "jump": Binding("space", GestureKind.TRIGGER),
        },
    )
    pipeline, _, fake = build(build_body_gestures({"walk", "sprint", "jump"}), profile)
    poses = _rest(20) + _bouncing_march(20, 12, FAST)
    for pose in poses:
        pipeline.step(pose, pose.t)
    assert ("press", "w") in fake.events
    assert [e for e in fake.events if e[1] == "space"] == []
    start = 20 + 12 * FAST + 60  # para de marchar, espera soltar e a base repousar
    for i in range(20 + 12 * FAST, start + 40):
        pipeline.step(make_body(t=i * DT), i * DT)
    for i in range(start + 40, start + 52):  # pulo real: quadril sobe 50 px, joelhos baixos
        pipeline.step(make_body(t=i * DT, hip_y_px=250.0), i * DT)
    assert [e for e in fake.events if e[1] == "space"][:1] == [("press", "space")]
