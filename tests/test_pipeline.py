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
from inmotion.gestures.base import Direction, GestureKind, GestureParams
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
