"""Spec 002 T5 — Pipeline.step: segurança -> gestos (só em ACTIVE) -> FPS."""

import numpy as np
import pytest

from inmotion.filters import FloatArray
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.input.backend import MOUSE_BUTTONS
from inmotion.pipeline import Pipeline
from inmotion.safety import SafetyConfig, SafetyManager, SafetyState, TrackedInput
from inmotion.types import PoseFrame
from tests.fakes import FakeInputBackend, make_pose


class SpyGesture:
    """Gesto de teste: guarda as chamadas e aperta uma tecla a cada ``update``."""

    def __init__(self) -> None:
        self.updates: list[float] = []
        self.resets = 0

    def update(self, pose: PoseFrame, out: TrackedInput) -> None:
        self.updates.append(pose.t)
        out.press("space")

    def reset(self) -> None:
        self.resets += 1


def make_pipeline(spy: SpyGesture) -> tuple[Pipeline, SafetyManager, FakeInputBackend]:
    fake = FakeInputBackend()
    safety = SafetyManager(
        TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(lost_timeout_s=0.2, min_track_s=0.0)
    )
    return Pipeline(safety, [spy], FpsMeter(window=30)), safety, fake


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


class PoseRecordingGesture:
    """Gesto de teste: guarda o x do landmark 0 que recebeu."""

    def __init__(self) -> None:
        self.seen_x: list[float] = []

    def update(self, pose: PoseFrame, out: TrackedInput) -> None:
        self.seen_x.append(float(pose.landmarks[0, 0]))

    def reset(self) -> None:
        pass


def make_filtered_pipeline() -> tuple[Pipeline, PoseRecordingGesture, list[ConstantTemporalFilter]]:
    inner: list[ConstantTemporalFilter] = []

    def factory() -> ConstantTemporalFilter:
        inner.append(ConstantTemporalFilter())
        return inner[-1]

    safety = SafetyManager(
        TrackedInput(FakeInputBackend(), MOUSE_BUTTONS),
        SafetyConfig(lost_timeout_s=0.2, min_track_s=0.0),
    )
    gesture = PoseRecordingGesture()
    pipeline = Pipeline(safety, [gesture], FpsMeter(window=30), PoseFilter(factory))
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
# baixa). Como a pose não era None, o app não entrou em LOST: o pulo legado viu os quadris
# "subirem", segurou o espaço e, sem LOST, nada soltou a tecla nem recalibrou a base.

DT = 1 / 30
SIZE = (640, 480)
HIP_BASE_Y = 300.0
HIP_UP_Y = 250.0  # 50 px acima da base: passa do rise·H = 33,3 px
GHOST_VISIBILITY = 0.05


def _hips(y: float, t: float, visibility: float = 1.0) -> PoseFrame:
    from inmotion.vision.landmarks import PoseLandmark

    points = {PoseLandmark.LEFT_HIP: (300.0, y), PoseLandmark.RIGHT_HIP: (340.0, y)}
    vis = dict.fromkeys(range(33), visibility)
    return make_pose(points, size=SIZE, t=t, visibility=vis)


def _jump_pipeline() -> tuple[Pipeline, FakeInputBackend]:
    from inmotion.gestures.legacy import LegacyJump

    fake = FakeInputBackend()
    safety = SafetyManager(
        TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(lost_timeout_s=0.2, min_track_s=0.0)
    )
    return Pipeline(safety, [LegacyJump()], FpsMeter(window=30)), fake


def _run(pipeline: Pipeline, poses: list[PoseFrame]) -> None:
    for pose in poses:
        pipeline.step(pose, pose.t)


def test_ghost_pose_after_leaving_frame_does_not_hold_jump() -> None:
    pipeline, fake = _jump_pipeline()
    poses = [_hips(HIP_BASE_Y, i * DT) for i in range(10)]
    poses += [_hips(120.0, (10 + i) * DT, GHOST_VISIBILITY) for i in range(30)]  # 1 s
    _run(pipeline, poses)
    assert "space" not in fake.pressed


def test_returning_at_new_distance_after_ghost_does_not_hold_jump() -> None:
    """Volta ao quadro com os quadris em outra altura (mais perto/longe): sem pulo preso."""
    pipeline, fake = _jump_pipeline()
    poses = [_hips(HIP_BASE_Y, i * DT) for i in range(10)]
    poses += [_hips(120.0, (10 + i) * DT, GHOST_VISIBILITY) for i in range(30)]
    poses += [_hips(HIP_UP_Y, (40 + i) * DT) for i in range(30)]  # parado, só mais alto
    _run(pipeline, poses)
    assert "space" not in fake.pressed


def test_real_jump_still_presses_and_releases() -> None:
    """Controle: pulo de verdade (tudo visível) aperta e solta o espaço."""
    pipeline, fake = _jump_pipeline()
    ys = [HIP_BASE_Y] * 3 + [HIP_UP_Y] * 3 + [HIP_BASE_Y] * 3
    _run(pipeline, [_hips(y, i * DT) for i, y in enumerate(ys)])
    assert fake.events == [("press", "space"), ("release", "space")]


def test_hips_out_of_frame_counts_as_lost() -> None:
    """Opção A (custo aceito): sem quadris visíveis (muito perto) -> LOST e nada apertado."""
    from inmotion.vision.landmarks import PoseLandmark

    spy = SpyGesture()
    pipeline, _, fake = make_pipeline(spy)
    hidden = {PoseLandmark.LEFT_HIP: 0.1, PoseLandmark.RIGHT_HIP: 0.1}
    res = None
    for i in range(10):
        res = pipeline.step(make_pose({}, t=i * DT, visibility=hidden), i * DT)
    assert res is not None
    assert spy.updates == []  # gestos nunca rodaram com o tronco incompleto
    assert res.safety_state is SafetyState.LOST
    assert res.pose is not None  # o overlay ainda mostra os pontos
    assert fake.pressed == set()


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
    fake = FakeInputBackend()
    config = SafetyConfig(lost_timeout_s=0.2, min_track_s=min_track_s)
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), config)
    return Pipeline(safety, [spy], FpsMeter(window=30)), spy, fake


def test_flickering_ghost_from_recording_never_activates() -> None:
    pipeline, spy, fake = _persistent_pipeline()
    i = 0
    for has_pose, n in GHOST_RUNS:
        for _ in range(n):
            t = i * DT
            pose = make_pose({}, size=SIZE, t=t) if has_pose else None  # tudo visível
            pipeline.step(pose, t)
            i += 1
    assert spy.updates == []  # os gestos nunca rodaram com o fantasma
    assert fake.pressed == set()


def test_continuous_pose_activates_after_min_track_s() -> None:
    pipeline, spy, _ = _persistent_pipeline(min_track_s=1.0)
    for i in range(45):  # 1,5 s
        pipeline.step(make_pose({}, size=SIZE, t=i * DT), i * DT)
    assert spy.updates  # ligou
    assert min(spy.updates) == pytest.approx(1.0, abs=DT)  # e só depois de 1 s


def test_interruption_restarts_min_track_count() -> None:
    pipeline, spy, _ = _persistent_pipeline(min_track_s=1.0)
    for i in range(25):  # 0,83 s com pose
        pipeline.step(make_pose({}, size=SIZE, t=i * DT), i * DT)
    pipeline.step(None, 25 * DT)  # um quadro sem pose zera a contagem
    for i in range(26, 50):  # mais 0,8 s: ainda não chegou a 1 s contínuo
        pipeline.step(make_pose({}, size=SIZE, t=i * DT), i * DT)
    assert spy.updates == []


def test_min_track_s_default_and_validation() -> None:
    assert SafetyConfig().min_track_s == 1.0
    assert SafetyConfig(min_track_s=0.0).min_track_s == 0.0  # 0 desliga
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="min_track_s"):
            SafetyConfig(min_track_s=bad)
