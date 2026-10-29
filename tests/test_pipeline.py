"""Spec 002 T5 — Pipeline.step: segurança -> gestos (só em ACTIVE) -> FPS."""

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
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(lost_timeout_s=0.2))
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
