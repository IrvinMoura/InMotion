"""Testes dos gestos concretos (spec 008): soco, pulo, agachar e inclinar."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from inmotion.gestures.base import (
    Direction,
    EventKind,
    GestureEvent,
    GestureParams,
    GestureState,
)
from inmotion.gestures.body import BodyGesture, all_idle
from inmotion.gestures.features import BaselineParams, BodyBaseline, build_context
from inmotion.gestures.jump import jump_gesture
from inmotion.gestures.punch import punch_gesture, punch_pair
from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side
from tests.helpers import ARM_BENT_DOWN, ARM_HANGING, ARM_STRAIGHT_FORWARD, make_body

GEOMETRY = GeometryConfig()
DT = 1.0 / 30.0

PUNCH = GestureParams(0.70, 0.55, Direction.ABOVE, debounce_frames=3)
JUMP = GestureParams(0.25, 0.15, Direction.ABOVE, debounce_frames=2)


class Rig:
    """Roda frames por contexto -> gestos, como o pipeline fará (repouso = todos IDLE)."""

    def __init__(self, gestures: Sequence[BodyGesture]) -> None:
        self.gestures = list(gestures)
        self.baseline = BodyBaseline(BaselineParams())
        self.events: list[GestureEvent] = []

    def step(self, pose: PoseFrame) -> None:
        ctx = build_context(pose, pose.t, GEOMETRY, self.baseline, all_idle(self.gestures))
        for g in self.gestures:
            ev = g.update(ctx)
            if ev is not None:
                self.events.append(ev)

    def kinds(self, name: str) -> list[EventKind]:
        return [e.kind for e in self.events if e.gesture == name]


def test_straight_punch_starts_bent_down_does_not() -> None:
    """CA1: braço reto à frente -> START; dobrado para baixo -> nada."""
    straight = Rig([punch_gesture("attack", Side.RIGHT, PUNCH, GEOMETRY)])
    for i in range(10):
        straight.step(make_body(t=i * DT, right_arm=ARM_STRAIGHT_FORWARD))
    assert straight.kinds("attack") == [EventKind.START]

    bent = Rig([punch_gesture("attack", Side.RIGHT, PUNCH, GEOMETRY)])
    for i in range(30):
        bent.step(make_body(t=i * DT, right_arm=ARM_BENT_DOWN))
    assert bent.events == []


def test_punch_start_after_n_frames_and_end_on_release() -> None:
    """Debounce R5 (N = 3) e saída por histerese R6: START no 3º frame, END ao baixar o braço."""
    rig = Rig([punch_gesture("attack", Side.RIGHT, PUNCH, GEOMETRY)])
    for i in range(2):
        rig.step(make_body(t=i * DT, right_arm=ARM_STRAIGHT_FORWARD))
    assert rig.events == []
    rig.step(make_body(t=2 * DT, right_arm=ARM_STRAIGHT_FORWARD))
    assert rig.kinds("attack") == [EventKind.START]
    rig.step(make_body(t=3 * DT, right_arm=ARM_HANGING))
    assert rig.kinds("attack") == [EventKind.START, EventKind.END]


def test_handedness_maps_arms() -> None:
    """CA2: destro -> o braço direito ataca e o esquerdo usa item; canhoto inverte."""
    for handedness, attack_arm, use_arm in (
        (Side.RIGHT, "right", "left"),
        (Side.LEFT, "left", "right"),
    ):
        for extended, expected in ((attack_arm, "attack"), (use_arm, "use")):
            rig = Rig(list(punch_pair(handedness, PUNCH, PUNCH, GEOMETRY)))
            for i in range(5):
                arms = {f"{extended}_arm": ARM_STRAIGHT_FORWARD}
                rig.step(make_body(t=i * DT, **arms))
            assert [(e.gesture, e.kind) for e in rig.events] == [(expected, EventKind.START)]


def test_feature_gesture_close_reset_and_value() -> None:
    gesture = punch_gesture("attack", Side.RIGHT, PUNCH, GEOMETRY)
    rig = Rig([gesture])
    for i in range(5):
        rig.step(make_body(t=i * DT, right_arm=ARM_STRAIGHT_FORWARD))
    assert gesture.feature_value == pytest.approx(1.0)
    assert gesture.snapshot().state is GestureState.ACTIVE
    assert not all_idle([gesture])
    end = gesture.close(5 * DT)
    assert end is not None
    assert end.kind is EventKind.END
    assert gesture.close(6 * DT) is None
    gesture.reset()
    assert gesture.feature_value is None
    assert gesture.snapshot().state is GestureState.IDLE
    assert all_idle([gesture])


def _run_hip(rig: Rig, hip_y: NDArray[np.float64]) -> None:
    for i, y in enumerate(hip_y):
        rig.step(make_body(t=i * DT, hip_y_px=float(y), right_arm=ARM_HANGING))


def test_jump_single_pulse_and_still_none() -> None:
    """CA3: trajetória de pulo -> 1 PULSE; em pé com ruído de 1% de s por 60 s -> 0."""
    rng = np.random.default_rng(0)
    still = Rig([jump_gesture(JUMP)])
    _run_hip(still, 300.0 + rng.normal(0.0, 1.0, 60 * 30))  # s = 100 px: ruído de 1% de s
    assert still.events == []

    jumping = Rig([jump_gesture(JUMP)])
    hip = np.full(120, 300.0)
    hip[60:72] = 250.0  # o quadril sobe meia largura de ombros por 0,4 s
    _run_hip(jumping, hip)
    assert jumping.kinds("jump") == [EventKind.PULSE]


def test_slow_drift_no_jump() -> None:
    """CA4: deriva lenta (aproximação da câmera) -> nenhum pulo; a base acompanha."""
    rig = Rig([jump_gesture(JUMP)])
    n = 20 * 30
    for i in range(n):
        frac = i / (n - 1)
        rig.step(make_body(t=i * DT, scale=1.0 + 0.3 * frac, hip_y_px=300.0 - 30.0 * frac))
    assert rig.events == []


def test_slow_drift_baseline_follows() -> None:
    """CA4/J3: a base R7 acompanha a deriva do quadril com defasagem pequena (~v/λ)."""
    rig = Rig([jump_gesture(JUMP)])
    n = 20 * 30
    hip = 300.0
    for i in range(n):
        hip = 300.0 - 30.0 * i / (n - 1)
        rig.step(make_body(t=i * DT, scale=1.0 + 0.3 * i / (n - 1), hip_y_px=hip))
    assert rig.baseline.value is not None
    assert abs(rig.baseline.value - hip) < 5.0  # defasagem estacionária v/λ = 2,5 px
    assert rig.events == []


def test_two_jumps_two_pulses() -> None:
    rig = Rig([jump_gesture(JUMP)])
    hip = np.full(200, 300.0)
    hip[50:62] = 250.0
    hip[120:132] = 250.0
    _run_hip(rig, hip)
    assert rig.kinds("jump") == [EventKind.PULSE, EventKind.PULSE]


def test_baseline_frozen_during_jump_j2() -> None:
    """J2: a base fica congelada enquanto o pulo não volta a IDLE."""
    rig = Rig([jump_gesture(JUMP)])
    hip = np.full(120, 300.0)
    hip[50:90] = 250.0  # meio segundo a mais no ar: 40 frames
    _run_hip(rig, hip)
    assert rig.kinds("jump") == [EventKind.PULSE]
    assert rig.baseline.value is not None
    assert abs(rig.baseline.value - 300.0) < 2.0  # só ~2 frames de arming contaminam


def test_no_jump_before_baseline_is_ready() -> None:
    """RF3: a base é média de K frames; antes de K amostras não há pulo."""
    rig = Rig([jump_gesture(JUMP)])
    hip = np.full(40, 300.0)
    hip[5:15] = 200.0  # quadril "alto" cedo demais: faz parte da média, não é pulo
    _run_hip(rig, hip)
    assert rig.events == []


def test_jump_threshold_works_at_two_distances() -> None:
    """J1: o mesmo limiar vale a 1,5 m e a 3 m (escala 2x nos pixels)."""
    for scale in (1.0, 2.0):
        rig = Rig([jump_gesture(JUMP)])
        for i in range(120):
            lift = 0.5 * 100.0 * scale if 60 <= i < 72 else 0.0
            rig.step(
                make_body(
                    t=i * DT, scale=scale, hip_y_px=600.0 - lift, size=(1280, 960), hip_x_px=640.0
                )
            )
        assert rig.kinds("jump") == [EventKind.PULSE]
