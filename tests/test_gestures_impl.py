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
)
from inmotion.gestures.body import BodyGesture, all_idle
from inmotion.gestures.features import BaselineParams, BodyBaseline, build_context
from inmotion.gestures.jump import jump_gesture
from inmotion.gestures.punch import punch_gesture
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


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
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


def _run_hip(rig: Rig, hip_y: NDArray[np.float64]) -> None:
    for i, y in enumerate(hip_y):
        rig.step(make_body(t=i * DT, hip_y_px=float(y), right_arm=ARM_HANGING))


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
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


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_slow_drift_no_jump() -> None:
    """CA4: deriva lenta (aproximação da câmera) -> nenhum pulo; a base acompanha."""
    rig = Rig([jump_gesture(JUMP)])
    n = 20 * 30
    for i in range(n):
        frac = i / (n - 1)
        rig.step(make_body(t=i * DT, scale=1.0 + 0.3 * frac, hip_y_px=300.0 - 30.0 * frac))
    assert rig.events == []
