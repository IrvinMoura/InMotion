"""Spec 002 T9 — app.run com fakes (sem webcam, sem Windows)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from inmotion.app import AppDeps, run
from inmotion.filters.pose_filter import PoseFilter
from inmotion.safety import SafetyConfig
from inmotion.types import FramePacket, PoseFrame
from inmotion.vision.landmarks import PoseLandmark
from tests.fakes import (
    FakeClock,
    FakeFrameSource,
    FakeHotkeys,
    FakeInputBackend,
    FakePoseEstimator,
    make_pose,
)

DT = 1 / 30
HIP_BASE_Y = 300.0
HIP_UP_Y = 250.0  # sobe 50 px em 480 p: acima de rise·H = 33,3 px


class SimulatedEstimatorError(Exception):
    """Falha simulada do estimador (não herda de RuntimeError de propósito)."""


class ClockedSource(FakeFrameSource):
    """FakeFrameSource que avança o ``FakeClock`` a cada ``read`` (um frame = 1/30 s)."""

    def __init__(self, clock: FakeClock, n_frames: int) -> None:
        packets = [
            FramePacket(t=i * DT, index=i, image=np.zeros((480, 640, 3), np.uint8))
            for i in range(n_frames)
        ]
        super().__init__(packets)
        self._clock = clock

    def read(self, timeout_s: float) -> FramePacket | None:
        self._clock.advance(DT)
        return super().read(timeout_s)


def hip_pose(y: float, t: float) -> PoseFrame:
    return make_pose({PoseLandmark.LEFT_HIP: (300.0, y), PoseLandmark.RIGHT_HIP: (340.0, y)}, t=t)


JUMP = [hip_pose(y, i * DT) for i, y in enumerate([HIP_BASE_Y, HIP_BASE_Y, HIP_UP_Y, HIP_BASE_Y])]


def make_deps(
    clock: FakeClock,
    source: FakeFrameSource,
    estimator: FakePoseEstimator,
    backend: FakeInputBackend,
    hotkeys: FakeHotkeys | None = None,
) -> AppDeps:
    return AppDeps(
        source=source,
        estimator=estimator,
        backend=backend,
        hotkeys=hotkeys or FakeHotkeys([]),
        clock=clock,
        display=None,
        pose_filter=None,  # o One Euro atrasaria o pulo sintético; ele é testado à parte
    )


def test_jump_script_produces_press_release() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 4)
    estimator = FakePoseEstimator(JUMP)

    code = run(make_deps(clock, source, estimator, backend), SafetyConfig(), max_frames=4)

    assert code == 0
    jump_events = [e for e in backend.events if e[1] == "space"]
    assert jump_events == [("press", "space"), ("release", "space")]
    assert backend.pressed == set()
    assert source.closed
    assert estimator.closed


def test_f9_exits_and_releases_all() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 50)
    estimator = FakePoseEstimator(JUMP[:3] + [JUMP[3]] * 47)
    # F9 só no 4.º poll: a tecla do pulo está apertada quando o app sai.
    hotkeys = FakeHotkeys([frozenset(), frozenset(), frozenset(), frozenset({"f9"})])
    deps = make_deps(clock, source, estimator, backend, hotkeys)

    code = run(deps, SafetyConfig(), max_frames=None)

    assert code == 0
    assert ("press", "space") in backend.events
    assert backend.pressed == set()
    assert backend.events[-1][0] == "release"
    assert len(estimator.calls) < 50  # saiu antes de consumir tudo
    assert source.closed
    assert estimator.closed


def test_estimator_exception_releases_all() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 10)
    estimator = FakePoseEstimator([JUMP[0], JUMP[1], JUMP[2], SimulatedEstimatorError("boom")])

    with pytest.raises(SimulatedEstimatorError):
        run(make_deps(clock, source, estimator, backend), SafetyConfig(), max_frames=10)

    assert ("press", "space") in backend.events
    assert backend.pressed == set()
    assert source.closed
    assert estimator.closed


class StallingSource(ClockedSource):
    """No 4.º read o relógio salta 1 s (> timeout do watchdog) e o read "trava" até a thread
    do watchdog soltar as teclas (com limite de 5 s, para o teste não pendurar)."""

    def __init__(self, clock: FakeClock, backend: FakeInputBackend) -> None:
        super().__init__(clock, 4)
        self._backend = backend
        self.reads = 0
        self.released_during_stall: bool | None = None

    def read(self, timeout_s: float) -> FramePacket | None:
        self.reads += 1
        if self.reads < 4:
            return super().read(timeout_s)
        self._clock.advance(1.0)
        deadline = time.monotonic() + 5.0
        while self._backend.pressed and time.monotonic() < deadline:
            time.sleep(0.01)
        self.released_during_stall = not self._backend.pressed
        return None


def test_watchdog_releases_keys_while_loop_is_stalled() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = StallingSource(clock, backend)
    estimator = FakePoseEstimator([JUMP[0], JUMP[1], JUMP[2]])

    run(make_deps(clock, source, estimator, backend), SafetyConfig(), max_frames=4)

    assert ("press", "space") in backend.events
    assert source.released_during_stall is True  # soltou enquanto o read ainda estava preso
    assert backend.pressed == set()


def test_default_deps_use_one_euro_filter() -> None:
    clock = FakeClock()
    deps = AppDeps(
        source=FakeFrameSource([]),
        estimator=FakePoseEstimator([]),
        backend=FakeInputBackend(),
        hotkeys=FakeHotkeys([]),
        clock=clock,
        display=None,
    )
    assert isinstance(deps.pose_filter, PoseFilter)
