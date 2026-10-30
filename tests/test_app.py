"""Spec 002 T9 — app.run com fakes (sem webcam, sem Windows)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from inmotion.app import AppDeps, run
from inmotion.config.schema import Binding, GameProfile
from inmotion.filters.pose_filter import PoseFilter
from inmotion.gestures.base import GestureKind
from inmotion.safety import SafetyConfig
from inmotion.types import FramePacket, PoseFrame
from tests.fakes import (
    FakeClock,
    FakeFrameSource,
    FakeHotkeys,
    FakeInputBackend,
    FakePoseEstimator,
)
from tests.helpers import ARM_STRAIGHT_FORWARD, make_body

DT = 1 / 30
# Persistência (min_track_s) desligada: estes testes cobrem outras coisas; ela tem os seus.
NO_PERSISTENCE = SafetyConfig(min_track_s=0.0)
HIP_BASE_Y = 300.0
HIP_UP_Y = 250.0  # sobe 50 px = 0,5 largura de ombro (> theta_on do pulo, 0,25)
BASE_FRAMES = 20  # a base R7 precisa de K = 15 frames de repouso


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
    return make_body(t=t, hip_y_px=y)


def punch_pose(t: float) -> PoseFrame:
    """Braço direito (ataque, destro) esticado para a câmera: segura ``mouse_left``."""
    return make_body(t=t, right_arm=ARM_STRAIGHT_FORWARD)


JUMP = [
    hip_pose(y, i * DT)
    for i, y in enumerate([HIP_BASE_Y] * BASE_FRAMES + [HIP_UP_Y] * 3 + [HIP_BASE_Y] * 8)
]
PUNCH = [punch_pose(i * DT) for i in range(60)]


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
    source = ClockedSource(clock, len(JUMP))
    estimator = FakePoseEstimator(JUMP)

    code = run(make_deps(clock, source, estimator, backend), NO_PERSISTENCE, max_frames=len(JUMP))

    assert code == 0
    jump_events = [e for e in backend.events if e[1] == "space"]
    assert jump_events == [("press", "space"), ("release", "space")]
    assert backend.pressed == set()
    assert source.closed
    assert estimator.closed


def test_f9_exits_and_releases_all() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 60)
    estimator = FakePoseEstimator(PUNCH)
    # F9 só no 11.º poll: o clique do soco está apertado quando o app sai.
    hotkeys = FakeHotkeys([frozenset()] * 10 + [frozenset({"f9"})])
    deps = make_deps(clock, source, estimator, backend, hotkeys)

    code = run(deps, NO_PERSISTENCE, max_frames=None)

    assert code == 0
    assert ("press", "mouse_left") in backend.events
    assert backend.pressed == set()
    assert backend.events[-1][0] == "release"
    assert len(estimator.calls) < 60  # saiu antes de consumir tudo
    assert source.closed
    assert estimator.closed


def test_estimator_exception_releases_all() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 10)
    estimator = FakePoseEstimator([*PUNCH[:6], SimulatedEstimatorError("boom")])

    with pytest.raises(SimulatedEstimatorError):
        run(make_deps(clock, source, estimator, backend), NO_PERSISTENCE, max_frames=10)

    assert ("press", "mouse_left") in backend.events
    assert backend.pressed == set()
    assert source.closed
    assert estimator.closed


class StallingSource(ClockedSource):
    """No 6.º read o relógio salta 1 s (> timeout do watchdog) e o read "trava" até a thread
    do watchdog soltar as teclas (com limite de 5 s, para o teste não pendurar)."""

    def __init__(self, clock: FakeClock, backend: FakeInputBackend) -> None:
        super().__init__(clock, 6)
        self._backend = backend
        self.reads = 0
        self.released_during_stall: bool | None = None

    def read(self, timeout_s: float) -> FramePacket | None:
        self.reads += 1
        if self.reads < 6:
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
    estimator = FakePoseEstimator(PUNCH[:5])

    run(make_deps(clock, source, estimator, backend), NO_PERSISTENCE, max_frames=6)

    assert ("press", "mouse_left") in backend.events
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


# --- Zona de saída temporária: mão parada no canto superior esquerdo por 4 s ---


def punch_in_exit_zone(t: float) -> PoseFrame:
    """Braço direito esticado para a câmera (soco) com o pulso na zona de saída.

    O pulso fica em (64, 120) px = (0,10; 0,25) normalizado: o ombro direito está 50 px à
    esquerda do quadril e 150 px acima dele (``k = 250 px/m``).
    """
    return make_body(t=t, hip_x_px=114.0, hip_y_px=270.0, right_arm=ARM_STRAIGHT_FORWARD)


def test_hand_held_in_exit_zone_exits_without_clicking() -> None:
    from inmotion.exit_zone import ExitZoneConfig

    clock = FakeClock()
    backend = FakeInputBackend()
    n = 200
    source = ClockedSource(clock, n)
    estimator = FakePoseEstimator([punch_in_exit_zone(i * DT) for i in range(n)])
    deps = make_deps(clock, source, estimator, backend)
    deps.exit_zone = ExitZoneConfig(hold_s=4.0)

    code = run(deps, NO_PERSISTENCE, max_frames=n)

    assert code == 0
    assert 4.0 / DT <= len(estimator.calls) <= 4.0 / DT + 3  # saiu logo após 4 s
    assert ("press", "mouse_left") not in backend.events  # gestos suspensos na zona
    assert backend.pressed == set()


def test_same_pose_clicks_when_exit_zone_disabled() -> None:
    """Controle: sem a zona, a mesma pose é um soco (o teste acima não é vazio)."""
    clock = FakeClock()
    backend = FakeInputBackend()
    source = ClockedSource(clock, 6)
    estimator = FakePoseEstimator([punch_in_exit_zone(i * DT) for i in range(6)])
    deps = make_deps(clock, source, estimator, backend)
    deps.exit_zone = None

    run(deps, NO_PERSISTENCE, max_frames=6)

    assert ("press", "mouse_left") in backend.events
    assert backend.pressed == set()


def test_watchdog_timeout_comes_from_safety_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """CA9: o limiar do watchdog vem do ``SafetyConfig`` (config), não de uma constante."""
    import inmotion.app as app_module

    timeouts: list[float] = []

    class SpyWatchdog(app_module.Watchdog):
        def __init__(self, timeout_s: float = 0.5) -> None:
            timeouts.append(timeout_s)
            super().__init__(timeout_s)

    monkeypatch.setattr(app_module, "Watchdog", SpyWatchdog)
    clock = FakeClock()
    deps = make_deps(
        clock, ClockedSource(clock, 1), FakePoseEstimator(JUMP[:1]), FakeInputBackend()
    )
    run(deps, SafetyConfig(min_track_s=0.0, watchdog_timeout_s=1.25), max_frames=1)
    assert timeouts == [1.25]


def test_profile_bindings_choose_the_keys() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    deps = make_deps(clock, ClockedSource(clock, 10), FakePoseEstimator(PUNCH[:10]), backend)
    deps.profile = GameProfile("teste", {"attack": Binding("w", GestureKind.HOLD)})

    run(deps, NO_PERSISTENCE, max_frames=10)

    assert ("press", "w") in backend.events  # o gesto attack aperta a tecla do perfil
    assert ("press", "mouse_left") not in backend.events
    assert backend.pressed == set()


def test_pose_filter_from_config_selects_kind() -> None:
    from inmotion.app import pose_filter_from_config
    from inmotion.config.schema import FilterConfig

    assert pose_filter_from_config(FilterConfig(kind="none")) is None
    assert pose_filter_from_config(FilterConfig(kind="ema")) is not None
    assert pose_filter_from_config(FilterConfig(kind="one_euro")) is not None
