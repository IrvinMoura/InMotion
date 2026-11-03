"""Spec 002 T9 — app.run com fakes (sem webcam, sem Windows)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from inmotion.app import AppDeps, run
from inmotion.config.schema import AppConfig, Binding, CrouchConfig, GameProfile
from inmotion.filters.pose_filter import PoseFilter
from inmotion.gestures.base import Direction, GestureKind, GestureParams
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


def test_handedness_left_from_config_swaps_the_arms() -> None:
    """``handedness: left``: o braço esquerdo ataca (``mouse_left``); CA2 de ponta a ponta."""
    from inmotion.config.schema import AppConfig

    clock = FakeClock()
    backend = FakeInputBackend()
    poses = [make_body(t=i * DT, left_arm=ARM_STRAIGHT_FORWARD) for i in range(10)]
    deps = make_deps(clock, ClockedSource(clock, 10), FakePoseEstimator(poses), backend)
    deps.config = AppConfig(handedness="left")

    run(deps, NO_PERSISTENCE, max_frames=10)

    assert ("press", "mouse_left") in backend.events
    assert ("press", "mouse_right") not in backend.events


def test_gesture_params_come_from_config() -> None:
    """``config.gestures`` vale: com debounce 8 o soco de 6 frames não chega a clicar."""
    from dataclasses import replace

    from inmotion.config.schema import AppConfig

    clock = FakeClock()
    backend = FakeInputBackend()
    slow = replace(AppConfig().gestures["attack"], debounce_frames=8)
    deps = make_deps(clock, ClockedSource(clock, 6), FakePoseEstimator(PUNCH[:6]), backend)
    deps.config = AppConfig(gestures={**AppConfig().gestures, "attack": slow})

    run(deps, NO_PERSISTENCE, max_frames=6)

    assert ("press", "mouse_left") not in backend.events


# --- Spec 005 T7b: o app aplica o pré-processamento (blur A2) antes da pose ---


class ImageSpyEstimator(FakePoseEstimator):
    """Guarda a imagem que chegou ao estimador."""

    def __init__(self) -> None:
        super().__init__([])
        self.images: list[np.ndarray] = []

    def estimate(self, image_bgr: np.ndarray, t: float) -> PoseFrame | None:
        self.images.append(image_bgr)
        return super().estimate(image_bgr, t)


class SpyDisplay:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []

    def show(self, image: np.ndarray, result: object) -> bool:
        self.images.append(image)
        return True

    def close(self) -> None:
        pass


def test_preprocess_runs_before_pose_and_display_shows_raw_frame() -> None:
    clock = FakeClock()
    estimator = ImageSpyEstimator()
    display = SpyDisplay()
    deps = make_deps(clock, ClockedSource(clock, 3), estimator, FakeInputBackend())
    deps.display = display
    deps.preprocess = lambda image: np.full_like(image, 7)

    run(deps, NO_PERSISTENCE, max_frames=3)

    assert len(estimator.images) == 3
    assert all((img == 7).all() for img in estimator.images)  # pose vê a imagem tratada
    assert all(not img.any() for img in display.images)  # janela mostra o frame cru


def test_without_preprocess_estimator_gets_raw_frame() -> None:
    clock = FakeClock()
    estimator = ImageSpyEstimator()
    deps = make_deps(clock, ClockedSource(clock, 2), estimator, FakeInputBackend())
    assert deps.preprocess is None

    run(deps, NO_PERSISTENCE, max_frames=2)

    assert all(not img.any() for img in estimator.images)


def _crouch_poses(*, shoulders_drop: bool) -> list[PoseFrame]:
    """20 quadros em pé e 15 agachados: descem só os ombros ou só o quadril."""
    standing = [make_body(t=i * DT) for i in range(BASE_FRAMES)]
    drop = {"shoulder_dy_px": 120.0} if shoulders_drop else {"hip_y_px": 340.0}
    crouched = [make_body(t=(BASE_FRAMES + i) * DT, **drop) for i in range(15)]  # type: ignore[arg-type]
    return standing + crouched


def _run_crouch(poses: list[PoseFrame], config: AppConfig) -> FakeInputBackend:
    clock = FakeClock()
    backend = FakeInputBackend()
    deps = make_deps(clock, ClockedSource(clock, len(poses)), FakePoseEstimator(poses), backend)
    deps.config = config
    run(deps, NO_PERSISTENCE, max_frames=len(poses))
    return backend


def test_crouch_feature_from_config_selects_the_signal() -> None:
    """Spec 017 RF3: ``crouch.feature`` chega à fábrica (ombros por padrão; ``hip`` opcional)."""
    shoulders = _run_crouch(_crouch_poses(shoulders_drop=True), AppConfig())
    assert ("press", "shift") in shoulders.events
    assert shoulders.pressed == set()

    hip_gestures = {**AppConfig().gestures, "crouch": GestureParams(0.35, 0.25, Direction.ABOVE)}
    hip_config = AppConfig(crouch=CrouchConfig("hip"), gestures=hip_gestures)
    by_hip = _run_crouch(_crouch_poses(shoulders_drop=False), hip_config)
    assert ("press", "shift") in by_hip.events
    # Com o padrão (ombros), descer só o quadril não agacha.
    assert ("press", "shift") not in _run_crouch(
        _crouch_poses(shoulders_drop=False), AppConfig()
    ).events


def test_stale_frames_do_not_generate_input() -> None:
    clock = FakeClock(t0=10.0)
    backend = FakeInputBackend()
    estimator = FakePoseEstimator(PUNCH[:10])
    deps = make_deps(clock, ClockedSource(clock, 10), estimator, backend)
    run(deps, NO_PERSISTENCE, max_frames=10)
    assert not any(kind == "press" for kind, _ in backend.events)


def test_inference_delay_is_included_in_frame_age() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()

    class SlowEstimator(FakePoseEstimator):
        def estimate(self, image_bgr: np.ndarray, t: float) -> PoseFrame | None:
            clock.advance(0.25)
            return super().estimate(image_bgr, t)

    deps = make_deps(clock, ClockedSource(clock, 6), SlowEstimator(PUNCH[:6]), backend)
    run(deps, NO_PERSISTENCE, max_frames=6)
    assert not any(kind == "press" for kind, _ in backend.events)


def test_short_exit_zone_visit_allows_held_gesture_to_rearm() -> None:
    clock = FakeClock()
    backend = FakeInputBackend()
    poses = [*PUNCH[:6], punch_in_exit_zone(6 * DT), *PUNCH[7:14]]
    deps = make_deps(clock, ClockedSource(clock, len(poses)), FakePoseEstimator(poses), backend)
    run(deps, NO_PERSISTENCE, max_frames=len(poses))
    assert backend.events.count(("press", "mouse_left")) == 2


def test_setup_failure_closes_source_and_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    import inmotion.app as app_module

    def fail(*args: object) -> None:
        raise ValueError("montagem inválida")

    clock = FakeClock()
    source = ClockedSource(clock, 1)
    estimator = FakePoseEstimator([])
    deps = make_deps(clock, source, estimator, FakeInputBackend())
    monkeypatch.setattr(app_module, "create_pipeline", fail)
    with pytest.raises(ValueError, match="montagem"):
        run(deps, NO_PERSISTENCE)
    assert source.closed
    assert estimator.closed


def test_watchdog_retries_transient_release_failure() -> None:
    import threading

    from inmotion.app import _watch
    from inmotion.safety import SafetyManager, TrackedInput
    from inmotion.watchdog import Watchdog

    stop = threading.Event()

    class FailsOnce(FakeInputBackend):
        attempts = 0

        def release(self, key: str) -> None:
            if key == "w":
                self.attempts += 1
                if self.attempts == 1:
                    raise OSError("falha transitória")
                stop.set()
            super().release(key)

    backend = FailsOnce()
    safety = SafetyManager(TrackedInput(backend), NO_PERSISTENCE)
    safety.input.press("w")
    watchdog = Watchdog(0.5)
    watchdog.beat(0)
    timeout = threading.Timer(2.0, stop.set)
    timeout.start()
    try:
        _watch(watchdog, safety, FakeClock(1.0), stop, 0.5)
    finally:
        timeout.cancel()
    assert backend.attempts == 2
    assert backend.pressed == set()
