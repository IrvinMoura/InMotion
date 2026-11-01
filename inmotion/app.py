"""Composição do aplicativo: liga câmera, pose, pipeline, segurança e exibição.

Camada de composição: pode importar tudo. A lógica por frame mora no ``Pipeline`` (núcleo);
aqui ficam o loop, a thread do watchdog e o ``OpenCvDisplay``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.config.schema import AppConfig, Binding, FilterConfig, GameProfile
from inmotion.exit_zone import ExitZone, ExitZoneConfig
from inmotion.filters.ema import Ema
from inmotion.filters.one_euro import OneEuro
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.arbiter import ArbiterConfig
from inmotion.gestures.base import GestureKind
from inmotion.gestures.factory import build_body_gestures
from inmotion.input.backend import HotkeySource, InputBackend
from inmotion.overlay import render
from inmotion.pipeline import Pipeline, StepResult
from inmotion.ports import Clock, Display, FrameSource, PoseEstimator
from inmotion.safety import (
    SafetyAction,
    SafetyConfig,
    SafetyManager,
    TrackedInput,
    register_atexit,
)
from inmotion.vision.landmarks import Side
from inmotion.watchdog import Watchdog

_log = logging.getLogger(__name__)

WATCHDOG_POLL_S = 0.05  # período da thread que consulta o watchdog
READ_TIMEOUT_S = 0.1  # espera máxima por frame novo
FPS_WINDOW = 30
WINDOW_NAME = "InMotion"
_THREAD_JOIN_S = 1.0

# Perfil usado quando não há ``--game``: o do Minecraft (usuário destro: ataque = braço
# direito, botão esquerdo; usar item = braço esquerdo, botão direito; spec 017: sem ``walk``
# até a marcha da spec 018, e agachar vence o pulo). Um teste confere que
# é igual a ``config/games/minecraft.yaml``.
DEFAULT_PROFILE = GameProfile(
    "minecraft",
    {
        "attack": Binding("mouse_left", GestureKind.HOLD),
        "use": Binding("mouse_right", GestureKind.HOLD),
        "jump": Binding("space", GestureKind.TRIGGER),
        "crouch": Binding("shift", GestureKind.HOLD),
        "walk": Binding("w", GestureKind.HOLD),
        "sprint": Binding("ctrl", GestureKind.HOLD),
    },
    ArbiterConfig(
        priority=("both_arms", "crouch", "jump", "attack", "use", "walk", "sprint"),
        exclusive=(
            frozenset({"crouch", "jump"}),
            frozenset({"crouch", "walk"}),
            frozenset({"crouch", "sprint"}),
        ),
        combos={"both_arms": frozenset({"attack", "use"})},
    ),
)


def pose_filter_from_config(config: FilterConfig) -> PoseFilter | None:
    """Um filtro para image e outro para world, do tipo e parâmetros da config (R1-R4)."""
    if config.kind == "none":
        return None
    if config.kind == "ema":
        ema_params = config.ema
        return PoseFilter(lambda: Ema(ema_params))
    one_euro_params = config.one_euro
    return PoseFilter(lambda: OneEuro(one_euro_params))


def default_pose_filter() -> PoseFilter | None:
    """Filtro com os parâmetros padrão do ``FilterConfig`` (One Euro)."""
    return pose_filter_from_config(FilterConfig())


@dataclass
class AppDeps:
    """Dependências injetadas em ``run`` (adaptadores reais ou fakes)."""

    source: FrameSource
    estimator: PoseEstimator
    backend: InputBackend
    hotkeys: HotkeySource
    clock: Clock
    display: Display | None
    pose_filter: PoseFilter | None = field(default_factory=default_pose_filter)
    profile: GameProfile = DEFAULT_PROFILE
    config: AppConfig = field(default_factory=AppConfig)  # gestos, base R7, mapper, geometria
    exit_zone: ExitZoneConfig | None = field(default_factory=ExitZoneConfig)
    # Pré-processamento antes da pose (spec 005, A2: blur gaussiano); None = imagem crua.
    preprocess: Callable[[NDArray[np.uint8]], NDArray[np.uint8]] | None = None


class OpenCvDisplay:
    """``Display`` sobre ``cv2.imshow``: desenha o overlay e detecta a janela fechada."""

    def __init__(self, window_name: str = WINDOW_NAME, handedness: str | None = None) -> None:
        self._name = window_name
        self._handedness = handedness  # "right" | "left": o overlay mostra o modo (spec 017)

    def show(self, image: NDArray[np.uint8], result: StepResult) -> bool:
        """Mostra o frame com overlay; False se a janela foi fechada."""
        cv2.imshow(self._name, render(image, result, self._handedness))
        cv2.waitKey(1)
        return bool(cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) >= 1)

    def close(self) -> None:
        cv2.destroyAllWindows()


def _watch(
    watchdog: Watchdog, safety: SafetyManager, clock: Clock, stop: threading.Event, timeout_s: float
) -> None:
    """Thread do watchdog: se o loop parar de bater, solta todo o input."""
    while not stop.wait(WATCHDOG_POLL_S):
        if watchdog.expired(clock.now()):
            _log.error("Loop travado por mais de %.2f s: soltando todo o input", timeout_s)
            try:
                safety.on_stall()
            except Exception:
                _log.exception("on_stall() falhou")


def _close_quietly(name: str, obj: FrameSource | PoseEstimator | Display | None) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        _log.exception("Falha ao fechar %s", name)


def run(deps: AppDeps, safety_config: SafetyConfig, max_frames: int | None = None) -> int:
    """Loop principal. Devolve 0 na saída normal (F9, janela fechada, zona de saída ou
    ``max_frames``).

    Zona de saída (temporária): com a mão na zona, todo o input é solto e os gestos
    ficam suspensos (o braço esticado no canto não vira clique); segurar por
    ``hold_s`` encerra o loop.

    Todo o loop roda dentro de ``safety.guard()``: qualquer saída (inclusive exceção do
    estimador ou do display) solta o input. Exceções são propagadas ao chamador.
    """
    tracked = TrackedInput(deps.backend, safety_config.always_release)
    safety = SafetyManager(tracked, safety_config)
    config = deps.config
    gestures = build_body_gestures(
        deps.profile.bindings,
        Side.RIGHT if config.handedness == "right" else Side.LEFT,
        config.gestures,
        config.geometry,
        config.crouch.feature,
        config.march,
    )
    pipeline = Pipeline(
        safety,
        gestures,
        FpsMeter(window=FPS_WINDOW),
        deps.pose_filter,
        profile=deps.profile,
        mapper_params=config.mapper,
        baseline_params=config.baseline,
        march_params=config.march,
        geometry=config.geometry,
    )
    watchdog = Watchdog(safety_config.watchdog_timeout_s)
    exit_zone = ExitZone(deps.exit_zone) if deps.exit_zone is not None else None
    register_atexit(safety)

    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch,
        args=(watchdog, safety, deps.clock, stop, safety_config.watchdog_timeout_s),
        name="watchdog",
        daemon=True,
    )
    frames = 0
    try:
        watcher.start()
        with safety.guard():
            while max_frames is None or frames < max_frames:
                frames += 1
                t = deps.clock.now()
                watchdog.beat(t)
                if safety.on_hotkeys(deps.hotkeys.poll()) is SafetyAction.EXIT:
                    _log.info("Saída solicitada (%s)", safety_config.exit_key)
                    break
                packet = deps.source.read(READ_TIMEOUT_S)
                pose = None
                if packet is not None:
                    image = packet.image
                    if deps.preprocess is not None:
                        image = deps.preprocess(image)  # só a pose vê; a janela mostra o cru
                    pose = deps.estimator.estimate(image, packet.t)
                zone = exit_zone.update(pose, t) if exit_zone is not None else None
                if zone is not None and zone.inside:
                    tracked.release_all()
                    result = replace(pipeline.step(None, t), pose=pose, exit_zone=zone)
                else:
                    result = replace(pipeline.step(pose, t), exit_zone=zone)
                if (
                    deps.display is not None
                    and packet is not None
                    and not deps.display.show(packet.image, result)
                ):
                    _log.info("Janela fechada")
                    break
                if zone is not None and zone.triggered:
                    _log.info("Saída pela zona de saída (mão segurada %.0f s)", zone.config.hold_s)
                    break
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(_THREAD_JOIN_S)
        _close_quietly("source", deps.source)
        _close_quietly("estimator", deps.estimator)
        _close_quietly("display", deps.display)
    return 0
