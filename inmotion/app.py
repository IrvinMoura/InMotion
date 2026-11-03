"""Composição do aplicativo: liga câmera, pose, pipeline, segurança e exibição.

Camada de composição: pode importar tudo. A lógica por frame mora no ``Pipeline`` (núcleo);
aqui ficam o loop, a thread do watchdog e o ``OpenCvDisplay``.
"""

from __future__ import annotations

import atexit
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.config.schema import AppConfig, FilterConfig, GameProfile
from inmotion.exit_zone import ExitZone, ExitZoneConfig
from inmotion.filters.pose_filter import PoseFilter
from inmotion.input.backend import HotkeySource, InputBackend
from inmotion.overlay import render
from inmotion.pipeline import StepResult
from inmotion.ports import Clock, Display, FrameSource, PoseEstimator
from inmotion.safety import (
    SafetyAction,
    SafetyConfig,
    SafetyManager,
    TrackedInput,
    register_atexit,
)
from inmotion.session import DEFAULT_PROFILE, create_pipeline, pose_filter_from_config
from inmotion.watchdog import Watchdog

_log = logging.getLogger(__name__)

WATCHDOG_POLL_S = 0.05  # período da thread que consulta o watchdog
READ_TIMEOUT_S = 0.1  # espera máxima por frame novo
WINDOW_NAME = "InMotion"
_THREAD_JOIN_S = 1.0


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
    retry_release = False
    while not stop.wait(WATCHDOG_POLL_S):
        expired = watchdog.expired(clock.now())
        if expired:
            _log.error("Loop travado por mais de %.2f s: soltando todo o input", timeout_s)
        if expired or retry_release:
            try:
                safety.on_stall()
                retry_release = False
            except Exception:
                retry_release = True
                _log.exception("on_stall() falhou")


def _close_quietly(name: str, obj: FrameSource | PoseEstimator | Display | None) -> None:
    if obj is None:
        return
    try:
        obj.close()
    except Exception:
        _log.exception("Falha ao fechar %s", name)


def _run(deps: AppDeps, safety_config: SafetyConfig, max_frames: int | None = None) -> int:
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
    pipeline = create_pipeline(safety, config, deps.profile, deps.pose_filter)
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
    zone_inside = False
    clean_exit = False
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
                # A segurança usa o horário atual, incluindo espera e inferência.
                t = deps.clock.now()
                if packet is not None and t - packet.t > safety_config.lost_timeout_s:
                    pose = None
                zone = exit_zone.update(pose, t) if exit_zone is not None else None
                if zone is not None and zone.inside:
                    if not zone_inside:
                        safety.on_stall()
                    result = replace(pipeline.step(None, t), pose=pose, exit_zone=zone)
                else:
                    result = replace(pipeline.step(pose, t), exit_zone=zone)
                zone_inside = zone is not None and zone.inside
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
        clean_exit = True
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(_THREAD_JOIN_S)
        if clean_exit:
            atexit.unregister(safety.input.release_all)
    return 0


def run(deps: AppDeps, safety_config: SafetyConfig, max_frames: int | None = None) -> int:
    """Executa a sessão e fecha recursos inclusive se a montagem do pipeline falhar."""
    try:
        return _run(deps, safety_config, max_frames)
    finally:
        _close_quietly("source", deps.source)
        _close_quietly("estimator", deps.estimator)
        _close_quietly("display", deps.display)
