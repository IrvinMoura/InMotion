"""Composição do aplicativo: liga câmera, pose, pipeline, segurança e exibição.

Camada de composição: pode importar tudo. A lógica por frame mora no ``Pipeline`` (núcleo);
aqui ficam o loop, a thread do watchdog e o ``OpenCvDisplay``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.fps import FpsMeter
from inmotion.gestures.legacy import LegacyGesture, LegacyJump, LegacyPunch
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

WATCHDOG_TIMEOUT_S = 0.5  # sem beat por mais que isso, o loop é dado como travado
WATCHDOG_POLL_S = 0.05  # período da thread que consulta o watchdog
READ_TIMEOUT_S = 0.1  # espera máxima por frame novo
FPS_WINDOW = 30
WINDOW_NAME = "InMotion"
_THREAD_JOIN_S = 1.0

# Mesmas teclas e lados do protótipo (usuário destro): ataque = braço direito, botão
# esquerdo; usar item = braço esquerdo, botão direito.
_ATTACK_KEY = "mouse_left"
_USE_ITEM_KEY = "mouse_right"


@dataclass
class AppDeps:
    """Dependências injetadas em ``run`` (adaptadores reais ou fakes)."""

    source: FrameSource
    estimator: PoseEstimator
    backend: InputBackend
    hotkeys: HotkeySource
    clock: Clock
    display: Display | None


class OpenCvDisplay:
    """``Display`` sobre ``cv2.imshow``: desenha o overlay e detecta a janela fechada."""

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self._name = window_name

    def show(self, image: NDArray[np.uint8], result: StepResult) -> bool:
        """Mostra o frame com overlay; False se a janela foi fechada."""
        cv2.imshow(self._name, render(image, result))
        cv2.waitKey(1)
        return bool(cv2.getWindowProperty(self._name, cv2.WND_PROP_VISIBLE) >= 1)

    def close(self) -> None:
        cv2.destroyAllWindows()


def _watch(watchdog: Watchdog, safety: SafetyManager, clock: Clock, stop: threading.Event) -> None:
    """Thread do watchdog: se o loop parar de bater, solta todo o input."""
    while not stop.wait(WATCHDOG_POLL_S):
        if watchdog.expired(clock.now()):
            _log.error("Loop travado por mais de %.2f s: soltando todo o input", WATCHDOG_TIMEOUT_S)
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
    """Loop principal. Devolve 0 na saída normal (F9, janela fechada ou ``max_frames``).

    Todo o loop roda dentro de ``safety.guard()``: qualquer saída (inclusive exceção do
    estimador ou do display) solta o input. Exceções são propagadas ao chamador.
    """
    tracked = TrackedInput(deps.backend, safety_config.always_release)
    safety = SafetyManager(tracked, safety_config)
    gestures: list[LegacyGesture] = [
        LegacyJump(),
        LegacyPunch(Side.RIGHT, _ATTACK_KEY),
        LegacyPunch(Side.LEFT, _USE_ITEM_KEY),
    ]
    pipeline = Pipeline(safety, gestures, FpsMeter(window=FPS_WINDOW))
    watchdog = Watchdog(WATCHDOG_TIMEOUT_S)
    register_atexit(safety)

    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch, args=(watchdog, safety, deps.clock, stop), name="watchdog", daemon=True
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
                pose = deps.estimator.estimate(packet.image, packet.t) if packet else None
                result = pipeline.step(pose, t)
                if (
                    deps.display is not None
                    and packet is not None
                    and not deps.display.show(packet.image, result)
                ):
                    _log.info("Janela fechada")
                    break
    finally:
        stop.set()
        if watcher.is_alive():
            watcher.join(_THREAD_JOIN_S)
        _close_quietly("source", deps.source)
        _close_quietly("estimator", deps.estimator)
        _close_quietly("display", deps.display)
    return 0
