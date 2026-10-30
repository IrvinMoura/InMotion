"""Câmera em thread: guarda só o frame mais recente (adaptador).

Uma thread lê a captura o mais rápido que o driver entregar e sobrescreve o último
frame; o loop principal pega sempre o mais novo, sem fila crescendo (latência mínima).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import cv2

from inmotion.capture import DEFAULT_CAMERA_INDEX
from inmotion.ports import Clock
from inmotion.types import FramePacket

_log = logging.getLogger(__name__)

_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
_JOIN_TIMEOUT_S = 1.0
_READ_FAIL_WAIT_S = 0.01
_THREAD_NAME = "inmotion-camera"


@dataclass(frozen=True)
class CameraConfig:
    """Parâmetros da câmera (os padrões migram para ``config/default.yaml`` na spec 004)."""

    index: int = DEFAULT_CAMERA_INDEX
    backend: Literal["dshow", "msmf", "any"] = "dshow"
    width: int = 640
    height: int = 480
    fps: int = 60
    fourcc: str = "MJPG"
    exposure: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index deve ser >= 0: {self.index}")
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend inválido: {self.backend!r} (use {sorted(_BACKENDS)})")
        for name in ("width", "height", "fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser > 0: {getattr(self, name)}")
        if len(self.fourcc) != 4:
            raise ValueError(f"fourcc deve ter 4 caracteres: {self.fourcc!r}")
        if self.exposure is not None and not math.isfinite(self.exposure):
            raise ValueError(f"exposure deve ser finito ou None: {self.exposure}")


class Camera:
    """Satisfaz o port ``FrameSource``; captura em thread própria."""

    def __init__(
        self,
        config: CameraConfig,
        clock: Clock,
        capture_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._factory: Callable[[int, int], Any] = capture_factory or cv2.VideoCapture
        self._cap: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._latest: FramePacket | None = None
        self._last_returned = -1
        self._next_index = 0
        self._error: BaseException | None = None
        self._actual: tuple[int, int, float] = (0, 0, 0.0)

    @property
    def actual(self) -> tuple[int, int, float]:
        """(W, H, FPS) que o driver realmente aceitou (lidos via ``get``)."""
        return self._actual

    def start(self) -> None:
        """Abre a captura, aplica as configurações e inicia a thread.

        Raises:
            OSError: se a câmera não abrir.
        """
        if self._thread is not None:
            raise RuntimeError("Camera já iniciada")
        cfg = self._config
        cap = self._factory(cfg.index, _BACKENDS[cfg.backend])
        self._cap = cap
        if not cap.isOpened():
            cap.release()
            raise OSError(f"não foi possível abrir a câmera {cfg.index} ({cfg.backend})")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*cfg.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, cfg.exposure)
        self._actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS)),
        )
        _log.info(
            "câmera %d (%s): pedido %dx%d@%d %s, obtido %dx%d@%.1f",
            cfg.index,
            cfg.backend,
            cfg.width,
            cfg.height,
            cfg.fps,
            cfg.fourcc,
            *self._actual,
        )
        self._thread = threading.Thread(target=self._run, name=_THREAD_NAME, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ok, image = self._cap.read()
                if not ok or image is None:
                    self._stop.wait(_READ_FAIL_WAIT_S)
                    continue
                packet = FramePacket(t=self._clock.now(), index=self._next_index, image=image)
                self._next_index += 1
                with self._cond:
                    self._latest = packet
                    self._cond.notify_all()
        except BaseException as exc:  # relançada no read()
            _log.exception("thread da câmera falhou")
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def read(self, timeout_s: float) -> FramePacket | None:
        """Devolve um frame NOVO (índice maior que o último entregue) ou None no timeout.

        Raises:
            Exception: a que encerrou a thread de captura, se houver.
        """
        with self._cond:
            self._cond.wait_for(
                lambda: (
                    self._error is not None
                    or (self._latest is not None and self._latest.index > self._last_returned)
                ),
                timeout=max(timeout_s, 0.0),
            )
            if self._error is not None:
                raise self._error
            packet = self._latest
            if packet is None or packet.index <= self._last_returned:
                return None
            self._last_returned = packet.index
            return packet

    def close(self) -> None:
        """Para a thread (join em < 1 s) e solta a captura. Idempotente."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                _log.warning("thread da câmera não encerrou em %.1f s", _JOIN_TIMEOUT_S)
            self._thread = None
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
