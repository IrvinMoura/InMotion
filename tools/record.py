"""Grava sessões de landmarks rotuladas por teclas globais (spec 011, T4; RF1 e RF2).

Uso::

    python -m tools.record --person p1 --lighting clara --distance 2.0 --out data/p1.npz

Durante a gravação, segure uma tecla de rótulo enquanto executa o gesto (teclas globais,
funcionam com qualquer janela em foco). F9 salva o arquivo e encerra. Qualquer saída,
inclusive por exceção, salva o que já foi gravado e fecha câmera e estimador.

Ferramenta de composição: monta os adaptadores reais; a lógica testável fica em
``record_session``, que recebe os ports por injeção.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.capture.camera import Camera
from inmotion.clock import MonotonicClock
from inmotion.config.loader import load_camera_config
from inmotion.config.schema import CameraConfig, PoseConfig
from inmotion.input.backend import HotkeySource, KeyName
from inmotion.input.hotkeys import WindowsHotkeys
from inmotion.ports import Clock, FrameSource, PoseEstimator
from inmotion.recorder import Recorder
from inmotion.recording import SCHEMA_VERSION, LabelTracker, RecordingMeta
from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError

logger = logging.getLogger("record")

EXIT_KEY: KeyName = "f9"
DEFAULT_LABEL_KEYS: dict[KeyName, str] = {
    "1": "punch_right",
    "2": "punch_left",
    "3": "jump",
    "4": "crouch",
    "5": "lean",
}
LIGHTING_CHOICES: tuple[str, ...] = ("clara", "escura", "outra")
VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
READ_TIMEOUT_S = 0.5
_VIDEO_FOURCC = "mp4v"
_VIDEO_FALLBACK_FPS = 30.0


class NoFramesError(RuntimeError):
    """Sessão encerrada sem nenhum frame gravado: não há o que salvar."""


class VideoSink(Protocol):
    """Destino opcional do vídeo bruto (``.mp4`` separado do ``.npz``)."""

    def write(self, image: NDArray[np.uint8]) -> None: ...

    def close(self) -> None: ...


class Mp4VideoSink:
    """``VideoSink`` sobre ``cv2.VideoWriter``; abre o arquivo no primeiro frame."""

    def __init__(self, path: Path, fps: float) -> None:
        self._path = path
        self._fps = fps if fps > 0 else _VIDEO_FALLBACK_FPS
        self._writer: Any = None

    def write(self, image: NDArray[np.uint8]) -> None:
        if self._writer is None:
            height, width = image.shape[:2]
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = cv2.VideoWriter(
                str(self._path), cv2.VideoWriter.fourcc(*_VIDEO_FOURCC), self._fps, (width, height)
            )
        self._writer.write(image)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def compute_config_hash(
    camera: CameraConfig, variant: str, key_to_gesture: Mapping[str, str]
) -> str:
    """Hash estável (SHA-256, 12 hex) dos argumentos que afetam os dados gravados.

    Entram: configuração da câmera (índice, backend, resolução, FPS, fourcc, exposição),
    variante do modelo de pose e o mapa tecla -> gesto. Ficam de fora pessoa, distância,
    iluminação e caminhos (já estão nos metadados ou não alteram o conteúdo).
    """
    payload = {
        "camera": {
            "index": camera.index,
            "backend": camera.backend,
            "width": camera.width,
            "height": camera.height,
            "fps": camera.fps,
            "fourcc": camera.fourcc,
            "exposure": camera.exposure,
        },
        "model_variant": variant,
        "labels": dict(sorted(key_to_gesture.items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def record_session(
    source: FrameSource,
    estimator: PoseEstimator,
    hotkeys: HotkeySource,
    clock: Clock,
    meta: RecordingMeta,
    out: Path,
    max_frames: int | None = None,
    key_to_gesture: Mapping[KeyName, str] | None = None,
    video: VideoSink | None = None,
) -> Path:
    """Grava frames e rótulos até F9 (ou ``max_frames``) e salva o ``.npz``.

    Por frame: lê a câmera, sonda as teclas globais, estima a pose e acumula ``(t, pose)``
    com ``t`` do próprio frame; as teclas de rótulo seguradas viram intervalos no mesmo
    eixo de tempo. ``meta.created`` é preenchido aqui (UTC, ISO 8601). Em qualquer saída
    (F9, limite, exceção) salva o já gravado e fecha fonte, estimador e vídeo.

    Raises:
        NoFramesError: se a sessão terminou normalmente sem nenhum frame.
    """
    labels_map = dict(DEFAULT_LABEL_KEYS if key_to_gesture is None else key_to_gesture)
    meta = replace(meta, created=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    tracker = LabelTracker(labels_map)
    label_keys = frozenset(labels_map)
    recorder: Recorder | None = None
    last_t = 0.0
    n_frames = 0
    started = clock.now()
    saved: Path | None = None
    try:
        while max_frames is None or n_frames < max_frames:
            packet = source.read(READ_TIMEOUT_S)
            keys = hotkeys.poll()
            if EXIT_KEY in keys:
                break
            if packet is None:
                continue
            if recorder is None:
                height, width = packet.image.shape[:2]
                recorder = Recorder(meta, (width, height))
            pose = estimator.estimate(packet.image, packet.t)
            recorder.add(packet.t, pose)
            if video is not None:
                video.write(packet.image)
            tracker.update(keys & label_keys, packet.t)
            last_t = packet.t
            n_frames += 1
    finally:
        try:
            source.close()
            estimator.close()
            if video is not None:
                video.close()
        finally:
            if recorder is not None:
                saved = recorder.save(out, tracker.close(last_t))
                logger.info(
                    "%d frames em %.1f s salvos em %s", n_frames, clock.now() - started, saved
                )
    if saved is None:
        raise NoFramesError("nenhum frame gravado; nada foi salvo")
    return saved


def _label_help() -> str:
    return ", ".join(f"{key}={gesture}" for key, gesture in DEFAULT_LABEL_KEYS.items())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grava landmarks de uma sessão com rótulos por teclas globais.",
        epilog=(
            f"Teclas de rótulo (segure durante o gesto): {_label_help()}. "
            f"{EXIT_KEY.upper()} salva e sai."
        ),
    )
    parser.add_argument("--person", required=True, help="id anônimo da pessoa (ex.: p1)")
    parser.add_argument("--distance", type=float, default=None, help="distância da câmera (m)")
    parser.add_argument("--lighting", choices=LIGHTING_CHOICES, default="outra")
    parser.add_argument("--out", type=Path, required=True, help="caminho do .npz de saída")
    parser.add_argument("--video", type=Path, default=None, help="grava também um .mp4 bruto")
    parser.add_argument("--camera", type=int, default=None, help="índice da câmera")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--model-variant", choices=VARIANTS, default="full")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Monta os adaptadores reais e grava. 0 = ok, 1 = falha, 2 = modelo ausente."""
    args = _build_parser().parse_args(argv)
    pose_config = PoseConfig(model_dir=args.model_dir, variant=args.model_variant)
    if not pose_config.model_path.is_file():
        logger.error(
            "Modelo não encontrado: %s. Baixe com: python tools/download_model.py --variant %s",
            pose_config.model_path,
            args.model_variant,
        )
        return 2

    camera_config = load_camera_config(index=args.camera)
    clock = MonotonicClock()
    meta = RecordingMeta(
        schema_version=SCHEMA_VERSION,
        created="",
        person=args.person,
        distance_m=args.distance,
        lighting=args.lighting,
        camera=f"{camera_config.backend}:{camera_config.index}",
        config_hash=compute_config_hash(camera_config, args.model_variant, DEFAULT_LABEL_KEYS),
    )
    camera = Camera(camera_config, clock)
    estimator: PoseEstimator | None = None
    try:
        estimator = MediaPipePoseEstimator(pose_config)
        camera.start()
    except (ModelNotFoundError, OSError) as exc:
        logger.error("Falha ao iniciar: %s", exc)
        camera.close()
        if estimator is not None:
            estimator.close()
        return 2 if isinstance(exc, ModelNotFoundError) else 1
    video = Mp4VideoSink(args.video, camera.actual[2]) if args.video else None
    hotkeys = WindowsHotkeys(frozenset(DEFAULT_LABEL_KEYS) | {EXIT_KEY})
    logger.info("Gravando. Rótulos: %s. %s salva e sai.", _label_help(), EXIT_KEY.upper())
    try:
        record_session(camera, estimator, hotkeys, clock, meta, args.out, video=video)
    except NoFramesError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
