"""Ponto de entrada: ``python -m inmotion``. Composição: lê a CLI e configura o logging."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inmotion.capture import DEFAULT_CAMERA_INDEX

_log = logging.getLogger("inmotion")

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EXIT_USAGE = 2  # configuração ou ambiente inválido (modelo ausente, câmera indisponível)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inmotion",
        description="Controla jogos por movimento corporal usando uma webcam.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help=f"índice da câmera (padrão: {DEFAULT_CAMERA_INDEX})",
    )
    parser.add_argument(
        "--camera-backend",
        choices=("dshow", "msmf", "any"),
        default="dshow",
        help="backend do OpenCV para a câmera (padrão: dshow)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="pasta com o modelo pose_landmarker_<variante>.task (padrão: models)",
    )
    parser.add_argument(
        "--model-variant",
        choices=("lite", "full", "heavy"),
        default="full",
        help="variante do modelo de pose (padrão: full)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="nível de log (padrão: INFO)",
    )
    parser.add_argument(
        "--log-file",
        metavar="NOME",
        help="grava o log também em logs/NOME (a pasta logs/ é ignorada pelo git)",
    )
    parser.add_argument(
        "--no-overlay", action="store_true", help="não abre a janela de debug (mais FPS)"
    )
    return parser


def configure_logging(level: str, log_file: str | None) -> None:
    """Console (stderr) sempre; arquivo em ``logs/`` quando ``log_file`` é dado."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        if len(path.parts) == 1:
            path = LOG_DIR / path
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=handlers, force=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.log_file)
    # Imports tardios: `--help` não carrega OpenCV nem MediaPipe.
    from inmotion.app import AppDeps, OpenCvDisplay, run
    from inmotion.capture.camera import Camera, CameraConfig
    from inmotion.clock import MonotonicClock
    from inmotion.input.hotkeys import WindowsHotkeys
    from inmotion.input.keyboard_mouse import KeyboardMouseBackend
    from inmotion.safety import SafetyConfig
    from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError, PoseConfig

    safety_config = SafetyConfig()
    # O modelo é checado ANTES de abrir a câmera: sem ele, nada de hardware é tocado.
    try:
        estimator = MediaPipePoseEstimator(
            PoseConfig(model_dir=args.model_dir, variant=args.model_variant)
        )
    except ModelNotFoundError as exc:
        _log.error("%s", exc)
        return EXIT_USAGE

    clock = MonotonicClock()
    camera = Camera(CameraConfig(index=args.camera, backend=args.camera_backend), clock)
    try:
        camera.start()
    except OSError as exc:
        estimator.close()
        _log.error("%s", exc)
        return EXIT_USAGE
    _log.info("Câmera aberta: %dx%d a %.0f FPS", *camera.actual)

    try:
        deps = AppDeps(
            source=camera,
            estimator=estimator,
            backend=KeyboardMouseBackend(),
            hotkeys=WindowsHotkeys(frozenset({safety_config.pause_key, safety_config.exit_key})),
            clock=clock,
            display=None if args.no_overlay else OpenCvDisplay(),
        )
    except BaseException:
        # Falhou antes do run(): quem fecha a câmera e o estimador somos nós.
        camera.close()
        estimator.close()
        raise
    _log.info("Rodando: %s pausa/retoma, %s sai.", safety_config.pause_key, safety_config.exit_key)
    return run(deps, safety_config)


if __name__ == "__main__":
    sys.exit(main())
