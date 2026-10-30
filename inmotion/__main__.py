"""Ponto de entrada: ``python -m inmotion``. Composição: lê a CLI e configura o logging."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from inmotion.config.loader import DEFAULT_CONFIG_PATH, load_config
from inmotion.config.schema import AppConfig, GameProfile
from inmotion.config.validate import ConfigError

_log = logging.getLogger("inmotion")

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EXIT_USAGE = 2  # configuração ou ambiente inválido (modelo ausente, câmera indisponível)


def build_parser() -> argparse.ArgumentParser:
    """CLI. Os atalhos (``--camera``...) não têm padrão: o padrão vem do YAML (RF5/RF6)."""
    parser = argparse.ArgumentParser(
        prog="python -m inmotion",
        description="Controla jogos por movimento corporal usando uma webcam.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        metavar="ARQUIVO",
        help="arquivo YAML base, no lugar de config/default.yaml",
    )
    parser.add_argument(
        "--game",
        metavar="NOME",
        help="perfil de jogo em config/games/NOME.yaml (ex.: minecraft)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="CAMINHO=VALOR",
        help="sobrescreve um campo, valor lido como YAML (repetível): --set filters.ema.alpha=0.4",
    )
    parser.add_argument("--camera", type=int, help="índice da câmera (padrão: camera.index)")
    parser.add_argument(
        "--camera-backend",
        choices=("dshow", "msmf", "any"),
        help="backend do OpenCV para a câmera (padrão: camera.backend)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="pasta com o modelo pose_landmarker_<variante>.task (padrão: pose.model_dir)",
    )
    parser.add_argument(
        "--model-variant",
        choices=("lite", "full", "heavy"),
        help="variante do modelo de pose (padrão: pose.variant)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="nível de log (padrão: logging.level)",
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


def flag_overrides(args: argparse.Namespace) -> list[str]:
    """Traduz os atalhos da CLI em ``caminho=valor`` (valores citados como YAML/JSON)."""
    pairs: list[tuple[str, object]] = [
        ("camera.index", args.camera),
        ("camera.backend", args.camera_backend),
        ("pose.model_dir", None if args.model_dir is None else str(args.model_dir)),
        ("pose.variant", args.model_variant),
        ("logging.level", args.log_level),
        ("logging.file", args.log_file),
        ("overlay.enabled", False if args.no_overlay else None),
    ]
    return [f"{path}={json.dumps(value)}" for path, value in pairs if value is not None]


def resolve_config(args: argparse.Namespace) -> tuple[AppConfig, GameProfile | None]:
    """Camadas default <- jogo <- CLI; os atalhos vêm antes dos ``--set`` (que vencem)."""
    return load_config(
        default_path=args.config,
        game=args.game,
        overrides=[*flag_overrides(args), *args.overrides],
    )


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
    try:
        config, profile = resolve_config(args)
    except ConfigError as exc:
        configure_logging("INFO", None)
        _log.error("Configuração inválida: %s", exc)
        return EXIT_USAGE
    configure_logging(config.logging.level, config.logging.file)
    # Imports tardios: `--help` não carrega OpenCV nem MediaPipe.
    from inmotion.app import (
        AppDeps,
        OpenCvDisplay,
        pose_filter_from_config,
        run,
    )
    from inmotion.capture.camera import Camera
    from inmotion.clock import MonotonicClock
    from inmotion.config.schema import validate_profile
    from inmotion.gestures.factory import BODY_GESTURE_NAMES
    from inmotion.input.hotkeys import WindowsHotkeys
    from inmotion.input.keyboard_mouse import KeyboardMouseBackend
    from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError

    safety_config = config.safety
    backend = KeyboardMouseBackend()
    if profile is not None:
        try:
            validate_profile(profile, BODY_GESTURE_NAMES, backend.supported_keys())
        except ConfigError as exc:
            _log.error("Perfil de jogo inválido: %s", exc)
            return EXIT_USAGE
        _log.info("Perfil de jogo: %s", profile.name)
    # O modelo é checado ANTES de abrir a câmera: sem ele, nada de hardware é tocado.
    try:
        estimator = MediaPipePoseEstimator(config.pose)
    except ModelNotFoundError as exc:
        _log.error("%s", exc)
        return EXIT_USAGE

    clock = MonotonicClock()
    camera = Camera(config.camera, clock)
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
            backend=backend,
            hotkeys=WindowsHotkeys(frozenset({safety_config.pause_key, safety_config.exit_key})),
            clock=clock,
            display=OpenCvDisplay() if config.overlay.enabled else None,
            pose_filter=pose_filter_from_config(config.filters),
            exit_zone=config.exit_zone,
            config=config,
        )
        if profile is not None:
            deps.profile = profile
    except BaseException:
        # Falhou antes do run(): quem fecha a câmera e o estimador somos nós.
        camera.close()
        estimator.close()
        raise
    _log.info("Rodando: %s pausa/retoma, %s sai.", safety_config.pause_key, safety_config.exit_key)
    return run(deps, safety_config)


if __name__ == "__main__":
    sys.exit(main())
