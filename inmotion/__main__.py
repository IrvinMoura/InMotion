"""Interface de linha de comando do InMotion."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from inmotion.config.loader import DEFAULT_CONFIG_PATH, load_config
from inmotion.config.schema import AppConfig, GameProfile
from inmotion.config.validate import ConfigError

_log = logging.getLogger("inmotion")

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EXIT_USAGE = 2  # configuração ou ambiente inválido (modelo ausente, câmera indisponível)


def build_parser() -> argparse.ArgumentParser:
    """Os valores padrão das opções vêm do YAML."""
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
    parser.add_argument("--profile", type=Path, help="YAML do usuário, aplicado após o jogo")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="CAMINHO=VALOR",
        help="sobrescreve um campo, valor lido como YAML (repetível): --set filters.ema.alpha=0.4",
    )
    parser.add_argument("--camera", type=int, help="índice da câmera (padrão: camera.index)")
    parser.add_argument("--voice", action="store_true", help="ativa comandos de voz offline")
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
        ("voice.enabled", True if args.voice else None),
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
        profile_path=args.profile,
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
    from inmotion.capture.microphone import MicrophoneCommands
    from inmotion.clock import MonotonicClock
    from inmotion.config.schema import validate_profile
    from inmotion.gestures.factory import BODY_GESTURE_NAMES
    from inmotion.input.hotkeys import WindowsHotkeys
    from inmotion.input.keyboard_mouse import KeyboardMouseBackend
    from inmotion.session import DEFAULT_PROFILE
    from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError
    from inmotion.vision.preprocess import Preprocessor

    safety_config = config.safety
    profile = profile if profile is not None else DEFAULT_PROFILE
    try:
        backend = KeyboardMouseBackend()
        hotkeys = WindowsHotkeys(frozenset({safety_config.pause_key, safety_config.exit_key}))
        supported = backend.supported_keys()
        validate_profile(profile, BODY_GESTURE_NAMES, supported)
        if config.voice.enabled:
            for key in (config.voice.inventory_key, config.voice.menu_key):
                if (
                    key not in supported
                    or key in {safety_config.pause_key, safety_config.exit_key}
                    or key.startswith("mouse_")
                ):
                    raise ConfigError("voice", f"tecla de menu inválida ou reservada: {key!r}")
        for gesture, binding in profile.bindings.items():
            if binding.input in {safety_config.pause_key, safety_config.exit_key}:
                raise ConfigError(
                    f"bindings.{gesture}.input", "conflita com uma tecla de segurança"
                )
    except (ValueError, OSError) as exc:
        _log.error("Configuração de input inválida: %s", exc)
        return EXIT_USAGE
    _log.info("Perfil de jogo: %s", profile.name)
    clock = MonotonicClock()
    with ExitStack() as startup:
        try:
            estimator = MediaPipePoseEstimator(config.pose)
        except (ModelNotFoundError, RuntimeError, ValueError, OSError) as exc:
            _log.error("%s", exc)
            return EXIT_USAGE
        startup.callback(estimator.close)
        camera = Camera(config.camera, clock)
        startup.callback(camera.close)
        try:
            camera.start()
        except (OSError, ValueError, RuntimeError) as exc:
            _log.error("Falha ao abrir a câmera: %s", exc)
            return EXIT_USAGE
        except KeyboardInterrupt:
            return 0
        _log.info("Câmera aberta: %dx%d a %.0f FPS", *camera.actual)

        deps = AppDeps(
            source=camera,
            estimator=estimator,
            backend=backend,
            hotkeys=hotkeys,
            clock=clock,
            display=(
                OpenCvDisplay(handedness=config.handedness, mirror=config.overlay.mirror)
                if config.overlay.enabled
                else None
            ),
            pose_filter=pose_filter_from_config(config.filters),
            exit_zone=config.exit_zone,
            config=config,
            preprocess=Preprocessor(config.preprocess).apply,
            profile=profile,
        )
        if config.voice.enabled:
            try:
                deps.voice = MicrophoneCommands(config.voice, clock)
            except Exception as exc:
                _log.error("Falha ao iniciar a voz: %s", exc)
                return EXIT_USAGE
            startup.callback(deps.voice.close)
        startup.pop_all()  # run() assume e fecha todos os recursos da sessão.
    _log.info("Rodando: %s pausa/retoma, %s sai.", safety_config.pause_key, safety_config.exit_key)
    try:
        return run(deps, safety_config)
    except KeyboardInterrupt:
        _log.info("Sessão encerrada pelo teclado")
        return 0
    except Exception:
        _log.exception("Falha durante a sessão")
        return 1


if __name__ == "__main__":
    sys.exit(main())
