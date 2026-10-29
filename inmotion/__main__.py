"""Ponto de entrada: ``python -m inmotion``. Composição: lê a CLI e configura o logging."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

_log = logging.getLogger("inmotion")

LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
EXIT_USAGE = 2  # configuração ou ambiente inválido (modelo ausente, câmera indisponível)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inmotion",
        description="Controla jogos por movimento corporal usando uma webcam.",
    )
    parser.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão: 0)")
    parser.add_argument(
        "--camera-backend",
        choices=("dshow", "msmf", "any"),
        default="dshow",
        help="backend do OpenCV para a câmera (padrão: dshow)",
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
    _log.error("Estimador de pose ainda não ligado ao aplicativo (spec 003).")
    return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
