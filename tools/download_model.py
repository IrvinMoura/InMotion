"""Baixa o modelo ``pose_landmarker_<variante>.task`` para ``models/`` (spec 003, RF4).

Uso: ``python tools/download_model.py [--variant lite|full|heavy] [--force]``.
O arquivo ``.task`` não é versionado no git (binário de vários MB).
"""

from __future__ import annotations

import argparse
import logging
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("download_model")

VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)
_CHUNK_BYTES = 1 << 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baixa o modelo do PoseLandmarker.")
    parser.add_argument("--variant", choices=VARIANTS, default="full", help="variante do modelo")
    parser.add_argument("--url", default=None, help="URL alternativa (substitui a oficial)")
    parser.add_argument("--force", action="store_true", help="sobrescreve se já existir")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="pasta de destino")
    return parser


def main(argv: Sequence[str] | None = None, urlopen: Callable[..., Any] | None = None) -> int:
    """Executa o download. Devolve 0 em sucesso e 1 em falha (2 para argumentos inválidos)."""
    args = _build_parser().parse_args(argv)
    opener = urlopen if urlopen is not None else urllib.request.urlopen
    url: str = args.url or URL_TEMPLATE.format(variant=args.variant)
    target: Path = args.models_dir / f"pose_landmarker_{args.variant}.task"

    if target.exists() and not args.force:
        logger.error("%s já existe; use --force para sobrescrever.", target)
        return 1

    partial = target.with_name(target.name + ".part")
    try:
        args.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Baixando %s", url)
        with opener(url, timeout=30) as response, partial.open("wb") as out:
            while chunk := response.read(_CHUNK_BYTES):
                out.write(chunk)
        if partial.stat().st_size == 0:
            raise OSError("o servidor devolveu um modelo vazio")
        partial.replace(target)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        logger.error("Falha ao baixar o modelo: %s", exc)
        return 1
    logger.info("Modelo salvo em %s", target)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
