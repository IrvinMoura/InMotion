"""Replay de uma gravação ``.npz`` pelo mesmo pipeline do app (spec 011 T5).

Uso: ``python tools/replay.py --recording sessao.npz``. Imprime (no log) frames, duração e
as entradas press/release por tecla. Pendência: vídeo com overlay (``--video``).
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from inmotion.app import DEFAULT_PROFILE
from inmotion.fps import FpsMeter
from inmotion.gestures.factory import build_body_gestures
from inmotion.input.backend import InputBackend
from inmotion.pipeline import Pipeline
from inmotion.recorder import load_recording
from inmotion.replay import ReplayResult, run_replay
from inmotion.safety import SafetyConfig, SafetyManager, TrackedInput

logger = logging.getLogger("replay")


def build_pipeline(
    backend: InputBackend, config: SafetyConfig | None = None
) -> tuple[Pipeline, SafetyManager]:
    """Mesmos gestos, teclas e árbitro do app com o perfil padrão (Minecraft, destro).

    Sem ``config``, usa a mesma ``SafetyConfig()`` padrão do app (persistência inclusa).
    """
    config = config if config is not None else SafetyConfig()
    safety = SafetyManager(TrackedInput(backend, config.always_release), config)
    gestures = build_body_gestures(DEFAULT_PROFILE.bindings)
    pipeline = Pipeline(safety, gestures, FpsMeter(), profile=DEFAULT_PROFILE)
    return pipeline, safety


def summarize(frames: int, duration_s: float, result: ReplayResult) -> list[str]:
    """Linhas do resumo: frames, duração e nº de entradas por tecla."""
    presses = Counter(key for _, kind, key in result.inputs if kind == "press")
    releases = Counter(key for _, kind, key in result.inputs if kind == "release")
    lines = [f"frames: {frames}", f"duração: {duration_s:.3f} s"]
    for key in sorted(set(presses) | set(releases)):
        lines.append(f"{key}: {presses[key]} press, {releases[key]} release")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay determinístico de uma gravação.")
    parser.add_argument("--recording", type=Path, required=True, help="arquivo .npz")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        recording = load_recording(args.recording)
    except (OSError, ValueError) as exc:
        logger.error("falha ao ler a gravação: %s", exc)
        return 1
    result = run_replay(recording, build_pipeline)
    for line in summarize(int(recording.t.shape[0]), recording.duration_s, result):
        logger.info(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
