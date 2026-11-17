"""Replay de uma gravação ``.npz`` pelo mesmo pipeline do app (spec 011 T5).

Uso: ``python -m tools.replay --recording sessao.npz``. Não envia input real.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from inmotion.config.loader import DEFAULT_CONFIG_PATH, load_config
from inmotion.config.schema import AppConfig, GameProfile, validate_profile
from inmotion.gestures.factory import BODY_GESTURE_NAMES
from inmotion.input.backend import InputBackend
from inmotion.pipeline import Pipeline
from inmotion.recorder import load_recording
from inmotion.replay import ReplayResult, run_replay
from inmotion.safety import SafetyConfig, SafetyManager, TrackedInput
from inmotion.session import DEFAULT_PROFILE, create_pipeline, pose_filter_from_config

logger = logging.getLogger("replay")


def build_pipeline(
    backend: InputBackend,
    config: SafetyConfig | None = None,
    *,
    app_config: AppConfig | None = None,
    profile: GameProfile = DEFAULT_PROFILE,
) -> tuple[Pipeline, SafetyManager]:
    """Mesma montagem do app, incluindo filtros, lateralidade e parâmetros do YAML."""
    app_config = app_config if app_config is not None else load_config()[0]
    if config is not None:
        app_config = replace(app_config, safety=config)
    safety = SafetyManager(
        TrackedInput(backend, app_config.safety.always_release),
        app_config.safety,
    )
    pipeline = create_pipeline(
        safety,
        app_config,
        profile,
        pose_filter_from_config(app_config.filters),
    )
    return pipeline, safety


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Opções iguais no replay e no simulador."""
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML base")
    parser.add_argument("--game", help="perfil em config/games/<nome>.yaml")
    parser.add_argument("--profile", type=Path, help="YAML do usuário, aplicado após o jogo")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides", help="CAMINHO=VALOR"
    )
    parser.add_argument("--output", type=Path, help="salva relatório JSON")


def resolve_settings(args: argparse.Namespace) -> tuple[AppConfig, GameProfile]:
    config, profile = load_config(
        default_path=args.config,
        game=args.game,
        profile_path=args.profile,
        overrides=args.overrides,
    )
    profile = profile if profile is not None else DEFAULT_PROFILE
    # Sem backend físico: valida estrutura e gestos; teclas são verificadas na execução real.
    validate_profile(
        profile, BODY_GESTURE_NAMES, frozenset(b.input for b in profile.bindings.values())
    )
    return config, profile


def write_report(path: Path, result: ReplayResult, frames: int, duration_s: float) -> None:
    """Relatório determinístico para comparar execuções e inspecionar eventos."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "frames": frames,
                "duration_s": duration_s,
                "inputs": [{"t": t, "kind": kind, "key": key} for t, kind, key in result.inputs],
                "events": [
                    {**asdict(event), "kind": event.kind.name.lower()} for event in result.events
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument("--timeline", action="store_true", help="mostra cada comando com timestamp")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        recording = load_recording(args.recording)
        config, profile = resolve_settings(args)
        result = run_replay(recording, partial(build_pipeline, app_config=config, profile=profile))
        if args.output is not None:
            write_report(args.output, result, len(recording.t), recording.duration_s)
    except (OSError, ValueError) as exc:
        logger.error("falha no replay: %s", exc)
        return 1
    for line in summarize(int(recording.t.shape[0]), recording.duration_s, result):
        logger.info(line)
    if args.timeline:
        for t, kind, key in result.inputs:
            logger.info("%.3f s | %s %s", t, kind, key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
