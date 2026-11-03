"""Demonstração visual de gravações, sem webcam, modelo de pose ou input real.

Uso: ``python -m tools.simulate``. ``--headless`` executa sem abrir janela.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.pipeline import StepResult
from inmotion.recorder import load_recording
from inmotion.recording import Recording
from inmotion.replay import ReplayResult, run_replay
from inmotion.safety import SafetyState
from inmotion.vision.pose import POSE_CONNECTIONS
from tools.replay import (
    add_config_arguments,
    build_pipeline,
    resolve_settings,
    summarize,
    write_report,
)

logger = logging.getLogger("simulate")
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sessions"
SCENARIOS = ("march_2m", "march_1m5", "crouch_2m", "crouch_1m5")
WINDOW = "InMotion - Simulador"
_BG = (27, 22, 19)
_TEXT = (238, 232, 225)
_MUTED = (170, 160, 145)
_ACCENT = (166, 227, 90)


def render_frame(
    recording: Recording,
    result: StepResult,
    replay: ReplayResult,
    *,
    paused: bool = False,
    finished: bool = False,
) -> NDArray[np.uint8]:
    """Desenha esqueleto, rótulos, comandos e histórico em uma imagem independente."""
    out = np.full((640, 1040, 3), _BG, dtype=np.uint8)

    def label(text: str, x: int, y: int, scale: float = 0.55, color: tuple = _TEXT) -> None:
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    label("InMotion / Simulador", 24, 36, 0.8)
    label("Gravacao de landmarks - sem camera e sem enviar teclas ou cliques", 24, 65, color=_MUTED)
    cv2.rectangle(out, (24, 90), (664, 570), (43, 36, 30), -1)
    pose = result.pose
    if pose is not None:
        points = pose.landmarks
        visible = (
            np.isfinite(points[:, :2]).all(axis=1)
            & (points[:, 3] >= 0.5)
            & (points[:, :2] >= 0).all(axis=1)
            & (points[:, :2] <= 1).all(axis=1)
        )
        pixels = {
            i: (24 + round(float(p[0]) * 640), 90 + round(float(p[1]) * 480))
            for i, p in enumerate(points)
            if visible[i]
        }
        for a, b in POSE_CONNECTIONS:
            if a in pixels and b in pixels:
                cv2.line(out, pixels[a], pixels[b], _ACCENT, 2, cv2.LINE_AA)
        for point in pixels.values():
            cv2.circle(out, point, 3, _TEXT, -1, cv2.LINE_AA)
    else:
        label("Sem pose neste frame", 210, 330, color=_MUTED)
    state = "FIM" if finished else "PAUSADO" if paused else "REPRODUZINDO"
    label(state, 690, 112, 0.6, _ACCENT)
    start = float(recording.t[0]) if len(recording.t) else 0.0
    label(f"Tempo: {result.t - start:.2f} / {recording.duration_s:.2f} s", 690, 145)
    safety_label = {
        SafetyState.ACTIVE: "ATIVO",
        SafetyState.PAUSED: "PAUSADO",
        SafetyState.LOST: "SEM POSE",
    }[result.safety_state]
    label(f"Seguranca: {safety_label}", 690, 176)
    active_labels = [lab.gesture for lab in recording.labels if lab.start <= result.t < lab.end]
    label("Rotulo da coleta:", 690, 220, color=_MUTED)
    label(", ".join(active_labels)[:35] or "--", 690, 247)
    label("Comandos simulados ativos:", 690, 290, color=_MUTED)
    keys = sorted(result.pressed) if not finished else []
    for i, key in enumerate(keys[:7]):
        label(key, 706, 320 + i * 25, 0.65, _ACCENT)
    if not keys:
        label("Nenhum", 706, 320)
    label("Ultimos comandos:", 690, 485, color=_MUTED)
    recent = [(t, kind, key) for t, kind, key in replay.inputs if t <= result.t][-3:]
    for i, (t, kind, key) in enumerate(recent):
        label(f"{t - start:6.2f}s  {kind} {key}", 690, 510 + i * 25, 0.48)
    label("ESPACO: pausar/retomar   R: reiniciar   N: proximo frame   ESC: sair", 24, 607, 0.51)
    return out


def show(recording: Recording, replay: ReplayResult, speed: float) -> None:
    """Reproduz snapshots já calculados; apenas a navegação usa o relógio real."""
    snapshots = replay.snapshots or ()
    if not snapshots:
        raise ValueError("a gravação não contém frames")
    index = 0
    paused = False
    start_wall = time.perf_counter()
    start_t = snapshots[0].t
    try:
        while True:
            if not paused:
                target = start_t + (time.perf_counter() - start_wall) * speed
                while index + 1 < len(snapshots) and snapshots[index + 1].t <= target:
                    index += 1
            finished = index == len(snapshots) - 1
            frame = render_frame(
                recording, snapshots[index], replay, paused=paused, finished=finished
            )
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(15) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("r"):
                index, paused = 0, False
                start_t, start_wall = snapshots[0].t, time.perf_counter()
            elif key == ord(" "):
                paused = not paused
                start_t, start_wall = snapshots[index].t, time.perf_counter()
            elif key == ord("n"):
                paused = True
                index = min(index + 1, len(snapshots) - 1)
    finally:
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simula controles sem câmera nem jogo.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recording", type=Path, help="gravação .npz própria")
    source.add_argument("--scenario", choices=SCENARIOS, help="sessão incluída (padrão: march_2m)")
    parser.add_argument("--headless", action="store_true", help="executa sem janela")
    parser.add_argument("--speed", type=float, default=1.0, help="velocidade de reprodução (> 0)")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed deve ser finito e > 0")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = args.recording or FIXTURES / f"{args.scenario or 'march_2m'}.npz"
    try:
        recording = load_recording(path)
        if not len(recording.t):
            raise ValueError("a gravação não contém frames")
        config, profile = resolve_settings(args)
        result = run_replay(
            recording,
            partial(build_pipeline, app_config=config, profile=profile),
            keep_snapshots=not args.headless,
        )
        for line in summarize(len(recording.t), recording.duration_s, result):
            logger.info(line)
        if args.output is not None:
            write_report(args.output, result, len(recording.t), recording.duration_s)
        if not args.headless:
            show(recording, result, args.speed)
    except (OSError, ValueError, cv2.error) as exc:
        logger.error("falha na simulação: %s", exc)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
