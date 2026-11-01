"""Sessão guiada de validação: janela grande, instruções, timer e barras dos gestos.

Uso::

    python -m tools.guided --person p1 --distance 2.0

Pensada para quem está a ~2 m do PC e não consegue ler o terminal. **Nenhum input é
enviado**: as teclas que seriam apertadas aparecem na tela. A sessão grava um ``.npz``
rotulado com a janela de cada passo (spec 011) e mostra no fim um resumo OK/FALHOU por
passo, também salvo em ``.json`` ao lado da gravação. ESC ou F9 encerram a qualquer
momento (o que já foi gravado é salvo).

Ferramenta de composição: a lógica testável (roteiro, pontuação, rótulos, desenho) fica
em funções puras; ``main`` liga câmera, estimador, pipeline e janela.
"""

from __future__ import annotations

import argparse
import json
import logging
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.config.loader import load_config
from inmotion.gestures.base import GestureState
from inmotion.input.backend import KeyName
from inmotion.recording import SCHEMA_VERSION, LabelInterval, RecordingMeta

logger = logging.getLogger("guided")

PREP_S = 3.0  # contagem "PRÓXIMO: ..." antes de cada passo (não pontuada)
GRACE_S = 0.6  # tolerância de latência: press logo após o fim do passo ainda conta
READ_TIMEOUT_S = 0.5
SUMMARY_HOLD_S = 20.0  # tempo máximo do resumo na tela
WINDOW_NAME = "InMotion - sessao guiada"
CANVAS = (1280, 720)
_ESC = 27


class Phase(Enum):
    PREP = "prep"
    RUN = "run"
    DONE = "done"


@dataclass(frozen=True)
class GuidedStep:
    """Um passo do roteiro: ``gesture=None`` com ``expected=0`` é "parado" (nenhuma tecla);
    ``expected=None`` não é pontuado (ex.: posicionar-se)."""

    label: str
    title: str
    duration_s: float
    gesture: str | None
    expected: int | None


def default_steps(handedness: str) -> tuple[GuidedStep, ...]:
    """Roteiro padrão (~80 s): parado, socos, pulos, agachar, inclinar, parado."""
    dominant, other = ("DIREITO", "ESQUERDO") if handedness == "right" else ("ESQUERDO", "DIREITO")
    return (
        GuidedStep("setup", "VA PARA ~2 m, CORPO INTEIRO NA TELA", 6.0, None, None),
        GuidedStep("rest", "FIQUE PARADO", 8.0, None, 0),
        GuidedStep("punch_dominant", f"3 SOCOS - BRACO {dominant}", 8.0, "attack", 3),
        GuidedStep("punch_other", f"3 SOCOS - BRACO {other}", 8.0, "use", 3),
        GuidedStep("jump", "3 PULOS", 8.0, "jump", 3),
        GuidedStep("crouch", "AGACHE E SEGURE", 6.0, "crouch", 1),
        GuidedStep("lean", "INCLINE PARA FRENTE E SEGURE", 6.0, "walk", 1),
        GuidedStep("rest_end", "FIQUE PARADO", 8.0, None, 0),
    )


class Schedule:
    """Linha do tempo: para cada passo, ``prep_s`` de aviso e depois a execução."""

    def __init__(self, steps: Sequence[GuidedStep], prep_s: float = PREP_S) -> None:
        self.steps = tuple(steps)
        self._prep = prep_s
        starts: list[float] = []
        t = 0.0
        for step in self.steps:
            starts.append(t + prep_s)
            t += prep_s + step.duration_s
        self._run_starts = tuple(starts)
        self.total_s = t

    def run_window(self, i: int) -> tuple[float, float]:
        start = self._run_starts[i]
        return start, start + self.steps[i].duration_s

    def at(self, t: float) -> tuple[int, Phase, float]:
        """(passo, fase, segundos restantes na fase) no instante ``t`` desde o início."""
        for i in range(len(self.steps)):
            start, end = self.run_window(i)
            if t < start:
                return i, Phase.PREP, start - t
            if t < end:
                return i, Phase.RUN, end - t
        return len(self.steps) - 1, Phase.DONE, 0.0


@dataclass(frozen=True)
class StepScore:
    """``got``: presses da tecla esperada; ``extras``: outras teclas; ``ok`` None = sem nota."""

    label: str
    title: str
    expected: int | None
    got: int
    extras: tuple[str, ...]
    ok: bool | None


def score_steps(
    schedule: Schedule,
    presses: Sequence[tuple[float, KeyName]],
    gesture_keys: Mapping[str, KeyName],
    grace_s: float = GRACE_S,
    releases: Sequence[tuple[float, KeyName]] | None = None,
) -> list[StepScore]:
    """Compara as teclas que teriam sido apertadas com o esperado de cada passo.

    A tecla esperada conta em ``[início, fim + grace_s)``; qualquer outra tecla apertada
    durante a execução do passo, **ou apertada antes e ainda segurada quando ele começa**,
    é "extra" e reprova (só com ``releases``; ``None`` = soltas não registradas, não checa).
    Passo com ``expected=None`` não é pontuado.
    """
    scores: list[StepScore] = []
    for i, step in enumerate(schedule.steps):
        start, end = schedule.run_window(i)
        key = gesture_keys.get(step.gesture) if step.gesture else None
        got = sum(1 for t, k in presses if k == key and start <= t < end + grace_s)
        held = () if releases is None else _held_at(presses, releases, start, exclude=key)
        extras = held + tuple(k for t, k in presses if k != key and start <= t < end)
        ok = None if step.expected is None else (got == step.expected and not extras)
        scores.append(StepScore(step.label, step.title, step.expected, got, extras, ok))
    return scores


def _held_at(
    presses: Sequence[tuple[float, KeyName]],
    releases: Sequence[tuple[float, KeyName]],
    t: float,
    exclude: KeyName | None,
) -> tuple[KeyName, ...]:
    """Teclas apertadas antes de ``t`` e ainda não soltas em ``t`` (ordem de aperto)."""
    held: dict[KeyName, None] = {}
    events = sorted([(tp, 1, k) for tp, k in presses] + [(tr, 0, k) for tr, k in releases])
    for te, is_press, k in events:
        if te >= t:
            break
        if is_press:
            held[k] = None
        else:
            held.pop(k, None)
    return tuple(k for k in held if k != exclude)


def step_labels(schedule: Schedule) -> list[LabelInterval]:
    """Rótulos da gravação: a janela de execução de cada passo pontuado."""
    return [
        LabelInterval(step.label, *schedule.run_window(i))
        for i, step in enumerate(schedule.steps)
        if step.expected is not None
    ]


def step_count(
    schedule: Schedule,
    index: int,
    presses: Sequence[tuple[float, KeyName]],
    gesture_keys: Mapping[str, KeyName],
) -> int | None:
    """Quantas vezes a tecla do passo ``index`` já saiu desde o início dele (contador)."""
    step = schedule.steps[index]
    key = gesture_keys.get(step.gesture) if step.gesture else None
    if key is None:
        return None
    start, _ = schedule.run_window(index)
    return sum(1 for t, k in presses if k == key and t >= start)


def gesture_totals(
    presses: Sequence[tuple[float, KeyName]], gesture_keys: Mapping[str, KeyName]
) -> dict[str, int]:
    """Total de presses por gesto na sessão inteira (só gestos com tecla no perfil)."""
    by_key = {key: name for name, key in gesture_keys.items()}
    totals: dict[str, int] = {}
    for _, key in presses:
        if key in by_key:
            totals[by_key[key]] = totals.get(by_key[key], 0) + 1
    return totals


def ascii_text(text: str) -> str:
    """A fonte Hershey do OpenCV só desenha ASCII: remove os acentos."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


# ----------------------------------------------------------------------- desenho

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (0, 200, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 220, 255)
_GRAY = (140, 140, 140)
_PANEL_W = 320


def _text(
    img: NDArray[np.uint8],
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thick: int,
) -> None:
    """Texto com contorno preto (legível sobre qualquer fundo)."""
    text = ascii_text(text)
    cv2.putText(img, text, org, _FONT, scale, _BLACK, thick + 4, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thick, cv2.LINE_AA)


def _bar(
    img: NDArray[np.uint8],
    y: int,
    name: str,
    value: float | None,
    on: float,
    off: float,
    active: bool,
    total: int = 0,
) -> None:
    """Barra da feature com marcas de θ_on (vermelho) e θ_off (amarelo) e o total."""
    x0, x1 = img.shape[1] - _PANEL_W + 20, img.shape[1] - 20
    top = max(1.0, on * 1.5)
    _text(img, f"{name.upper()} x{total}", (x0, y), 0.8, _GREEN if active else _WHITE, 2)
    val = "--" if value is None else f"{value:+.2f}"
    _text(img, val, (x1 - 95, y), 0.8, _WHITE, 2)
    by = y + 10
    cv2.rectangle(img, (x0, by), (x1, by + 26), _GRAY, 2)
    if value is not None:
        frac = min(max(value / top, 0.0), 1.0)
        fill = _GREEN if active else _WHITE
        cv2.rectangle(img, (x0, by), (x0 + round(frac * (x1 - x0)), by + 26), fill, -1)
    for mark, color in ((on, _RED), (off, _YELLOW)):
        mx = x0 + round(min(mark / top, 1.0) * (x1 - x0))
        cv2.line(img, (mx, by - 4), (mx, by + 30), color, 3)


def render_frame(
    frame: NDArray[np.uint8],
    title: str,
    countdown_s: float,
    phase: Phase,
    state_text: str,
    bars: Sequence[tuple[str, float | None, float, float, bool]],
    last_keys: Sequence[str],
    size: tuple[int, int] = CANVAS,
    counter: tuple[int, int | None] | None = None,
    totals: Mapping[str, int] | None = None,
) -> NDArray[np.uint8]:
    """Tela da sessão: câmera à esquerda, barras à direita, instrução e timer grandes.

    ``counter`` = (feitos, esperados) do passo atual, desenhado bem grande; ``totals``
    mostra quantas vezes cada gesto já saiu na sessão, ao lado da barra dele.
    """
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cam_w = width - _PANEL_W
    canvas[:, :cam_w] = cv2.resize(frame, (cam_w, height))
    heading = f"PROXIMO: {title}" if phase is Phase.PREP else title
    color = _YELLOW if phase is Phase.PREP else _WHITE
    _text(canvas, heading, (20, 60), 1.4, color, 3)
    _text(canvas, f"{countdown_s:4.1f}", (20, height - 30), 4.0, color, 8)
    state_color = _GREEN if state_text == "ATIVO" else _RED
    _text(canvas, state_text, (cam_w - 290, height - 40), 1.4, state_color, 3)
    if last_keys:
        _text(canvas, "TECLA: " + " ".join(last_keys), (20, 130), 1.6, _GREEN, 4)
    if counter is not None:
        done, expected = counter
        goal = "" if expected is None else f"/{expected}"
        hit = expected is not None and done >= expected
        _text(canvas, f"{done}{goal}", (cam_w - 330, 200), 4.0, _GREEN if hit else _WHITE, 8)
    for i, (name, value, on, off, active) in enumerate(bars):
        total = 0 if totals is None else totals.get(name, 0)
        _bar(canvas, 60 + i * 110, name, value, on, off, active, total)
    return canvas


def render_summary(
    scores: Sequence[StepScore], size: tuple[int, int] = CANVAS
) -> NDArray[np.uint8]:
    """Tela final: uma linha por passo pontuado, OK em verde, FALHOU em vermelho."""
    width, height = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    _text(canvas, "RESULTADO (ESC ou F9 fecha)", (30, 60), 1.3, _WHITE, 3)
    y = 130
    for s in scores:
        if s.ok is None:
            continue
        mark, color = ("OK", _GREEN) if s.ok else ("FALHOU", _RED)
        extra = f"  extras: {' '.join(s.extras)}" if s.extras else ""
        _text(canvas, f"{mark:7s}{s.title}: {s.got}/{s.expected}{extra}", (30, y), 1.0, color, 2)
        y += 70
    return canvas


# ----------------------------------------------------------------------- execução


class _LogBackend:
    """``InputBackend`` que só registra (nenhum input real). ``t`` é o relógio da sessão."""

    def __init__(self) -> None:
        self.t = 0.0
        self.presses: list[tuple[float, KeyName]] = []
        self.releases: list[tuple[float, KeyName]] = []

    def press(self, key: KeyName) -> None:
        self.presses.append((self.t, key))

    def release(self, key: KeyName) -> None:
        self.releases.append((self.t, key))


def _counter(
    schedule: Schedule,
    index: int,
    phase: Phase,
    presses: Sequence[tuple[float, KeyName]],
    keys: Mapping[str, KeyName],
) -> tuple[int, int | None] | None:
    """Contador do passo em execução (no aviso "PRÓXIMO" não há contador)."""
    if phase is not Phase.RUN:
        return None
    done = step_count(schedule, index, presses, keys)
    return None if done is None else (done, schedule.steps[index].expected)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sessão guiada de validação (sem input real).")
    parser.add_argument("--person", default="p1", help="id anônimo da pessoa")
    parser.add_argument("--distance", type=float, default=None, help="distância da câmera (m)")
    parser.add_argument("--game", default="minecraft", help="perfil de jogo (config/games)")
    parser.add_argument(
        "--out", type=Path, default=None, help="gravação .npz (padrão: recordings/)"
    )
    parser.add_argument("--fullscreen", action="store_true", help="janela em tela cheia")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Imports pesados aqui: o --help não abre câmera nem carrega o MediaPipe.
    from inmotion.app import pose_filter_from_config
    from inmotion.capture.camera import Camera
    from inmotion.clock import MonotonicClock
    from inmotion.fps import FpsMeter
    from inmotion.gestures.factory import build_body_gestures
    from inmotion.input.hotkeys import WindowsHotkeys
    from inmotion.overlay import render
    from inmotion.pipeline import Pipeline
    from inmotion.recorder import Recorder
    from inmotion.safety import SafetyAction, SafetyManager, SafetyState, TrackedInput
    from inmotion.vision.landmarks import Side
    from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError
    from inmotion.vision.preprocess import Preprocessor
    from tools.record import compute_config_hash

    config, profile = load_config(game=args.game)
    if profile is None:
        logger.error("perfil %r não encontrado", args.game)
        return 2
    keys = {name: b.input for name, b in profile.bindings.items()}
    backend = _LogBackend()
    safety = SafetyManager(TrackedInput(backend, config.safety.always_release), config.safety)
    gestures = build_body_gestures(
        profile.bindings, Side(config.handedness), config.gestures, config.geometry
    )
    pipeline = Pipeline(
        safety,
        gestures,
        FpsMeter(),
        pose_filter_from_config(config.filters),
        profile=profile,
        mapper_params=config.mapper,
        baseline_params=config.baseline,
        geometry=config.geometry,
    )
    schedule = Schedule(default_steps(config.handedness))
    preprocess = Preprocessor(config.preprocess).apply
    try:
        estimator = MediaPipePoseEstimator(config.pose)
    except ModelNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    clock = MonotonicClock()
    camera = Camera(config.camera, clock)
    hotkeys = WindowsHotkeys(frozenset({config.safety.pause_key, config.safety.exit_key}))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out: Path = args.out or Path("recordings") / f"guided_{args.person}_{stamp}.npz"
    meta = RecordingMeta(
        schema_version=SCHEMA_VERSION,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        person=args.person,
        distance_m=args.distance,
        lighting="outra",
        camera=f"{config.camera.backend}:{config.camera.index}",
        config_hash=compute_config_hash(config.camera, config.pose.variant, keys),
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *CANVAS)
    if args.fullscreen:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    recorder: Recorder | None = None
    t0: float | None = None
    t_rel = 0.0
    aborted = False
    camera.start()
    try:
        with estimator:
            while True:
                if safety.on_hotkeys(hotkeys.poll()) is SafetyAction.EXIT:
                    aborted = True
                    break
                packet = camera.read(READ_TIMEOUT_S)
                if packet is None:
                    continue
                if t0 is None:
                    t0 = packet.t
                    height, width = packet.image.shape[:2]
                    recorder = Recorder(meta, (width, height))
                t_rel = packet.t - t0
                index, phase, remaining = schedule.at(t_rel)
                if phase is Phase.DONE:
                    break
                pose = estimator.estimate(preprocess(packet.image), packet.t)
                assert recorder is not None
                recorder.add(t_rel, pose)
                backend.t = t_rel
                result = pipeline.step(pose, packet.t)
                bars = [
                    (
                        g.name,
                        g.feature_value,
                        config.gestures[g.name].theta_on,
                        config.gestures[g.name].theta_off,
                        g.snapshot().state is GestureState.ACTIVE,
                    )
                    for g in gestures
                ]
                recent = [k.upper() for t, k in backend.presses if t_rel - t < 1.0]
                state = {SafetyState.ACTIVE: "ATIVO", SafetyState.PAUSED: "PAUSADO"}.get(
                    result.safety_state, "SEM POSE"
                )
                canvas = render_frame(
                    render(packet.image, result),
                    schedule.steps[index].title,
                    remaining,
                    phase,
                    state,
                    bars,
                    recent,
                    counter=_counter(schedule, index, phase, backend.presses, keys),
                    totals=gesture_totals(backend.presses, keys),
                )
                cv2.imshow(WINDOW_NAME, canvas)
                if cv2.waitKey(1) & 0xFF == _ESC:
                    aborted = True
                    break
    finally:
        camera.close()
        if recorder is not None:
            labels = [lb for lb in step_labels(schedule) if lb.end <= t_rel]
            saved = recorder.save(out, labels)
            logger.info("gravação salva em %s", saved)

    scores = score_steps(schedule, backend.presses, keys, releases=backend.releases)
    summary = out.with_suffix(".json")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"aborted": aborted, "steps": [asdict(s) for s in scores]}, indent=2),
        encoding="utf-8",
    )
    for s in scores:
        if s.ok is not None:
            mark = "OK    " if s.ok else "FALHOU"
            extra = f" extras={list(s.extras)}" if s.extras else ""
            logger.info("%s %s: %d/%d%s", mark, s.title, s.got, s.expected, extra)
    logger.info("resumo salvo em %s", summary)
    if not aborted:
        cv2.imshow(WINDOW_NAME, render_summary(scores))
        end = clock.now() + SUMMARY_HOLD_S
        while clock.now() < end:
            if cv2.waitKey(100) & 0xFF == _ESC or config.safety.exit_key in hotkeys.poll():
                break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
