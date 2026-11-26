"""Métricas por gesto e gráficos para o relatório (spec 011 T7).

Roda o replay de sessões guiadas (``tools.guided``) com a mesma montagem do app e mede,
por gesto, precisão, recall, FP/min e o tempo até o primeiro disparo de cada passo
(``inmotion.metrics.score_guided``). Também compara os filtros (bruto × EMA × One Euro):
tremor parado (``jitter``) e atraso no soco (``lag_frames``).

Saídas em ``--out`` (padrão ``reports/evaluate``): ``metrics.md``, ``metrics.csv``,
``filters.csv`` e, com o ``matplotlib`` instalado, ``confusion.png`` e
``filters_<gravação>.png``.

Uso::

    python -m tools.evaluate                                  # fixtures de tests/
    python -m tools.evaluate --recordings recordings/guided_*.npz --out reports/e3
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from inmotion.config.loader import DEFAULT_CONFIG_PATH
from inmotion.config.schema import AppConfig, GameProfile
from inmotion.metrics import (
    NO_GESTURE,
    GestureMetrics,
    StepExpectation,
    confusion_counts,
    jitter,
    lag_frames,
    score_guided,
)
from inmotion.recorder import load_recording
from inmotion.recording import Recording
from inmotion.replay import run_replay
from inmotion.session import pose_filter_from_config
from inmotion.vision.landmarks import PoseLandmark
from tools.guided import SCRIPTS, GuidedStep
from tools.replay import build_pipeline, resolve_settings

logger = logging.getLogger("evaluate")

DEFAULT_RECORDINGS = "tests/fixtures/sessions/*.npz"
DEFAULT_OUT = Path("reports/evaluate")
# Metas da Fase 5 (ROADMAP): recall ≥ 90%, FP ≤ 1/min, FPS ≥ 25.
TARGET_RECALL = 0.90
TARGET_FP_PER_MIN = 1.0
TARGET_FPS = 25.0
DURATION_TOL_S = 0.5  # o rótulo de um passo dura o mesmo que o passo do roteiro
FILTER_KINDS = ("none", "ema", "one_euro")
FILTER_NAMES = {"none": "bruto", "ema": "EMA", "one_euro": "One Euro"}
MAX_LAG_FRAMES = 15
_WRIST = {"right": PoseLandmark.RIGHT_WRIST, "left": PoseLandmark.LEFT_WRIST}


@dataclass(frozen=True)
class SessionResult:
    """Resultado de uma gravação: script reconhecido, métricas, confusão e FPS."""

    name: str
    script: str
    distance_m: float | None
    handedness: str
    metrics: Mapping[str, GestureMetrics]
    confusion: Mapping[tuple[str, str], int]
    fps: float | None


@dataclass(frozen=True)
class FilterResult:
    """Tremor parado (px) e atraso no soco (ms) de um filtro numa gravação."""

    recording: str
    kind: str
    jitter_px: float | None
    lag_ms: float | None


def expectations(steps: Sequence[GuidedStep]) -> dict[str, StepExpectation]:
    """O que cada passo pontuado do roteiro pede (passos sem nota ficam de fora)."""
    return {
        s.label: StepExpectation(s.gesture, s.expected, s.at_least, s.allow)
        for s in steps
        if s.expected is not None
    }


def detect_script(recording: Recording, handedness: str) -> str | None:
    """Roteiro cujos passos têm o mesmo nome e duração de todos os rótulos da gravação.

    Os nomes se repetem entre roteiros com contagens diferentes ("crouch" pede 1, 2 ou 3),
    por isso a duração desempata. O primeiro e o último rótulo podem ser mais curtos (as
    fixtures são recortes da sessão). ``None`` se nenhum roteiro casa.
    """
    labels = recording.labels
    for name, build in SCRIPTS.items():
        steps = {s.label: s for s in build(handedness)}

        def fits(i: int, steps: dict[str, GuidedStep] = steps) -> bool:
            lab = labels[i]
            step = steps.get(lab.gesture)
            if step is None:
                return False
            gap = step.duration_s - (lab.end - lab.start)
            edge = i in (0, len(labels) - 1)
            return abs(gap) <= DURATION_TOL_S or (edge and gap > 0)

        if labels and all(fits(i) for i in range(len(labels))):
            return name
    return None


def sidecar_handedness(path: Path) -> str | None:
    """Lateralidade da sessão pelo JSON do ``tools.guided`` ao lado do ``.npz``.

    O placar guarda o título de cada passo; o do soco dominante diz o braço ("BRACO
    ESQUERDO" = canhoto). ``None`` sem JSON ou sem esse passo.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return None
    steps = json.loads(sidecar.read_text(encoding="utf-8")).get("steps", [])
    title = next((s.get("title", "") for s in steps if s.get("label") == "punch_dominant"), "")
    if "ESQUERDO" in title:
        return "left"
    if "DIREITO" in title:
        return "right"
    return None


def fps_of(recording: Recording) -> float | None:
    """FPS da gravação: ``1 / mediana(Δt)``."""
    if recording.t.size < 2:
        return None
    return float(1.0 / np.median(np.diff(recording.t)))


def presses_by_gesture(
    inputs: Sequence[tuple[float, str, str]], profile: GameProfile
) -> list[tuple[float, str]]:
    """``(t, gesto)`` de cada press, pelo mapa inverso tecla → gesto do perfil."""
    by_key = {binding.input: gesture for gesture, binding in profile.bindings.items()}
    return [(t, by_key[key]) for t, kind, key in inputs if kind == "press" and key in by_key]


def evaluate_session(
    path: Path, config: AppConfig, profile: GameProfile, script: str | None = None
) -> tuple[Recording, SessionResult] | None:
    """Replay de uma gravação guiada e as métricas dela; ``None`` sem roteiro reconhecido."""
    recording = load_recording(path)
    handedness = sidecar_handedness(path)
    if handedness is not None and handedness != config.handedness:
        config = replace(config, handedness=handedness)  # type: ignore[arg-type]
    name = script or detect_script(recording, config.handedness)
    if name is None:
        logger.warning("%s: rótulos não casam com nenhum roteiro; pulada", path.name)
        return None
    exp = expectations(SCRIPTS[name](config.handedness))
    result = run_replay(recording, partial(build_pipeline, app_config=config, profile=profile))
    presses = presses_by_gesture(result.inputs, profile)
    return recording, SessionResult(
        name=path.stem,
        script=name,
        handedness=config.handedness,
        distance_m=recording.meta.distance_m,
        metrics=score_guided(recording.labels, exp, presses),
        confusion=confusion_counts(recording.labels, exp, presses),
        fps=fps_of(recording),
    )


def merge(results: Sequence[Mapping[str, GestureMetrics]]) -> dict[str, GestureMetrics]:
    """Soma TP, FP, FN, latências e trecho negativo de várias sessões, por gesto."""
    acc: dict[str, list[GestureMetrics]] = defaultdict(list)
    for metrics in results:
        for g, m in metrics.items():
            acc[g].append(m)
    return {
        g: GestureMetrics(
            g,
            sum(m.tp for m in ms),
            sum(m.fp for m in ms),
            sum(m.fn for m in ms),
            tuple(lat for m in ms for lat in m.latencies_s),
            sum(m.negative_s for m in ms),
        )
        for g, ms in sorted(acc.items())
    }


def filtered_series(
    recording: Recording, config: AppConfig, kind: str, landmark: int, axis: int
) -> NDArray[np.float64]:
    """Coordenada ``axis`` (pixels) de ``landmark`` em cada frame, depois do filtro ``kind``.

    O filtro é o mesmo do app (``pose_filter_from_config``) com só o ``kind`` trocado.
    """
    pose_filter = pose_filter_from_config(replace(config.filters, kind=kind))  # type: ignore[arg-type]
    size = recording.image_size[axis]
    out = np.full(recording.t.shape, np.nan)
    for i, (_, pose) in enumerate(recording.poses()):
        if pose_filter is not None:
            pose = pose_filter(pose)
        if pose is not None:
            out[i] = pose.landmarks[landmark, axis] * size
    return out


def _window(recording: Recording, label: str) -> NDArray[np.bool_] | None:
    lab = next((x for x in recording.labels if x.gesture == label), None)
    if lab is None:
        return None
    return np.asarray((recording.t >= lab.start) & (recording.t < lab.end), dtype=np.bool_)


def compare_filters(
    recording: Recording, name: str, config: AppConfig
) -> tuple[list[FilterResult], dict[str, NDArray[np.float64]]]:
    """Tremor do nariz parado (passo ``rest``) e atraso do punho dominante no soco.

    O tremor é a média de ``jitter`` em x e y do nariz (px). O atraso é ``lag_frames`` da
    altura do punho (y, px) filtrada contra a bruta, convertido em ms pela mediana de Δt.
    Devolve também as séries de y do punho (para o gráfico).
    """
    rest, punch = _window(recording, "rest"), _window(recording, "punch_dominant")
    wrist = _WRIST[config.handedness]
    dt = float(np.median(np.diff(recording.t))) if recording.t.size > 1 else math.nan
    raw_wrist = filtered_series(recording, config, "none", wrist, 1)
    results: list[FilterResult] = []
    series: dict[str, NDArray[np.float64]] = {}
    for kind in FILTER_KINDS:
        jit: float | None = None
        if rest is not None:
            values = [
                jitter(filtered_series(recording, config, kind, PoseLandmark.NOSE, axis)[rest])
                for axis in (0, 1)
            ]
            known = [v for v in values if v is not None]
            jit = sum(known) / len(known) if known else None
        lag: float | None = None
        if punch is not None:
            wrist_y = filtered_series(recording, config, kind, wrist, 1)
            series[kind] = wrist_y[punch]
            frames = lag_frames(raw_wrist[punch], wrist_y[punch], MAX_LAG_FRAMES)
            lag = None if frames is None else frames * dt * 1000.0
        results.append(FilterResult(name, kind, jit, lag))
    return results, series


def _pct(v: float | None) -> str:
    return "—" if v is None else f"{100 * v:.0f}%"


def _num(v: float | None, fmt: str = "{:.2f}") -> str:
    return "—" if v is None else fmt.format(v)


def _ok(ok: bool | None) -> str:
    return "" if ok is None else (" ✅" if ok else " ❌")


def metrics_rows(metrics: Mapping[str, GestureMetrics]) -> list[list[str]]:
    """Linhas da tabela: gesto, TP, FP, FN, precisão, recall, FP/min e latência mediana."""
    rows = []
    for g, m in metrics.items():
        lat = float(np.median(m.latencies_s)) if m.latencies_s else None
        rec_ok = None if m.recall is None else m.recall >= TARGET_RECALL
        fp_ok = None if m.fp_per_min is None else m.fp_per_min <= TARGET_FP_PER_MIN
        rows.append(
            [
                g,
                str(m.tp),
                str(m.fp),
                str(m.fn),
                _pct(m.precision),
                _pct(m.recall) + _ok(rec_ok),
                _num(m.fp_per_min) + _ok(fp_ok),
                _num(lat),
            ]
        )
    return rows


HEADER = ["gesto", "TP", "FP", "FN", "precisão", "recall", "FP/min", "1º disparo (s)"]


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def write_reports(
    out: Path, sessions: Sequence[SessionResult], filters: Sequence[FilterResult]
) -> None:
    """``metrics.md`` (tabelas), ``metrics.csv`` (por sessão e gesto) e ``filters.csv``."""
    out.mkdir(parents=True, exist_ok=True)
    total = merge([s.metrics for s in sessions])
    lines = [
        "# Métricas por gesto (sessões guiadas)",
        "",
        f"Metas da Fase 5: recall ≥ {TARGET_RECALL:.0%}, FP ≤ {TARGET_FP_PER_MIN:g}/min, "
        f'FPS ≥ {TARGET_FPS:g}. "1º disparo" é o tempo do comando na tela até a primeira '
        "tecla do passo (inclui a reação da pessoa).",
        "",
        "## Todas as sessões",
        "",
        *_md_table(HEADER, metrics_rows(total)),
    ]
    for dist in sorted({s.distance_m for s in sessions}, key=lambda d: (d is None, d)):
        group = [s for s in sessions if s.distance_m == dist]
        label = "distância desconhecida" if dist is None else f"{dist:g} m"
        lines += ["", f"## {label} ({len(group)} sessões)", ""]
        lines += _md_table(HEADER, metrics_rows(merge([s.metrics for s in group])))
    lines += ["", "## Sessões", ""]
    lines += _md_table(
        ["gravação", "roteiro", "mão", "distância", "FPS"],
        [
            [
                s.name,
                s.script,
                "canhoto" if s.handedness == "left" else "destro",
                _num(s.distance_m, "{:g} m"),
                _num(s.fps, "{:.1f}") + _ok(None if s.fps is None else s.fps >= TARGET_FPS),
            ]
            for s in sessions
        ],
    )
    if filters:
        lines += ["", "## Filtros (bruto × EMA × One Euro)", ""]
        lines += _md_table(
            ["gravação", "filtro", "tremor parado (px)", "atraso no soco (ms)"],
            [
                [f.recording, FILTER_NAMES[f.kind], _num(f.jitter_px), _num(f.lag_ms, "{:.0f}")]
                for f in filters
            ],
        )
    (out / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["session", "distance_m", "gesture", "tp", "fp", "fn", "negative_s"])
        for s in sessions:
            for g, m in s.metrics.items():
                writer.writerow([s.name, s.distance_m, g, m.tp, m.fp, m.fn, f"{m.negative_s:.3f}"])
    with (out / "filters.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["recording", "filter", "jitter_px", "lag_ms"])
        for f in filters:
            writer.writerow([f.recording, f.kind, f.jitter_px, f.lag_ms])


def write_plots(
    out: Path,
    sessions: Sequence[SessionResult],
    filter_series: Mapping[str, Mapping[str, NDArray[np.float64]]],
) -> list[Path]:
    """Matriz de confusão e bruto × EMA × One Euro; nada se o ``matplotlib`` faltar."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib não instalado: gráficos não gerados")
        return []
    written: list[Path] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for s in sessions:
        for key, n in s.confusion.items():
            counts[key] += n
    if counts:
        rows = sorted({r for r, _ in counts}, key=lambda r: (r == NO_GESTURE, r))
        cols = sorted({c for _, c in counts})
        grid = np.array([[counts.get((r, c), 0) for c in cols] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(1.1 * len(cols) + 2.5, 0.6 * len(rows) + 2))
        ax.imshow(grid, cmap="Blues")
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("gesto disparado")
        ax.set_ylabel("gesto pedido no passo")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if grid[i, j]:
                    ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center")
        ax.set_title("Disparos por passo (todas as sessões)")
        fig.tight_layout()
        path = out / "confusion.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    for name, series in filter_series.items():
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.5))
        for kind in FILTER_KINDS:
            if kind in series:
                ax.plot(series[kind], label=FILTER_NAMES[kind], linewidth=1.2)
        ax.invert_yaxis()  # y da imagem cresce para baixo: o soco sobe na tela
        ax.set_xlabel("frame no passo de socos")
        ax.set_ylabel("altura do punho (px)")
        ax.set_title(f"{name}: bruto × EMA × One Euro")
        ax.legend()
        fig.tight_layout()
        path = out / f"filters_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Métricas por gesto sobre sessões guiadas.")
    parser.add_argument(
        "--recordings", nargs="+", default=[DEFAULT_RECORDINGS], help="arquivos .npz (glob)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="pasta de saída")
    parser.add_argument("--script", choices=sorted(SCRIPTS), help="força o roteiro")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML base")
    parser.add_argument("--game", default="minecraft", help="perfil em config/games/")
    parser.add_argument("--profile", type=Path, help="YAML do usuário, aplicado após o jogo")
    parser.add_argument(
        "--set", action="append", default=[], dest="overrides", help="CAMINHO=VALOR"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = sorted({Path(p) for pattern in args.recordings for p in glob.glob(pattern)})
    if not paths:
        logger.error("nenhuma gravação encontrada em %s", " ".join(args.recordings))
        return 1
    try:
        config, profile = resolve_settings(args)
        sessions: list[SessionResult] = []
        filters: list[FilterResult] = []
        series: dict[str, dict[str, NDArray[np.float64]]] = {}
        for path in paths:
            done = evaluate_session(path, config, profile, args.script)
            if done is None:
                continue
            recording, session = done
            sessions.append(session)
            session_config = replace(config, handedness=session.handedness)  # type: ignore[arg-type]
            found, wrist = compare_filters(recording, session.name, session_config)
            if any(f.jitter_px is not None or f.lag_ms is not None for f in found):
                filters += found
                series[session.name] = wrist
            logger.info("%s: roteiro %s", path.name, session.script)
    except (OSError, ValueError) as exc:
        logger.error("falha na avaliação: %s", exc)
        return 1
    if not sessions:
        logger.error("nenhuma sessão guiada reconhecida")
        return 1
    write_reports(args.out, sessions, filters)
    for path in write_plots(args.out, sessions, series):
        logger.info("gráfico: %s", path)
    logger.info("relatório: %s", args.out / "metrics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
