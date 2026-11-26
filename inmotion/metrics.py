"""Métricas por gesto: pareamento evento ↔ rótulo, precisão, recall, FP/min, latência.

Núcleo puro (spec 011, "Modelo matemático"). Para cada gesto g:

- Um evento ``e`` de disparo casa com o rótulo ``[a, b]`` se ``t_e ∈ [a − δ, b + δ]``.
  Contam como disparo só ``START`` (gestos HOLD) e ``PULSE`` (gestos TRIGGER); ``END`` e
  ``VALUE`` são ignorados (o fim de um HOLD e a saída analógica não são "detecções").
- Pareamento guloso e um-para-um: os rótulos, em ordem de início, tomam o evento livre
  mais cedo dentro da janela. Como os rótulos de um mesmo gesto não se sobrepõem, o
  resultado é ótimo (plano 011).
- ``TP`` = pares; ``FN`` = rótulos sem evento; ``FP`` = eventos sem rótulo.
- ``precisão = TP / (TP + FP)``; ``recall = TP / (TP + FN)``.
- ``FP/min = FP / (D⁻ / 60)``, com ``D⁻ = D − |união dos rótulos|`` (trecho negativo, s).
- ``latência = t_e − a`` (pode ser levemente negativa, dentro da tolerância).

Métrica de denominador zero é ``None`` (indefinida), nunca 0 nem NaN.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from inmotion.gestures.base import EventKind, GestureEvent
from inmotion.recording import LabelInterval, Recording

# Folga numérica nas bordas da janela [a − δ, b + δ] (erro de ponto flutuante em a ± δ).
_EDGE_SLACK: Final = 1e-9

# Tipos de evento que contam como detecção de um gesto.
_TRIGGER_KINDS: Final = frozenset({EventKind.START, EventKind.PULSE})


@dataclass(frozen=True)
class GestureMetrics:
    """Contagens e latências de um gesto; precisão, recall e FP/min derivam delas."""

    gesture: str
    tp: int
    fp: int
    fn: int
    latencies_s: tuple[float, ...]
    negative_s: float

    def __post_init__(self) -> None:
        if min(self.tp, self.fp, self.fn) < 0:
            raise ValueError("tp, fp e fn devem ser >= 0")
        if not (math.isfinite(self.negative_s) and self.negative_s >= 0.0):
            raise ValueError("negative_s deve ser finito e >= 0")

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP); ``None`` se não houve nenhuma detecção."""
        total = self.tp + self.fp
        return None if total == 0 else self.tp / total

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN); ``None`` se não há rótulos."""
        total = self.tp + self.fn
        return None if total == 0 else self.tp / total

    @property
    def fp_per_min(self) -> float | None:
        """FP / (D⁻ / 60); ``None`` se não há trecho negativo."""
        if self.negative_s <= 0.0:
            return None
        return self.fp / (self.negative_s / 60.0)


def _covered_length(labels: Sequence[LabelInterval]) -> float:
    """Comprimento da união dos intervalos (rótulos podem tocar-se ou se sobrepor)."""
    total = 0.0
    end = -math.inf
    for lab in sorted(labels, key=lambda x: x.start):
        start = max(lab.start, end)
        if lab.end > start:
            total += lab.end - start
        end = max(end, lab.end)
    return total


def _match_one(
    gesture: str,
    labels: Sequence[LabelInterval],
    times: Sequence[float],
    duration_s: float,
    tolerance_s: float,
) -> GestureMetrics:
    used = [False] * len(times)
    latencies: list[float] = []
    for lab in sorted(labels, key=lambda x: x.start):
        lo = lab.start - tolerance_s - _EDGE_SLACK
        hi = lab.end + tolerance_s + _EDGE_SLACK
        i = bisect_left(times, lo)
        while i < len(times) and times[i] <= hi:
            if not used[i]:
                used[i] = True
                latencies.append(times[i] - lab.start)
                break
            i += 1
    tp = len(latencies)
    negative = max(0.0, duration_s - _covered_length(labels))
    return GestureMetrics(
        gesture=gesture,
        tp=tp,
        fp=len(times) - tp,
        fn=len(labels) - tp,
        latencies_s=tuple(latencies),
        negative_s=negative,
    )


def match_events(
    labels: Sequence[LabelInterval],
    events: Sequence[GestureEvent],
    duration_s: float,
    tolerance_s: float = 0.15,
) -> dict[str, GestureMetrics]:
    """Pareia eventos e rótulos por gesto e devolve as métricas (ordenadas por nome).

    ``duration_s`` é a duração D da gravação; ``tolerance_s`` é o δ (padrão 0,15 s).
    Aparecem no resultado os gestos que têm rótulo ou evento de disparo.
    """
    if not (math.isfinite(duration_s) and duration_s >= 0.0):
        raise ValueError("duration_s deve ser finito e >= 0")
    if not (math.isfinite(tolerance_s) and tolerance_s >= 0.0):
        raise ValueError("tolerance_s deve ser finito e >= 0")
    by_labels: dict[str, list[LabelInterval]] = defaultdict(list)
    for lab in labels:
        by_labels[lab.gesture].append(lab)
    by_times: dict[str, list[float]] = defaultdict(list)
    for ev in events:
        if ev.kind in _TRIGGER_KINDS:
            by_times[ev.gesture].append(ev.t)
    return {
        name: _match_one(name, by_labels[name], sorted(by_times[name]), duration_s, tolerance_s)
        for name in sorted(set(by_labels) | set(by_times))
    }


def evaluate(
    recording: Recording, events: Sequence[GestureEvent], tolerance_s: float = 0.15
) -> dict[str, GestureMetrics]:
    """Métricas dos ``events`` contra os rótulos e a duração da ``recording``."""
    return match_events(recording.labels, events, recording.duration_s, tolerance_s)


# ---------------------------------------------------------------------------
# Sessões guiadas (spec 011 T7, revisão de 18/11/2026)
#
# Os dados rotulados reais são sessões do ``tools.guided``: cada rótulo é um passo do
# roteiro ("3 socos", "fique parado") com várias repetições dentro. O pareamento um
# evento ↔ um rótulo acima não serve; aqui a unidade é a repetição pedida no passo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepExpectation:
    """O que um passo pede: ``expected`` disparos de ``gesture`` (``None`` = parado, 0).

    ``at_least``: o passo conta como uma unidade, acertada com ``≥ 1`` disparo (marcha:
    uma pausa curta gera um segundo ``w``). ``allow``: gestos que podem disparar no passo
    sem contar como falso positivo (correr permite andar).
    """

    gesture: str | None
    expected: int
    at_least: bool = False
    allow: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected < 0:
            raise ValueError("expected deve ser >= 0")
        if self.gesture is None and self.expected != 0:
            raise ValueError("passo parado (gesture=None) deve ter expected = 0")


def score_guided(
    labels: Sequence[LabelInterval],
    expectations: Mapping[str, StepExpectation],
    presses: Sequence[tuple[float, str]],
    grace_s: float = 0.6,
) -> dict[str, GestureMetrics]:
    """Métricas por gesto numa sessão guiada, contando repetições.

    ``presses`` são ``(t, gesto)`` dos disparos; ``labels`` são as janelas ``[a, b]`` dos
    passos, cujo ``gesture`` é o nome do passo em ``expectations`` (passos desconhecidos
    são ignorados). Para cada gesto ``g`` e passo positivo ``s`` (que pede ``g``), com
    ``n_s`` disparos de ``g`` em ``[a, b + grace_s)``:

    - passo comum: ``TP += min(n_s, E_s)``, ``FN += E_s − min(n_s, E_s)``,
      ``FP += max(0, n_s − E_s)``;
    - passo ``at_least``: ``TP += 1`` se ``n_s ≥ 1``, senão ``FN += 1``;
    - disparo de ``g`` dentro de um passo negativo (que não pede nem permite ``g``):
      ``FP += 1``; o trecho negativo ``D⁻_g`` soma a duração desses passos;
    - latência: ``t₁ − a``, o primeiro disparo de ``g`` no passo (inclui a reação da
      pessoa ao comando na tela).

    Disparos fora de qualquer passo (posicionar-se, preparação) não contam.
    """
    if not (math.isfinite(grace_s) and grace_s >= 0.0):
        raise ValueError("grace_s deve ser finito e >= 0")
    steps = [(lab, expectations[lab.gesture]) for lab in labels if lab.gesture in expectations]
    by_gesture: dict[str, list[float]] = defaultdict(list)
    for t, gesture in presses:
        by_gesture[gesture].append(t)
    names = {exp.gesture for _, exp in steps if exp.gesture is not None} | set(by_gesture)
    out: dict[str, GestureMetrics] = {}
    for g in sorted(names):
        times = sorted(by_gesture[g])
        tp = fp = fn = 0
        negative = 0.0
        latencies: list[float] = []
        used: set[int] = set()  # disparo já contado num passo positivo (inclui a folga)
        for lab, exp in steps:
            if exp.gesture != g:
                continue
            inside = [i for i, t in enumerate(times) if lab.start <= t < lab.end + grace_s]
            used.update(inside)
            if inside:
                latencies.append(times[inside[0]] - lab.start)
            if exp.at_least:
                tp, fn = (tp + 1, fn) if inside else (tp, fn + 1)
            else:
                hit = min(len(inside), exp.expected)
                tp, fn, fp = tp + hit, fn + exp.expected - hit, fp + len(inside) - hit
        for lab, exp in steps:
            if exp.gesture == g or g in exp.allow:
                continue
            negative += lab.end - lab.start
            fp += sum(1 for i, t in enumerate(times) if i not in used and lab.start <= t < lab.end)
        out[g] = GestureMetrics(g, tp, fp, fn, tuple(latencies), negative)
    return out


NO_GESTURE: Final = "parado"


def confusion_counts(
    labels: Sequence[LabelInterval],
    expectations: Mapping[str, StepExpectation],
    presses: Sequence[tuple[float, str]],
) -> dict[tuple[str, str], int]:
    """Matriz de confusão: ``(gesto pedido no passo, gesto disparado) → nº de disparos``.

    O passo parado entra como ``"parado"``. Só contam disparos dentro da janela do passo.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for lab in labels:
        exp = expectations.get(lab.gesture)
        if exp is None:
            continue
        row = exp.gesture if exp.gesture is not None else NO_GESTURE
        for t, gesture in presses:
            if lab.start <= t < lab.end:
                counts[(row, gesture)] += 1
    return dict(counts)


def jitter(values: Sequence[float]) -> float | None:
    """Tremor de um sinal parado: ``J = std(Δx)``, ``Δx_t = x_t − x_{t−1}`` (ignora NaN).

    ``None`` com menos de 3 amostras finitas consecutivas.
    """
    x = np.asarray(values, dtype=np.float64)
    dx = np.diff(x)
    dx = dx[np.isfinite(dx)]
    if dx.size < 2:
        return None
    return float(np.std(dx))


def lag_frames(raw: Sequence[float], filtered: Sequence[float], max_lag: int) -> int | None:
    """Atraso do filtro: ``k* = argmax_k Σ_t r̃_t · f̃_{t+k}``, ``k ∈ [0, max_lag]``.

    ``r̃`` e ``f̃`` são os sinais sem a média (NaN vira 0 depois de tirar a média).
    ``None`` se algum dos sinais for constante ou curto demais.
    """
    if max_lag < 0:
        raise ValueError("max_lag deve ser >= 0")
    r = np.asarray(raw, dtype=np.float64)
    f = np.asarray(filtered, dtype=np.float64)
    if r.shape != f.shape or r.size <= max_lag + 1:
        return None
    r = np.nan_to_num(r - np.nanmean(r))
    f = np.nan_to_num(f - np.nanmean(f))
    if not r.any() or not f.any():
        return None
    n = r.size
    scores = [float(np.dot(r[: n - k], f[k:])) for k in range(max_lag + 1)]
    return int(np.argmax(scores))
