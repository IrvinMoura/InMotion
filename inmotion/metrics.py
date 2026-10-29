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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

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
