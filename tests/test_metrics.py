"""Testes das métricas por gesto (spec 011 T6; CA3)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.gestures.base import EventKind, GestureEvent
from inmotion.metrics import GestureMetrics, evaluate, match_events
from inmotion.recording import LabelInterval
from tests.test_recording import make_recording


def pulse(gesture: str, t: float) -> GestureEvent:
    return GestureEvent(gesture, EventKind.PULSE, t)


def test_known_counts_and_latency() -> None:
    labels = [
        LabelInterval("attack", 1.0, 2.0),
        LabelInterval("attack", 5.0, 6.0),
        LabelInterval("attack", 9.0, 10.0),  # sem evento: FN
        LabelInterval("crouch", 3.0, 4.0),
    ]
    events = [
        pulse("attack", 1.2),  # TP, latência 0,2
        pulse("attack", 3.0),  # FP
        pulse("attack", 4.84),  # fora de [4,85; 6,15]: FP
        pulse("attack", 5.05),  # TP, latência 0,05
        GestureEvent("crouch", EventKind.START, 2.9),  # TP em [2,85; 4,15], latência -0,1
        GestureEvent("crouch", EventKind.END, 4.0),  # ignorado
        GestureEvent("crouch", EventKind.START, 8.0),  # FP
        GestureEvent("lean", EventKind.VALUE, 1.0, value=0.3),  # ignorado
        GestureEvent("jump", EventKind.PULSE, 7.0),  # sem rótulos: FP
    ]
    out = match_events(labels, events, duration_s=20.0)

    assert set(out) == {"attack", "crouch", "jump"}
    attack = out["attack"]
    assert (attack.tp, attack.fp, attack.fn) == (2, 2, 1)
    assert attack.latencies_s == pytest.approx((0.2, 0.05))
    assert attack.negative_s == pytest.approx(17.0)
    assert attack.precision == pytest.approx(0.5)
    assert attack.recall == pytest.approx(2 / 3)
    assert attack.fp_per_min == pytest.approx(2 / (17.0 / 60.0))

    crouch = out["crouch"]
    assert (crouch.tp, crouch.fp, crouch.fn) == (1, 1, 0)
    assert crouch.latencies_s == pytest.approx((-0.1,))
    assert crouch.negative_s == pytest.approx(19.0)

    jump = out["jump"]
    assert (jump.tp, jump.fp, jump.fn) == (0, 1, 0)
    assert jump.negative_s == pytest.approx(20.0)


def test_matching_is_one_to_one_and_respects_tolerance() -> None:
    labels = [LabelInterval("a", 1.0, 2.0)]
    # Dois eventos na janela: só o primeiro casa, o segundo é FP.
    out = match_events(labels, [pulse("a", 1.1), pulse("a", 1.5)], 10.0)["a"]
    assert (out.tp, out.fp, out.fn) == (1, 1, 0)
    # Borda inclusiva: a − δ e b + δ.
    for t in (0.85, 2.15):
        edge = match_events(labels, [pulse("a", t)], 10.0, tolerance_s=0.15)["a"]
        assert edge.tp == 1
    miss = match_events(labels, [pulse("a", 2.16)], 10.0, tolerance_s=0.15)["a"]
    assert (miss.tp, miss.fp, miss.fn) == (0, 1, 1)


def test_evaluate_uses_recording_duration() -> None:
    rec = make_recording(31, labels=(LabelInterval("a", 0.2, 0.4),))  # 30 passos de 1/30 s
    out = evaluate(rec, [pulse("a", 0.3)])
    assert out["a"].tp == 1
    assert out["a"].negative_s == pytest.approx(1.0 - 0.2)


def test_undefined_metrics_are_none() -> None:
    nada = GestureMetrics("x", tp=0, fp=0, fn=0, latencies_s=(), negative_s=0.0)
    assert nada.precision is None
    assert nada.recall is None
    assert nada.fp_per_min is None
    so_fp = GestureMetrics("x", tp=0, fp=2, fn=0, latencies_s=(), negative_s=0.0)
    assert so_fp.precision == 0.0
    assert so_fp.recall is None
    assert so_fp.fp_per_min is None  # sem trecho negativo
    so_fn = GestureMetrics("x", tp=0, fp=0, fn=3, latencies_s=(), negative_s=30.0)
    assert so_fn.precision is None
    assert so_fn.recall == 0.0
    assert so_fn.fp_per_min == 0.0
    # Rótulos cobrindo toda a gravação: negativo = 0.
    cheio = match_events([LabelInterval("a", 0.0, 5.0)], [], duration_s=5.0)["a"]
    assert cheio.negative_s == 0.0
    assert cheio.fp_per_min is None


@given(
    spans=st.lists(
        st.tuples(st.integers(1, 30), st.integers(1, 30)), max_size=8
    ),  # (folga, duração) em décimos de segundo
    event_times=st.lists(st.floats(0.0, 120.0), max_size=25),
    tolerance=st.floats(0.0, 1.0),
)
def test_tp_plus_fn_equals_labels(
    spans: list[tuple[int, int]], event_times: list[float], tolerance: float
) -> None:
    labels: list[LabelInterval] = []
    cursor = 0.0
    for gap, length in spans:
        start = cursor + gap / 10.0
        labels.append(LabelInterval("g", start, start + length / 10.0))
        cursor = start + length / 10.0
    events = [pulse("g", t) for t in event_times]
    duration = 130.0

    out = match_events(labels, events, duration, tolerance)
    if not labels and not events:
        assert out == {}
        return
    m = out["g"]
    assert m.tp + m.fn == len(labels)
    assert m.tp + m.fp == len(events)
    assert len(m.latencies_s) == m.tp
    assert 0.0 <= m.negative_s <= duration
