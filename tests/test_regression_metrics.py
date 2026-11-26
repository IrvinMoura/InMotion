"""Spec 011 T9 — metas da Fase 5 sobre sessões reais recortadas (CA4, CA5).

Fixtures (pessoa p1, 1,5 m e 2 m), todas recortes de ``recordings/guided_p1_*``:

- ``punch_*``: parado, 3 socos com cada braço (roteiro ``full``, -032516 e -032334; com
  world, que o alcance frontal do soco usa);
- ``crouch_*``: parado, 3 pulos, 3 agachamentos, 3 inclinações sem tecla;
- ``march_*``: parado, marcha lenta, marcha rápida, pausas, 3 pulos.

As ``punch_*`` vêm de sessões que já davam 3/3: estes testes detectam piora, não provam a
meta (a medida de qualidade é o ``tools.evaluate`` sobre todas as gravações).

Os passos "parado" (e inclinar, que não aperta nada) são o trecho negativo do FP/min.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from inmotion.config.loader import load_config
from inmotion.metrics import GestureMetrics
from tools.evaluate import evaluate_session, merge

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"
METRIC_FIXTURES = (
    "punch_1m5.npz",
    "punch_2m.npz",
    "crouch_1m5.npz",
    "crouch_2m.npz",
    "march_1m5.npz",
    "march_2m.npz",
)
TARGET_RECALL = 0.90  # Fase 5 (ROADMAP)
TARGET_FP_PER_MIN = 1.0
GESTURES = ("attack", "use", "jump", "crouch", "walk", "sprint")
# Spec 019 (inclinação do tronco): walk_back dispara ao inclinar para a frente, nos pulos e
# nos socos (medido com o tools.evaluate em 18/11/2026). Fica à parte até a correção.
KNOWN_FP = frozenset({"walk_back"})


@cache
def fixture_metrics() -> dict[str, GestureMetrics]:
    config, profile = load_config(game="minecraft")
    assert profile is not None
    sessions = []
    for name in METRIC_FIXTURES:
        done = evaluate_session(FIXTURES / name, config, profile)
        assert done is not None, f"{name}: roteiro não reconhecido"
        sessions.append(done[1].metrics)
    return merge(sessions)


@pytest.mark.slow
@pytest.mark.parametrize("gesture", GESTURES)
def test_gesture_fixtures_recall(gesture: str) -> None:
    """CA5: recall ≥ 90% em cada gesto pedido nas fixtures."""
    m = fixture_metrics()[gesture]
    assert m.recall is not None
    assert m.recall >= TARGET_RECALL, f"{gesture}: TP {m.tp}, FN {m.fn}"


@pytest.mark.slow
def test_still_fixture_false_positives() -> None:
    """CA4: ≤ 1 disparo por minuto nos passos que não pedem o gesto."""
    over = {
        g: m.fp_per_min
        for g, m in fixture_metrics().items()
        if g not in KNOWN_FP and m.fp_per_min is not None and m.fp_per_min > TARGET_FP_PER_MIN
    }
    assert not over


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="spec 019: walk_back dispara sem ser pedido")
def test_walk_back_false_positives_known_issue() -> None:
    """Quando a 019 corrigir o walk_back, este teste passa e o xfail estrito avisa."""
    m = fixture_metrics()["walk_back"]
    assert m.fp_per_min is not None
    assert m.fp_per_min <= TARGET_FP_PER_MIN


def test_fixtures_under_1mb() -> None:
    """RF7 (ADR 0008): cada fixture versionada tem até 1 MB."""
    big = {p.name: p.stat().st_size for p in FIXTURES.glob("*.npz") if p.stat().st_size > 2**20}
    assert not big
