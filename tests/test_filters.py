"""Testes dos filtros temporais EMA (R1-R3) e One Euro (R4) — spec 006."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.filters import TemporalFilter
from inmotion.filters.ema import Ema, EmaParams
from inmotion.filters.one_euro import OneEuro, OneEuroParams

DT = 1.0 / 30.0
_values = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_alphas = st.floats(min_value=0.01, max_value=1.0)

FilterFactory = Callable[[], TemporalFilter]
FACTORIES: list[FilterFactory] = [
    lambda: Ema(EmaParams()),
    lambda: OneEuro(OneEuroParams()),
]
FACTORY_IDS = ["ema", "one_euro"]


def closed_form_ema(xs: list[float], alpha: float) -> float:
    """Oráculo independente: R2 com s_n = α·Σ_{k=0}^{n-1}(1−α)^k·x_{n−k} + (1−α)^n·x₀."""
    n = len(xs) - 1
    total = sum((1.0 - alpha) ** k * xs[n - k] for k in range(n))
    return alpha * total + (1.0 - alpha) ** n * xs[0]


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="stub da T1")
@given(xs=st.lists(_values, min_size=1, max_size=60), alpha=_alphas)
def test_ema_matches_closed_form(xs: list[float], alpha: float) -> None:
    """CA1: a recursão R1 coincide com a forma fechada R2 em todo n."""
    ema = Ema(EmaParams(alpha=alpha))
    for n, x in enumerate(xs):
        s = ema.update(x, n * DT)
        assert float(s) == pytest.approx(closed_form_ema(xs[: n + 1], alpha), abs=1e-9)


@given(alpha=_alphas, n=st.integers(min_value=1, max_value=500))
def test_ema_weights_sum_to_one(alpha: float, n: int) -> None:
    """CA2 (R3): α·Σ(1−α)^k + (1−α)^n = 1."""
    total = alpha * sum((1.0 - alpha) ** k for k in range(n)) + (1.0 - alpha) ** n
    assert total == pytest.approx(1.0, abs=1e-9)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="stub da T1")
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
@given(xs=st.lists(_values, min_size=1, max_size=60))
def test_output_within_input_range(factory: FilterFactory, xs: list[float]) -> None:
    """CA3: a saída fica no fecho convexo [min, max] das entradas."""
    f = factory()
    for n, x in enumerate(xs):
        s = float(f.update(x, n * DT))
        assert min(xs[: n + 1]) - 1e-9 <= s <= max(xs[: n + 1]) + 1e-9


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="stub da T1")
@given(
    xs=st.lists(_values, min_size=2, max_size=40),
    min_cutoff=st.floats(min_value=0.1, max_value=10.0),
    dt=st.floats(min_value=0.005, max_value=0.2),
)
def test_one_euro_beta_zero_equals_ema(xs: list[float], min_cutoff: float, dt: float) -> None:
    """F2: com β = 0 e Δt constante, One Euro = EMA com α = 1/(1 + 1/(2π·f_min·Δt))."""
    alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * min_cutoff * dt))
    euro = OneEuro(OneEuroParams(min_cutoff=min_cutoff, beta=0.0, gap_max_s=10.0))
    ema = Ema(EmaParams(alpha=alpha, gap_max_s=10.0))
    for n, x in enumerate(xs):
        a = euro.update(x, n * dt)
        b = ema.update(x, n * dt)
        assert float(a) == pytest.approx(float(b), abs=1e-9)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="stub da T1")
@pytest.mark.parametrize("factory", FACTORIES, ids=FACTORY_IDS)
@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_non_positive_dt_raises(factory: FilterFactory, bad_dt: float) -> None:
    """CA5: Δt ≤ 0 (tempo repetido ou regressivo) levanta ValueError."""
    f = factory()
    f.update(1.0, 1.0)
    with pytest.raises(ValueError, match="dt"):
        f.update(2.0, 1.0 + bad_dt)
