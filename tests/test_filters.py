"""Testes dos filtros temporais EMA (R1-R3) e One Euro (R4) — spec 006."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.filters import TemporalFilter
from inmotion.filters.ema import Ema, EmaParams
from inmotion.filters.one_euro import OneEuro, OneEuroParams, smoothing_factor

DT = 1.0 / 30.0
_values = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_alphas = st.floats(min_value=0.01, max_value=1.0)

FilterFactory = Callable[[], TemporalFilter]
XFAIL_NO_RAISE = pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason="T4")
FACTORIES: list[FilterFactory] = [
    lambda: Ema(EmaParams()),
    lambda: OneEuro(OneEuroParams()),
]
FACTORY_IDS = ["ema", "one_euro"]
# A EMA ainda não valida Δt (T4).
RANGE_PARAMS = [pytest.param(f, id=i) for f, i in zip(FACTORIES, FACTORY_IDS)]
DT_PARAMS = [
    pytest.param(FACTORIES[0], id="ema", marks=XFAIL_NO_RAISE),
    pytest.param(FACTORIES[1], id="one_euro"),
]


def closed_form_ema(xs: list[float], alpha: float) -> float:
    """Oráculo independente: R2 com s_n = α·Σ_{k=0}^{n-1}(1−α)^k·x_{n−k} + (1−α)^n·x₀."""
    n = len(xs) - 1
    total = sum((1.0 - alpha) ** k * xs[n - k] for k in range(n))
    return alpha * total + (1.0 - alpha) ** n * xs[0]


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


@pytest.mark.parametrize("factory", RANGE_PARAMS)
@given(xs=st.lists(_values, min_size=1, max_size=60))
def test_output_within_input_range(factory: FilterFactory, xs: list[float]) -> None:
    """CA3: a saída fica no fecho convexo [min, max] das entradas."""
    f = factory()
    for n, x in enumerate(xs):
        s = float(f.update(x, n * DT))
        assert min(xs[: n + 1]) - 1e-9 <= s <= max(xs[: n + 1]) + 1e-9


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


@pytest.mark.parametrize("factory", DT_PARAMS)
@pytest.mark.parametrize("bad_dt", [0.0, -0.01])
def test_non_positive_dt_raises(factory: FilterFactory, bad_dt: float) -> None:
    """CA5: Δt ≤ 0 (tempo repetido ou regressivo) levanta ValueError."""
    f = factory()
    f.update(1.0, 1.0)
    with pytest.raises(ValueError, match="dt"):
        f.update(2.0, 1.0 + bad_dt)


@given(xs=st.lists(_values, min_size=1, max_size=40))
def test_ema_alpha_one_is_identity(xs: list[float]) -> None:
    """F4: com α = 1 a saída é igual à entrada."""
    ema = Ema(EmaParams(alpha=1.0))
    for n, x in enumerate(xs):
        assert float(ema.update(x, n * DT)) == x


@given(c=_values, alpha=_alphas, n=st.integers(min_value=1, max_value=50))
def test_constant_input_constant_output(c: float, alpha: float, n: int) -> None:
    """F3: entrada constante gera saída constante e igual à entrada."""
    ema = Ema(EmaParams(alpha=alpha))
    for k in range(n):
        assert float(ema.update(c, k * DT)) == pytest.approx(c, abs=1e-9)


@given(
    lo=_values,
    height=st.floats(min_value=0.1, max_value=100.0),
    alpha=_alphas,
    up=st.booleans(),
)
def test_ema_step_response_monotone(lo: float, height: float, alpha: float, up: bool) -> None:
    """F5: a resposta a um degrau é monótona e não ultrapassa o patamar (sem overshoot)."""
    a, b = (lo, lo + height) if up else (lo + height, lo)
    ema = Ema(EmaParams(alpha=alpha))
    ema.update(a, 0.0)
    prev = a
    sign = 1.0 if up else -1.0
    for k in range(1, 40):
        s = float(ema.update(b, k * DT))
        assert sign * (s - prev) >= -1e-12
        assert sign * (s - b) <= 1e-9
        prev = s


def test_ema_params_validation() -> None:
    """Parâmetros fora de 0 < α ≤ 1 e gap_max_s > 0 são rejeitados."""
    for bad in (0.0, -0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="alpha"):
            EmaParams(alpha=bad)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="gap_max_s"):
            EmaParams(gap_max_s=bad)


@given(
    cutoff=st.floats(min_value=1e-3, max_value=1e3),
    dt=st.floats(min_value=1e-4, max_value=10.0),
)
def test_smoothing_factor_in_open_unit_interval(cutoff: float, dt: float) -> None:
    """F1: α = 1/(1 + τ/Δt), τ = 1/(2π·f_c), está em (0, 1) para Δt > 0, f_c > 0."""
    alpha = smoothing_factor(cutoff, dt)
    assert isinstance(alpha, float)
    assert 0.0 < alpha < 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff)
    assert alpha == pytest.approx(1.0 / (1.0 + tau / dt), rel=1e-12)


def test_smoothing_factor_accepts_arrays_and_rejects_invalid() -> None:
    """Aceita array de cortes (devolve array); Δt ≤ 0 ou f_c ≤ 0 levanta ValueError."""
    out = smoothing_factor(np.array([0.5, 1.0, 2.0]), DT)
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)
    assert np.all(np.diff(out) > 0)  # corte maior -> α maior
    with pytest.raises(ValueError, match="dt"):
        smoothing_factor(1.0, 0.0)
    with pytest.raises(ValueError, match="cutoff"):
        smoothing_factor(0.0, DT)


def test_one_euro_adapts_cutoff_to_speed() -> None:
    """R4: com β > 0, um degrau rápido faz o filtro seguir mais depressa que com β = 0."""
    slow = OneEuro(OneEuroParams(beta=0.0))
    fast = OneEuro(OneEuroParams(beta=5.0))
    for f in (slow, fast):
        f.update(0.0, 0.0)
    a = b = 0.0
    for k in range(1, 6):
        a = float(slow.update(10.0, k * DT))
        b = float(fast.update(10.0, k * DT))
    assert b > a


def test_one_euro_params_validation() -> None:
    """min_cutoff > 0, beta ≥ 0, d_cutoff > 0, gap_max_s > 0."""
    with pytest.raises(ValueError, match="min_cutoff"):
        OneEuroParams(min_cutoff=0.0)
    with pytest.raises(ValueError, match="beta"):
        OneEuroParams(beta=-0.1)
    with pytest.raises(ValueError, match="d_cutoff"):
        OneEuroParams(d_cutoff=0.0)
    with pytest.raises(ValueError, match="gap_max_s"):
        OneEuroParams(gap_max_s=0.0)
