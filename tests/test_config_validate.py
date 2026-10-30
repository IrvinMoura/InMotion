"""Spec 004 — construção, validação e mesclagem da configuração (núcleo puro)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.config.schema import AppConfig
from inmotion.config.validate import ConfigError, build, deep_merge
from inmotion.filters.ema import EmaParams


@dataclass(frozen=True)
class Leaf:
    alpha: float = 0.5
    name: str = "x"


@dataclass(frozen=True)
class Inner:
    leaf: Leaf = field(default_factory=Leaf)
    count: int = 3


@dataclass(frozen=True)
class Outer:
    inner: Inner
    flag: bool = False


@dataclass(frozen=True)
class WithEma:
    ema: EmaParams = field(default_factory=EmaParams)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_build_nested_dataclass() -> None:
    cfg = build(Outer, {"inner": {"leaf": {"alpha": 0.25}, "count": 7}, "flag": True})
    assert cfg == Outer(Inner(Leaf(alpha=0.25), count=7), flag=True)
    assert build(Outer, {"inner": {}}) == Outer(Inner())  # padrões preenchidos


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_missing_field_has_path() -> None:
    with pytest.raises(ConfigError) as exc:
        build(Outer, {})
    assert exc.value.path == "inner"
    assert "inner" in str(exc.value)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_unknown_key_suggests_name() -> None:
    with pytest.raises(ConfigError) as exc:
        build(Outer, {"inner": {"leaf": {"alpah": 0.3}}})
    assert exc.value.path == "inner.leaf.alpah"
    assert "alpha" in str(exc.value)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_post_init_error_becomes_config_error_with_path() -> None:
    with pytest.raises(ConfigError) as exc:
        build(WithEma, {"ema": {"alpha": 0.0}}, "filters")
    assert exc.value.path == "filters.ema.alpha"
    assert "0.0" in str(exc.value)


_leaf_values = st.one_of(st.integers(), st.text(max_size=3), st.booleans(), st.none())
_layers = st.recursive(
    st.dictionaries(st.sampled_from("abc"), _leaf_values, max_size=3),
    lambda inner: st.dictionaries(
        st.sampled_from("abc"), st.one_of(_leaf_values, inner), max_size=3
    ),
    max_leaves=8,
)


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
@given(a=_layers, b=_layers, c=_layers)
def test_merge_precedence_property(
    a: dict[str, object], b: dict[str, object], c: dict[str, object]
) -> None:
    # associativa
    assert deep_merge(deep_merge(a, b), c) == deep_merge(a, deep_merge(b, c))
    # vazio é identidade, dos dois lados
    assert deep_merge(a, {}) == a
    assert deep_merge({}, a) == a
    # a camada mais à direita vence nas folhas que ela define
    merged = deep_merge(a, b)
    for key, value in b.items():
        if not isinstance(value, dict):
            assert merged[key] == value


@pytest.mark.xfail(strict=True, raises=NotImplementedError)
def test_config_is_frozen() -> None:
    cfg = AppConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.handedness = "left"  # type: ignore[misc]
