"""Spec 004 — construção, validação e mesclagem da configuração (núcleo puro)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from inmotion.config.schema import (
    AppConfig,
    CameraConfig,
    FilterConfig,
    PoseConfig,
)
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


def test_build_nested_dataclass() -> None:
    cfg = build(Outer, {"inner": {"leaf": {"alpha": 0.25}, "count": 7}, "flag": True})
    assert cfg == Outer(Inner(Leaf(alpha=0.25), count=7), flag=True)
    assert build(Outer, {"inner": {}}) == Outer(Inner())  # padrões preenchidos


def test_missing_field_has_path() -> None:
    with pytest.raises(ConfigError) as exc:
        build(Outer, {})
    assert exc.value.path == "inner"
    assert "inner" in str(exc.value)


def test_unknown_key_suggests_name() -> None:
    with pytest.raises(ConfigError) as exc:
        build(Outer, {"inner": {"leaf": {"alpah": 0.3}}})
    assert exc.value.path == "inner.leaf.alpah"
    assert "alpha" in str(exc.value)


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


def test_config_is_frozen() -> None:
    cfg = AppConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.handedness = "left"  # type: ignore[misc]


class Color(Enum):
    RED = 1
    GREEN = 2


@dataclass(frozen=True)
class Kitchen:
    flag: bool = False
    count: int = 1
    ratio: float = 1.0
    text: str = "a"
    where: Path = Path("x")
    mode: Literal["a", "b"] = "a"
    maybe: float | None = None
    keys: frozenset[str] = frozenset()
    pair: tuple[float, ...] = ()
    table: dict[str, Leaf] = field(default_factory=dict)
    color: Color = Color.RED


def test_build_converts_all_supported_types() -> None:
    cfg = build(
        Kitchen,
        {
            "flag": True,
            "count": 4,
            "ratio": 2,  # int vale como float
            "text": "z",
            "where": "dir/file.task",
            "mode": "b",
            "maybe": 0.5,
            "keys": ["a", "b", "a"],
            "pair": [1, 2.5],
            "table": {"g": {"alpha": 0.1}},
            "color": "green",
        },
    )
    assert cfg == Kitchen(
        flag=True,
        count=4,
        ratio=2.0,
        text="z",
        where=Path("dir/file.task"),
        mode="b",
        maybe=0.5,
        keys=frozenset({"a", "b"}),
        pair=(1.0, 2.5),
        table={"g": Leaf(alpha=0.1)},
        color=Color.GREEN,
    )
    assert isinstance(cfg.ratio, float)
    assert build(Kitchen, {"maybe": None}).maybe is None


@pytest.mark.parametrize(
    ("data", "path", "needle"),
    [
        ({"flag": 1}, "flag", "bool"),
        ({"count": 1.5}, "count", "int"),
        ({"count": True}, "count", "int"),
        ({"ratio": "x"}, "ratio", "float"),
        ({"text": 3}, "text", "str"),
        ({"mode": "c"}, "mode", "'a'"),
        ({"keys": "a"}, "keys", "lista"),
        ({"pair": [1, "x"]}, "pair[1]", "float"),
        ({"table": {"g": 3}}, "table.g", "mapa"),
        ({"color": "blue"}, "color", "red"),
        ({"maybe": "x"}, "maybe", "float"),
    ],
)
def test_wrong_type_or_value_reports_path_received_and_expected(
    data: dict[str, object], path: str, needle: str
) -> None:
    with pytest.raises(ConfigError) as exc:
        build(Kitchen, data)
    assert exc.value.path == path
    assert needle in str(exc.value)
    assert "recebido" in str(exc.value)


def test_data_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError) as exc:
        build(Outer, {"inner": [1, 2]})
    assert exc.value.path == "inner"


def test_unsupported_type_raises_type_error() -> None:
    @dataclass(frozen=True)
    class Bad:
        value: complex = 0j

    with pytest.raises(TypeError, match="complex"):
        build(Bad, {"value": 1})


def test_non_dataclass_raises_type_error() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        build(int, {})  # type: ignore[type-var]


def test_app_config_composes_module_params_without_duplicating() -> None:
    from inmotion.exit_zone import ExitZoneConfig
    from inmotion.gestures.base import GestureParams
    from inmotion.safety import SafetyConfig

    cfg = build(AppConfig, {"gestures": {"jump": {"theta_on": 0.1, "theta_off": 0.2}}})
    assert cfg.safety == SafetyConfig()
    assert cfg.exit_zone == ExitZoneConfig()
    assert cfg.filters == FilterConfig()
    assert cfg.gestures == {"jump": GestureParams(theta_on=0.1, theta_off=0.2)}
    assert cfg.camera.index == 1


def test_app_config_error_paths_are_full() -> None:
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"filters": {"ema": {"alpha": 1.5}}})
    assert exc.value.path == "filters.ema.alpha"
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"safety": {"min_track_s": -1}})
    assert exc.value.path == "safety.min_track_s"
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"filters": {"ema": {"alpah": 0.4}}})
    assert "alpha" in str(exc.value)
    assert exc.value.path == "filters.ema.alpah"


@pytest.mark.parametrize(
    ("direction", "on", "off"),
    [("below", 0.5, 0.5), ("below", 0.6, 0.5), ("above", 0.5, 0.6), ("ABOVE", 0.4, 0.4)],
)
def test_gesture_hysteresis_order_is_validated(direction: str, on: float, off: float) -> None:
    gesture = {"theta_on": on, "theta_off": off, "direction": direction}
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"gestures": {"jump": gesture}})
    assert exc.value.path == "gestures.jump"
    assert "theta_on" in str(exc.value)


def test_gesture_direction_enum_by_name() -> None:
    from inmotion.gestures.base import Direction

    gesture = {"theta_on": 0.6, "theta_off": 0.5, "direction": "above"}
    cfg = build(AppConfig, {"gestures": {"jump": gesture}})
    assert cfg.gestures["jump"].direction is Direction.ABOVE


def test_moved_configs_keep_their_validation() -> None:
    with pytest.raises(ConfigError, match="width"):
        build(CameraConfig, {"width": 0})
    with pytest.raises(ConfigError, match="variant"):
        build(PoseConfig, {"variant": "ultra"})
