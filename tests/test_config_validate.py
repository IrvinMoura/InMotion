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
    Binding,
    CameraConfig,
    FilterConfig,
    GameProfile,
    PoseConfig,
    validate_profile,
)
from inmotion.config.validate import ConfigError, build, deep_merge, parse_override
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


_leaf_values = st.one_of(
    st.integers(),
    st.text(max_size=3),
    st.booleans(),
    st.none(),
    st.lists(st.integers(), max_size=2),
)


def _layer(depth: int) -> st.SearchStrategy[dict[str, object]]:
    """Camada com forma consistente: ``a``/``b`` sempre folhas, ``m``/``n`` sempre mapas.

    É o caso real (todas as camadas obedecem ao mesmo esquema). Misturar mapa e folha na
    mesma chave quebra a associatividade (``{a:1} ⊕ 0 ⊕ {b:2}``) e o ``build`` rejeita o
    tipo errado de qualquer forma.
    """
    leaves = st.dictionaries(st.sampled_from(["a", "b"]), _leaf_values, max_size=2)
    if depth == 0:
        return leaves
    maps = st.dictionaries(st.sampled_from(["m", "n"]), _layer(depth - 1), max_size=2)
    return st.builds(lambda x, y: {**x, **y}, leaves, maps)


_layers = _layer(2)


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


def test_merge_recursive_lists_replaced_and_inputs_untouched() -> None:
    a = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    b = {"x": {"z": [9]}, "k": {"n": 2}}
    merged = deep_merge(a, b)
    assert merged == {"x": {"y": 1, "z": [9]}, "k": {"n": 2}}
    assert a == {"x": {"y": 1, "z": [1, 2]}, "k": 1}  # entradas intactas
    merged["x"]["y"] = 5  # type: ignore[index]
    assert a["x"]["y"] == 1  # sem aliasing
    assert deep_merge() == {}


def test_parse_override() -> None:
    assert parse_override("filters.ema.alpha=0.4") == (("filters", "ema", "alpha"), "0.4")
    assert parse_override("camera.index=2") == (("camera", "index"), "2")
    assert parse_override("a=b=c") == (("a",), "b=c")  # só o primeiro '=' separa
    assert parse_override("pose.model_dir=") == (("pose", "model_dir"), "")


@pytest.mark.parametrize("text", ["filters.ema.alpha", "=3", "a..b=1", ".a=1", "a b=1"])
def test_parse_override_errors(text: str) -> None:
    with pytest.raises(ConfigError) as exc:
        parse_override(text)
    assert "--set" in str(exc.value)


KNOWN = frozenset({"jump", "attack", "use"})
KEYS = frozenset({"space", "mouse_left", "mouse_right"})


def _profile(bindings: dict[str, object]) -> GameProfile:
    return build(GameProfile, {"name": "teste", "bindings": bindings})


def test_profile_builds_and_validates() -> None:
    from inmotion.gestures.base import GestureKind

    profile = _profile(
        {
            "jump": {"input": "space", "kind": "hold"},
            "attack": {"input": "mouse_left", "kind": "trigger"},
        }
    )
    assert profile.bindings["jump"] == Binding("space", GestureKind.HOLD)
    assert profile.bindings["attack"].kind is GestureKind.TRIGGER
    validate_profile(profile, KNOWN, KEYS)  # não lança


def test_profile_rejects_unknown_gesture() -> None:
    profile = _profile({"jumpp": {"input": "space", "kind": "hold"}})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "bindings.jumpp"
    assert "jump" in str(exc.value)  # sugestão


def test_profile_rejects_unsupported_key() -> None:
    profile = _profile({"jump": {"input": "f13", "kind": "hold"}})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "bindings.jump.input"
    assert "f13" in str(exc.value)


def test_profile_rejects_bad_kind_and_empty_input() -> None:
    with pytest.raises(ConfigError) as exc:
        _profile({"jump": {"input": "space", "kind": "segurar"}})
    assert exc.value.path == "bindings.jump.kind"
    with pytest.raises(ConfigError) as exc:
        _profile({"jump": {"input": "", "kind": "hold"}})
    assert exc.value.path == "bindings.jump.input"


def test_safety_watchdog_timeout_is_validated() -> None:
    assert build(AppConfig, {}).safety.watchdog_timeout_s == 0.5
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"safety": {"watchdog_timeout_s": 0}})
    assert exc.value.path == "safety.watchdog_timeout_s"


# --- Spec 017: seção ``arbiter`` do perfil (CA2) -----------------------------------------------


def _arbiter_profile(arbiter: dict[str, object]) -> GameProfile:
    bindings = {"jump": {"input": "space", "kind": "hold"}}
    return build(GameProfile, {"name": "teste", "bindings": bindings, "arbiter": arbiter})


def test_arbiter_section_builds_into_an_immutable_config() -> None:
    """CA1/CA2: a seção do YAML vira ``ArbiterConfig`` (listas viram tuplas e conjuntos)."""
    profile = _arbiter_profile(
        {
            "priority": ["both_arms", "jump", "attack", "use"],
            "exclusive": [["jump", "attack"]],
            "combos": {"both_arms": ["attack", "use"]},
        }
    )
    assert profile.arbiter.priority == ("both_arms", "jump", "attack", "use")
    assert profile.arbiter.exclusive == (frozenset({"jump", "attack"}),)
    assert profile.arbiter.combos == {"both_arms": frozenset({"attack", "use"})}
    validate_profile(profile, KNOWN, KEYS)  # não lança


def test_profile_without_arbiter_section_uses_the_default() -> None:
    """Premissa 4: sem a seção vale o padrão anterior, e ele não é validado contra os gestos."""
    from inmotion.gestures.arbiter import DEFAULT_ARBITER

    profile = _profile({"jump": {"input": "space", "kind": "hold"}})
    assert profile.arbiter == DEFAULT_ARBITER
    validate_profile(profile, KNOWN, KEYS)  # `crouch` e `walk` do padrão não existem em KNOWN


def test_arbiter_rejects_repeated_name_in_priority() -> None:
    profile = _arbiter_profile({"priority": ["jump", "attack", "jump"]})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "arbiter.priority[2]"
    assert "repetido" in exc.value.message
    assert "jump" in exc.value.message


def test_arbiter_rejects_unknown_name_with_suggestion() -> None:
    profile = _arbiter_profile({"priority": ["jump", "atack"]})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "arbiter.priority[1]"
    assert "desconhecido" in exc.value.message
    assert "attack" in exc.value.message  # sugestão


def test_arbiter_rejects_unknown_name_in_exclusive_group() -> None:
    profile = _arbiter_profile({"priority": ["jump"], "exclusive": [["jump", "voar"]]})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "arbiter.exclusive[0]"
    assert "voar" in exc.value.message


@pytest.mark.parametrize(
    ("combos", "word"),
    [
        ({"both": []}, "vazio"),
        ({"both": ["both", "attack"]}, "si mesmo"),
        ({"both": ["attack", "voar"]}, "desconhecido"),
    ],
)
def test_arbiter_rejects_bad_combo(combos: dict[str, list[str]], word: str) -> None:
    profile = _arbiter_profile({"priority": ["both", "attack", "use"], "combos": combos})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "arbiter.combos.both"
    assert word in exc.value.message


def test_arbiter_rejects_combo_missing_from_priority() -> None:
    profile = _arbiter_profile({"priority": ["attack", "use"], "combos": {"both": ["attack"]}})
    with pytest.raises(ConfigError) as exc:
        validate_profile(profile, KNOWN, KEYS)
    assert exc.value.path == "arbiter.combos.both"
    assert "priority" in exc.value.message


def test_arbiter_wrong_types_are_config_errors() -> None:
    with pytest.raises(ConfigError) as exc:
        _arbiter_profile({"priority": "jump"})
    assert exc.value.path == "arbiter.priority"
    with pytest.raises(ConfigError) as exc:
        _arbiter_profile({"priority": ["jump"], "combos": {"both": "attack"}})
    assert exc.value.path == "arbiter.combos.both"
    with pytest.raises(ConfigError) as exc:
        _arbiter_profile({"prioridade": ["jump"]})
    assert exc.value.path == "arbiter.prioridade"


def test_crouch_feature_is_validated() -> None:
    """Spec 017 RF3: ``crouch.feature`` é ``shoulders`` (padrão) ou ``hip``."""
    assert build(AppConfig, {}).crouch.feature == "shoulders"
    assert build(AppConfig, {"crouch": {"feature": "hip"}}).crouch.feature == "hip"
    with pytest.raises(ConfigError) as exc:
        build(AppConfig, {"crouch": {"feature": "joelho"}})
    assert exc.value.path == "crouch.feature"
