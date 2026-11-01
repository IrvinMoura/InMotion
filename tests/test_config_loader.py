"""Spec 004 — carregamento de YAML, camadas e sobrescritas (adaptador)."""

from __future__ import annotations

from pathlib import Path

import pytest

from inmotion.config.loader import load_config
from inmotion.config.validate import ConfigError


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


DEFAULT = """
camera:
  index: 3
  width: 800
filters:
  kind: ema
  ema:
    alpha: 0.3
"""

GAME = """
name: teste
camera:
  width: 1280
bindings:
  jump: {input: space, kind: hold}
  attack: {input: mouse_left, kind: trigger}
"""

PROFILE = """
filters:
  ema:
    alpha: 0.6
bindings:
  jump: {input: w, kind: hold}
"""


def test_layers_and_overrides(tmp_path: Path) -> None:
    """Precedência: default ← jogo ← perfil ← --set (a da direita vence)."""
    default = write(tmp_path / "default.yaml", DEFAULT)
    write(tmp_path / "games" / "teste.yaml", GAME)
    profile = write(tmp_path / "perfil.yaml", PROFILE)

    cfg, prof = load_config(
        default,
        game="teste",
        profile_path=profile,
        overrides=["camera.index=5", "filters.ema.alpha=0.9", "camera.index=6"],
        games_dir=tmp_path / "games",
    )

    assert cfg.camera.index == 6  # --set, e o último --set vence
    assert cfg.camera.width == 1280  # jogo sobre o default
    assert cfg.filters.ema.alpha == 0.9  # --set sobre o perfil (0.6) e o default (0.3)
    assert cfg.filters.kind == "ema"  # só no default
    assert prof is not None
    assert prof.name == "teste"
    assert prof.bindings["jump"].input == "w"  # perfil sobre o jogo
    assert prof.bindings["attack"].input == "mouse_left"  # herdado do jogo


def test_without_game_there_is_no_profile(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", DEFAULT)
    cfg, prof = load_config(default)
    assert prof is None
    assert cfg.camera.index == 3


def test_override_value_is_read_as_yaml(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", "{}")
    cfg, _ = load_config(
        default,
        overrides=["camera.exposure=-6", "overlay.enabled=false", r'pose.model_dir="C:\\m"'],
    )
    assert cfg.camera.exposure == -6.0
    assert cfg.overlay.enabled is False
    assert str(cfg.pose.model_dir) == r"C:\m"


def test_python_tag_rejected(tmp_path: Path) -> None:
    """CA8: ``safe_load`` não constrói objetos Python; nada é executado."""
    marker = tmp_path / "executou.txt"
    default = write(
        tmp_path / "default.yaml",
        f"camera: !!python/object/apply:builtins.open [{marker.as_posix()}, w]\n",
    )
    with pytest.raises(ConfigError) as exc:
        load_config(default)
    assert "default.yaml" in str(exc.value)
    assert not marker.exists()


def test_set_without_equals_errors(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", "{}")
    with pytest.raises(ConfigError, match="--set"):
        load_config(default, overrides=["filters.ema.alpha"])


def test_invalid_set_value_yaml(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", "{}")
    with pytest.raises(ConfigError, match="--set"):
        load_config(default, overrides=["camera.index=[1, 2"])


def test_errors_cite_the_source_file(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", DEFAULT)
    bad = write(tmp_path / "ruim.yaml", "filters:\n  ema:\n    alpah: 0.4\n")
    with pytest.raises(ConfigError) as exc:
        load_config(default, profile_path=bad)
    assert exc.value.path == "filters.ema.alpah"
    assert "alpha" in str(exc.value)
    assert "ruim.yaml" in str(exc.value)

    with pytest.raises(ConfigError) as exc:
        load_config(default, overrides=["filters.ema.alpha=2"])
    assert exc.value.path == "filters.ema.alpha"
    assert "--set" in str(exc.value)


def test_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"nao_existe\.yaml"):
        load_config(tmp_path / "nao_existe.yaml")
    lista = write(tmp_path / "lista.yaml", "- 1\n- 2\n")
    with pytest.raises(ConfigError, match=r"lista\.yaml"):
        load_config(lista)
    quebrado = write(tmp_path / "quebrado.yaml", "a: [1, 2\n")
    with pytest.raises(ConfigError, match=r"quebrado\.yaml"):
        load_config(quebrado)
    # arquivo vazio vale como camada vazia
    vazio = write(tmp_path / "vazio.yaml", "")
    assert load_config(vazio)[0].camera.index == 1


def test_unknown_game_lists_available(tmp_path: Path) -> None:
    default = write(tmp_path / "default.yaml", "{}")
    write(tmp_path / "games" / "minecraft.yaml", "bindings: {}")
    with pytest.raises(ConfigError) as exc:
        load_config(default, game="minecrft", games_dir=tmp_path / "games")
    assert "minecraft" in str(exc.value)
    assert exc.value.path == "--game"


def test_default_yaml_matches_dataclass_defaults() -> None:
    """CA1: o ``config/default.yaml`` real carrega e é igual aos padrões das dataclasses."""
    from inmotion.config.schema import AppConfig

    config, profile = load_config()
    assert config == AppConfig()
    assert profile is None
    assert config.camera.index == 1  # padrão provisório (índice 0 = câmera virtual)
    assert config.safety.min_track_s == 1.0
    assert config.safety.min_torso_visibility == 0.5


def test_minecraft_profile_loads_and_matches_backend_keys() -> None:
    from inmotion.config.schema import validate_profile
    from inmotion.gestures.base import GestureKind
    from inmotion.gestures.factory import BODY_GESTURE_NAMES
    from inmotion.input.keyboard_mouse import KeyboardMouseBackend
    from tests.fakes import FakeDirectInput

    _, profile = load_config(game="minecraft")
    assert profile is not None
    assert {n: (b.input, b.kind) for n, b in profile.bindings.items()} == {
        "attack": ("mouse_left", GestureKind.HOLD),
        "use": ("mouse_right", GestureKind.HOLD),
        "jump": ("space", GestureKind.TRIGGER),
        "crouch": ("shift", GestureKind.HOLD),
        "walk": ("w", GestureKind.HOLD),
    }
    keys = KeyboardMouseBackend(api=FakeDirectInput()).supported_keys()
    validate_profile(profile, BODY_GESTURE_NAMES, keys)


def test_camera_default_comes_only_from_yaml() -> None:
    from inmotion.config.loader import load_camera_config

    assert load_camera_config().index == load_config()[0].camera.index == 1
    assert load_camera_config(index=3, width=800).width == 800
    assert load_camera_config(index=3).index == 3
    assert load_camera_config(index=None).index == 1


def test_game_layer_arbiter_section_reaches_the_profile(tmp_path: Path) -> None:
    """Spec 017: a seção ``arbiter`` do jogo chega ao ``GameProfile`` (não ao ``AppConfig``)."""
    games = tmp_path / "games"
    write(
        games / "teste.yaml",
        GAME + "arbiter:\n  priority: [jump, attack]\n  exclusive: [[jump, attack]]\n",
    )
    default = write(tmp_path / "default.yaml", DEFAULT)
    _, profile = load_config(default_path=default, game="teste", games_dir=games)
    assert profile is not None
    assert profile.arbiter.priority == ("jump", "attack")
    assert profile.arbiter.exclusive == (frozenset({"jump", "attack"}),)


def test_arbiter_error_names_the_file_that_defined_it(tmp_path: Path) -> None:
    games = tmp_path / "games"
    game = write(games / "teste.yaml", GAME + "arbiter:\n  priority: jump\n")
    default = write(tmp_path / "default.yaml", DEFAULT)
    with pytest.raises(ConfigError) as exc:
        load_config(default_path=default, game="teste", games_dir=games)
    assert exc.value.path == "arbiter.priority"
    assert str(game) in str(exc.value)
