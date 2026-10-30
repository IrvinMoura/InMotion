"""Spec 008 T10 — perfis de jogo reais: os gestos e as teclas existem."""

from __future__ import annotations

from inmotion.app import DEFAULT_PROFILE
from inmotion.config.loader import load_config
from inmotion.config.schema import validate_profile
from inmotion.gestures.factory import BODY_GESTURE_NAMES
from inmotion.input.keyboard_mouse import KeyboardMouseBackend
from tests.fakes import FakeDirectInput


def test_minecraft_profile_valid() -> None:
    """Todo gesto do perfil existe e toda tecla é suportada (teclas ``space``/``shift``/``w``)."""
    _, profile = load_config(game="minecraft")
    assert profile is not None
    assert set(profile.bindings) == BODY_GESTURE_NAMES
    keys = KeyboardMouseBackend(api=FakeDirectInput()).supported_keys()
    validate_profile(profile, BODY_GESTURE_NAMES, keys)


def test_minecraft_profile_valid_with_real_pydirectinput_keys() -> None:
    """Mesma checagem com a tabela de scancodes do pydirectinput de verdade."""
    import pydirectinput

    _, profile = load_config(game="minecraft")
    assert profile is not None
    keys = KeyboardMouseBackend(api=pydirectinput).supported_keys()
    validate_profile(profile, BODY_GESTURE_NAMES, keys)


def test_default_profile_equals_minecraft_yaml() -> None:
    """O perfil sem ``--game`` não pode divergir de ``config/games/minecraft.yaml``."""
    _, profile = load_config(game="minecraft")
    assert profile == DEFAULT_PROFILE
