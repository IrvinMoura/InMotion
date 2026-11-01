"""Spec 008 T10 — perfis de jogo reais: os gestos e as teclas existem."""

from __future__ import annotations

from inmotion.app import DEFAULT_PROFILE
from inmotion.config.loader import load_config
from inmotion.config.schema import validate_profile
from inmotion.gestures.factory import BODY_GESTURE_NAMES
from inmotion.input.keyboard_mouse import KeyboardMouseBackend
from tests.fakes import FakeDirectInput


def test_minecraft_profile_valid() -> None:
    """Todo gesto do perfil existe e toda tecla é suportada (teclas ``space``/``shift``)."""
    _, profile = load_config(game="minecraft")
    assert profile is not None
    assert set(profile.bindings) == BODY_GESTURE_NAMES - {"walk"}  # `walk` só na spec 018
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


def test_minecraft_arbiter() -> None:
    """Spec 017 RF2/RF7: a seção ``arbiter`` do perfil lista só gestos que existem hoje."""
    from inmotion.gestures.arbiter import ArbiterConfig
    from inmotion.gestures.factory import build_arbiter

    _, profile = load_config(game="minecraft")
    assert profile is not None
    assert profile.arbiter == ArbiterConfig(
        priority=("both_arms", "crouch", "jump", "attack", "use"),
        exclusive=(frozenset({"crouch", "jump"}),),
        combos={"both_arms": frozenset({"attack", "use"})},
    )
    arbiter, combo = build_arbiter(set(profile.bindings), profile.arbiter)
    assert combo is not None
    assert combo.members == {"attack", "use"}
    from inmotion.gestures.base import EventKind, GestureEvent

    start, pulse = EventKind.START, EventKind.PULSE
    out = arbiter.resolve(
        [GestureEvent("crouch", start, 0.0), GestureEvent("jump", pulse, 0.0)], 0.0
    )
    assert [e.gesture for e in out] == ["crouch"]  # agachar vence o pulo (RF2)


def test_minecraft_rest_presses_nothing() -> None:
    """CA11: parado, inclusive com o tronco inclinado acima do antigo limiar do ``walk``
    (0,30 rad), o perfil do Minecraft não aperta nenhuma tecla."""
    from inmotion.fps import FpsMeter
    from inmotion.gestures.factory import build_body_gestures
    from inmotion.input.backend import MOUSE_BUTTONS
    from inmotion.pipeline import Pipeline
    from inmotion.safety import SafetyConfig, SafetyManager, TrackedInput
    from tests.fakes import FakeInputBackend
    from tests.helpers import make_body

    config, profile = load_config(game="minecraft")
    assert profile is not None
    fake = FakeInputBackend()
    safety = SafetyManager(TrackedInput(fake, MOUSE_BUTTONS), SafetyConfig(min_track_s=0.0))
    gestures = build_body_gestures(profile.bindings, geometry=config.geometry)
    pipeline = Pipeline(safety, gestures, FpsMeter(window=30), profile=profile)
    dt = 1 / 30
    for i in range(120):
        pipeline.step(make_body(t=i * dt, lean=0.35), i * dt)
    assert fake.events == []
    assert fake.pressed == set()
