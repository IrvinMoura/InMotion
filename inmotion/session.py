"""Montagem compartilhada do pipeline para execução, replay e simulação sem hardware."""

from inmotion.config.schema import AppConfig, Binding, FilterConfig, GameProfile
from inmotion.filters.ema import Ema
from inmotion.filters.one_euro import OneEuro
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.arbiter import ArbiterConfig
from inmotion.gestures.base import GestureKind
from inmotion.gestures.factory import build_body_gestures
from inmotion.pipeline import Pipeline
from inmotion.safety import SafetyManager
from inmotion.vision.landmarks import Side

DEFAULT_PROFILE = GameProfile(
    "minecraft",
    {
        "attack": Binding("mouse_left", GestureKind.HOLD),
        "use": Binding("mouse_right", GestureKind.HOLD),
        "jump": Binding("space", GestureKind.TRIGGER),
        "crouch": Binding("shift", GestureKind.HOLD),
        "walk": Binding("w", GestureKind.HOLD),
        "sprint": Binding("ctrl", GestureKind.HOLD),
    },
    ArbiterConfig(
        priority=("both_arms", "crouch", "jump", "attack", "use", "walk", "sprint"),
        exclusive=(
            frozenset({"crouch", "jump"}),
            frozenset({"crouch", "walk"}),
            frozenset({"crouch", "sprint"}),
        ),
        combos={"both_arms": frozenset({"attack", "use"})},
    ),
)


def pose_filter_from_config(config: FilterConfig) -> PoseFilter | None:
    """Cria filtros independentes para coordenadas de imagem e mundo."""
    if config.kind == "none":
        return None
    if config.kind == "ema":
        return PoseFilter(lambda: Ema(config.ema))
    return PoseFilter(lambda: OneEuro(config.one_euro))


def create_pipeline(
    safety: SafetyManager,
    config: AppConfig,
    profile: GameProfile,
    pose_filter: PoseFilter | None,
) -> Pipeline:
    """Usa os mesmos parâmetros de gestos, filtros, bases e mapeamento em todo fluxo."""
    gestures = build_body_gestures(
        profile.bindings,
        Side(config.handedness),
        config.gestures,
        config.geometry,
        config.crouch.feature,
        config.march,
    )
    return Pipeline(
        safety,
        gestures,
        FpsMeter(window=30),
        pose_filter,
        profile=profile,
        mapper_params=config.mapper,
        baseline_params=config.baseline,
        march_params=config.march,
        geometry=config.geometry,
    )
