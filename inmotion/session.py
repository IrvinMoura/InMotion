"""Montagem compartilhada do pipeline para execução, replay e simulação sem hardware."""

from inmotion.config.loader import load_minecraft_profile
from inmotion.config.schema import AppConfig, FilterConfig, GameProfile
from inmotion.filters.ema import Ema
from inmotion.filters.one_euro import OneEuro
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.factory import build_body_gestures
from inmotion.pipeline import Pipeline
from inmotion.safety import SafetyManager
from inmotion.vision.landmarks import Side
from inmotion.voice import VoiceConfig

DEFAULT_PROFILE = load_minecraft_profile()


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
    voice: VoiceConfig | None = None,
) -> Pipeline:
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
        head_look=config.head_look if config.head_look.enabled else None,
        voice=voice,
    )
