"""Esquema da configuração (núcleo puro, spec 004).

Cada módulo é dono dos seus parâmetros (``SafetyConfig``, ``EmaParams``, ``GestureParams``,
``ExitZoneConfig``...); aqui eles só são **compostos** no ``AppConfig``. As configurações de
câmera, pose e pré-processamento moram aqui (e não nos adaptadores) para que o esquema não
importe ``cv2`` nem ``mediapipe`` por tabela. Os padrões são os de ``config/default.yaml``
(um teste garante que os dois coincidem).
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from inmotion.config.validate import ConfigError
from inmotion.exit_zone import ExitZoneConfig
from inmotion.filters.ema import EmaParams
from inmotion.filters.one_euro import OneEuroParams
from inmotion.gestures.arbiter import DEFAULT_ARBITER, ArbiterConfig
from inmotion.gestures.base import GestureKind, GestureParams
from inmotion.gestures.crouch import CrouchFeature
from inmotion.gestures.factory import DEFAULT_GESTURE_PARAMS
from inmotion.gestures.features import BaselineParams
from inmotion.gestures.march import MarchParams
from inmotion.input.backend import KeyName
from inmotion.mapping import MapperParams
from inmotion.safety import SafetyConfig
from inmotion.vision.geometry import GeometryConfig

CAMERA_BACKENDS: tuple[str, ...] = ("dshow", "msmf", "any")
POSE_VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
_POSE_CONFIDENCE_FIELDS: tuple[str, ...] = (
    "min_detection_confidence",
    "min_presence_confidence",
    "min_tracking_confidence",
)

CameraBackend = Literal["dshow", "msmf", "any"]
Variant = Literal["lite", "full", "heavy"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass(frozen=True)
class CameraConfig:
    """Parâmetros da câmera.

    ``index`` padrão é provisório: no PC de desenvolvimento o índice 0 é uma câmera virtual.
    """

    index: int = 1
    backend: CameraBackend = "dshow"
    width: int = 640
    height: int = 480
    fps: int = 60
    fourcc: str = "MJPG"
    exposure: float | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index deve ser >= 0: {self.index}")
        if self.backend not in CAMERA_BACKENDS:
            raise ValueError(f"backend inválido: {self.backend!r} (use {sorted(CAMERA_BACKENDS)})")
        for name in ("width", "height", "fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} deve ser > 0: {getattr(self, name)}")
        if len(self.fourcc) != 4:
            raise ValueError(f"fourcc deve ter 4 caracteres: {self.fourcc!r}")
        if self.exposure is not None and not math.isfinite(self.exposure):
            raise ValueError(f"exposure deve ser finito ou None: {self.exposure}")


@dataclass(frozen=True)
class PoseConfig:
    """Configuração do estimador. Confianças em [0, 1]; variante lite/full/heavy."""

    model_dir: Path = Path("models")
    variant: Variant = "full"
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.variant not in POSE_VARIANTS:
            raise ValueError(f"variant deve ser um de {POSE_VARIANTS}, veio {self.variant!r}")
        for name in _POSE_CONFIDENCE_FIELDS:
            value = getattr(self, name)
            if not (math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f"{name} deve estar em [0, 1], veio {value}")

    @property
    def model_path(self) -> Path:
        """Caminho do modelo: ``<model_dir>/pose_landmarker_<variant>.task``."""
        return self.model_dir / f"pose_landmarker_{self.variant}.task"


@dataclass(frozen=True)
class PreprocessConfig:
    """``blur_sigma`` em pixels (``None`` desliga a suavização); ``sobel_debug`` liga a vista."""

    blur_sigma: float | None = 1.0
    sobel_debug: bool = False

    def __post_init__(self) -> None:
        sigma = self.blur_sigma
        if sigma is not None and (not math.isfinite(sigma) or sigma <= 0):
            raise ValueError(f"blur_sigma deve ser None ou finito e > 0, recebido {sigma}")


@dataclass(frozen=True)
class FilterConfig:
    """Filtro temporal da pose: ``kind`` escolhe qual dos parâmetros vale (R1-R4)."""

    kind: Literal["none", "ema", "one_euro"] = "one_euro"
    ema: EmaParams = field(default_factory=EmaParams)
    one_euro: OneEuroParams = field(default_factory=OneEuroParams)


@dataclass(frozen=True)
class OverlayConfig:
    """Janela de debug: ``enabled=False`` equivale a ``--no-overlay`` (mais FPS)."""

    enabled: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    """Nível de log e arquivo opcional (nome simples vai para ``logs/``)."""

    level: LogLevel = "INFO"
    file: str | None = None


@dataclass(frozen=True)
class CrouchConfig:
    """Sinal do agachar (spec 017 RF3): ``shoulders`` (padrão) ou ``hip`` (spec 008).

    Os limiares de ``gestures.crouch`` valem para o sinal escolhido: trocar a feature exige
    trocar também ``theta_on`` e ``theta_off``.
    """

    feature: CrouchFeature = "shoulders"


@dataclass(frozen=True)
class AppConfig:
    """Configuração completa do aplicativo (``config/default.yaml`` ← jogo ← perfil ← CLI)."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    gestures: dict[str, GestureParams] = field(default_factory=lambda: dict(DEFAULT_GESTURE_PARAMS))
    crouch: CrouchConfig = field(default_factory=CrouchConfig)
    baseline: BaselineParams = field(default_factory=BaselineParams)
    march: MarchParams = field(default_factory=MarchParams)
    mapper: MapperParams = field(default_factory=MapperParams)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    exit_zone: ExitZoneConfig = field(default_factory=ExitZoneConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    handedness: Literal["right", "left"] = "right"


@dataclass(frozen=True)
class Binding:
    """Liga um gesto a uma tecla/botão: ``kind`` é ``hold``, ``trigger`` ou ``analog``."""

    input: KeyName
    kind: GestureKind

    def __post_init__(self) -> None:
        if not self.input:
            raise ValueError("input não pode ser vazio")


@dataclass(frozen=True)
class GameProfile:
    """Perfil de jogo: para cada gesto, o input mapeado e o tipo (``config/games/*.yaml``).

    ``arbiter`` (spec 017) define prioridade, grupos exclusivos e combos; sem a seção vale o
    padrão anterior (``DEFAULT_ARBITER``).
    """

    name: str
    bindings: dict[str, Binding]
    arbiter: ArbiterConfig = DEFAULT_ARBITER

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name não pode ser vazio")


def _suggestion(name: str, options: Collection[str]) -> str:
    close = difflib.get_close_matches(name, sorted(options), n=1)
    return f" Você quis dizer {close[0]!r}?" if close else ""


def validate_profile(
    profile: GameProfile, known_gestures: frozenset[str], supported_keys: frozenset[KeyName]
) -> None:
    """RF7: todo gesto do perfil existe e toda tecla/botão é suportado pelo backend.

    Valida também a seção ``arbiter`` (spec 017), exceto quando é o padrão (perfil sem a seção).

    Raises:
        ConfigError: com o caminho ``bindings.<gesto>``, ``bindings.<gesto>.input`` ou
            ``arbiter.*`` (``priority[i]``, ``exclusive[i]``, ``combos.<nome>``).
    """
    for gesture, binding in profile.bindings.items():
        path = f"bindings.{gesture}"
        if gesture not in known_gestures:
            raise ConfigError(
                path,
                f"gesto desconhecido {gesture!r}.{_suggestion(gesture, known_gestures)} "
                f"Gestos conhecidos: {sorted(known_gestures)}",
            )
        if binding.input not in supported_keys:
            hint = _suggestion(binding.input, supported_keys)
            raise ConfigError(
                f"{path}.input",
                f"tecla ou botão não suportado {binding.input!r}.{hint}"
                " Veja as teclas aceitas pelo backend de input.",
            )

    if profile.arbiter != DEFAULT_ARBITER:
        _validate_arbiter(profile.arbiter, known_gestures)


def _validate_arbiter(arbiter: ArbiterConfig, known_gestures: frozenset[str]) -> None:
    """Spec 017 RF2: nomes sem repetição e conhecidos; combos não vazios e fora de si mesmos.

    Um nome vale se for um gesto existente ou um combo declarado em ``combos``. Os membros de
    um combo e os de um grupo exclusivo devem ser gestos existentes, e todo combo precisa
    constar em ``priority`` (senão a supressão que ele faz nunca valeria).
    """
    combos = set(arbiter.combos)
    names = known_gestures | combos
    seen: set[str] = set()
    for i, name in enumerate(arbiter.priority):
        path = f"arbiter.priority[{i}]"
        if name in seen:
            raise ConfigError(path, f"nome repetido {name!r}; cada nome aparece uma vez")
        seen.add(name)
        if name not in names:
            raise ConfigError(path, _unknown_name(name, names))
    for i, group in enumerate(arbiter.exclusive):
        for name in sorted(group - names):
            raise ConfigError(f"arbiter.exclusive[{i}]", _unknown_name(name, names))
    for combo, members in arbiter.combos.items():
        path = f"arbiter.combos.{combo}"
        if combo not in seen:
            raise ConfigError(path, f"o combo {combo!r} precisa constar em arbiter.priority")
        if not members:
            raise ConfigError(path, "combo vazio: liste os gestos que ele reúne")
        if combo in members:
            raise ConfigError(path, f"o combo {combo!r} não pode conter a si mesmo")
        for name in sorted(members - known_gestures):
            raise ConfigError(path, _unknown_name(name, known_gestures))


def _unknown_name(name: str, options: Collection[str]) -> str:
    return (
        f"nome desconhecido {name!r}.{_suggestion(name, options)}"
        f" Nomes conhecidos: {sorted(options)}"
    )
