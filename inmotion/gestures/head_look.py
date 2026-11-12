"""Olhar com a cabeça (spec 014): orientação pelos pontos do rosto e câmera por velocidade.

Converte o deslocamento normalizado do nariz (adimensional: deslocamento / distância
entre as orelhas) em velocidade do ponteiro (px/s), com zona morta, expoente e limite
(C1-C4), e integra ``v·Δt`` com resto fracionário (C5). Núcleo puro: o tempo é ``t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig


@dataclass(frozen=True)
class ResponseCurve:
    """Curva ``v = sign(x)·min(k·max(0, |x| − d)^gamma, v_max)``.

    Propriedades: C1 ímpar, ``v(−x) = −v(x)``; C2 ``v = 0`` em ``|x| ≤ d`` e contínua em
    ``|x| = d``; C3 ``|v|`` não decrescente em ``|x|``; C4 ``|v| ≤ v_max``.

    Entradas não finitas: ``NaN`` vira ``0.0`` (dado inválido nunca move a câmera) e
    ``±inf`` vira ``±v_max``.
    """

    gain: float = 1500.0
    dead_zone: float = 0.08
    gamma: float = 1.5
    v_max: float = 1200.0

    def __post_init__(self) -> None:
        """Valida: ganho > 0, zona morta ≥ 0, gamma ≥ 1, v_max > 0, todos finitos."""
        values = (self.gain, self.dead_zone, self.gamma, self.v_max)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("parâmetros da curva devem ser finitos")
        if self.gain <= 0 or self.v_max <= 0:
            raise ValueError("gain e v_max devem ser > 0")
        if self.dead_zone < 0:
            raise ValueError("dead_zone deve ser >= 0")
        if self.gamma < 1:
            raise ValueError("gamma deve ser >= 1")

    def __call__(self, x: float) -> float:
        """Velocidade (px/s) para o deslocamento normalizado ``x``."""
        if math.isnan(x):
            return 0.0
        mag = abs(x) - self.dead_zone
        if mag <= 0.0:
            return 0.0
        try:
            speed = min(self.gain * mag**self.gamma, self.v_max)
        except OverflowError:
            speed = self.v_max
        return math.copysign(speed, x)


@dataclass(frozen=True)
class HeadPose:
    """Orientação da cabeça, adimensional (deslocamento do nariz / distância das orelhas).

    ``yaw > 0``: cabeça virada para a direita da pessoa. ``pitch > 0``: para cima.
    """

    yaw: float
    pitch: float


def head_pose(pose: PoseFrame, geometry: GeometryConfig, min_ear_px: float) -> HeadPose | None:
    """RF1: ``yaw = −(x_nariz − x̄_orelhas)/e``, ``pitch = (ȳ_olhos − y_nariz)/e``, ``e`` =
    distância entre as orelhas em pixels; ``None`` sem os pontos visíveis ou com ``e`` pequeno.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class HeadLookParams:
    """Parâmetros da câmera (spec 014 T0): curvas por eixo, neutro e pitch."""

    neutral_s: float = 1.0
    min_ear_px: float = 5.0
    yaw: ResponseCurve = ResponseCurve(gain=1200.0, dead_zone=0.15, gamma=1.5, v_max=800.0)
    pitch: ResponseCurve = ResponseCurve(gain=1200.0, dead_zone=0.15, gamma=1.5, v_max=800.0)
    pitch_enabled: bool = False
    invert_pitch: bool = False


class HeadLook:
    """Câmera em controle de velocidade: neutro automático (RF2), curva (RF3), C5."""

    def __init__(self, params: HeadLookParams) -> None:
        raise NotImplementedError

    def update(self, head: HeadPose | None, t: float, *, suppress: bool = False) -> tuple[int, int]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
