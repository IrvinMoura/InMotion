"""Olhar com a cabeça (spec 014): orientação pelos pontos do rosto e câmera por velocidade.

Converte o deslocamento normalizado do nariz (adimensional: deslocamento / distância
entre as orelhas) em velocidade do ponteiro (px/s), com zona morta, expoente e limite
(C1-C4), e integra ``v·Δt`` com resto fracionário (C5). Núcleo puro: o tempo é ``t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig, is_visible, to_pixels
from inmotion.vision.landmarks import PoseLandmark


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


_FACE = (
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
)


def head_pose(pose: PoseFrame, geometry: GeometryConfig, min_ear_px: float) -> HeadPose | None:
    """RF1: orientação da cabeça pelos pontos do rosto, em pixels (A5: ``p = diag(W, H, 1)·n``).

    - ``yaw = −(x_nariz − x̄_orelhas) / e``: o sinal é trocado porque, na imagem não
      espelhada (ADR 0007), virar para a direita da pessoa leva o nariz para a esquerda;
    - ``pitch = (ȳ_olhos − y_nariz) / e`` (o ``y`` da imagem cresce para baixo);
    - ``e = ‖orelha_e − orelha_d‖₂`` (A8).

    C6: as duas razões são adimensionais, invariantes a translação e escala na imagem
    (distância à câmera). ``None`` se algum ponto do rosto está abaixo de
    ``geometry.visibility_threshold`` ou se ``e < min_ear_px`` (de perfil, as orelhas se
    sobrepõem e a razão explodiria).
    """
    if not is_visible(pose, _FACE, geometry.visibility_threshold):
        return None
    w, h = pose.image_size
    px = to_pixels(pose.landmarks[list(_FACE), 0:2], w, h)
    nose, left_eye, right_eye, left_ear, right_ear = px
    ears = float(np.linalg.norm(left_ear - right_ear))
    if not (math.isfinite(ears) and ears >= min_ear_px):
        return None
    yaw = -(nose[0] - (left_ear[0] + right_ear[0]) / 2.0) / ears
    pitch = ((left_eye[1] + right_eye[1]) / 2.0 - nose[1]) / ears
    return HeadPose(yaw=float(yaw), pitch=float(pitch))


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
