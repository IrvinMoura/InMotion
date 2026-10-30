"""Features dos gestos de corpo (spec 008): alcance frontal, altura do quadril e base R7.

Núcleo puro: sem relógio; o tempo entra como parâmetro. Todas as features são
adimensionais (divididas pela largura dos ombros ``s``) ou angulares (RNF1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inmotion.types import PoseFrame
from inmotion.vision.geometry import (
    GeometryConfig,
    NormalizedSkeleton,
    is_visible,
    normalize_skeleton,
)
from inmotion.vision.landmarks import PoseLandmark, Side

_ARM: dict[Side, tuple[PoseLandmark, PoseLandmark, PoseLandmark]] = {
    Side.LEFT: (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    Side.RIGHT: (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}
_HIPS = (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)
_SHOULDERS = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)


@dataclass(frozen=True)
class BaselineParams:
    """Parâmetros da base R7: ``K`` frames iniciais e taxa ``λ`` da EMA lenta (por frame)."""

    init_frames: int = 15
    rate: float = 0.02

    def __post_init__(self) -> None:
        if self.init_frames < 1:
            raise ValueError("init_frames deve ser >= 1")
        if not (math.isfinite(self.rate) and 0.0 < self.rate <= 1.0):
            raise ValueError("rate deve estar em (0, 1]")


class BodyBaseline:
    """R7: base adaptativa da altura do quadril, ``b_n = (1−λ)·b_{n−1} + λ·h_n`` em repouso.

    - A base inicial é a média de ``K`` amostras de repouso, nunca um frame único (RF3).
    - Fora do repouso (algum gesto de corpo fora de IDLE) a base fica congelada (J2).
    - É uma EMA sobre os frames de repouso, logo valem R2 (``b`` fica entre o mínimo e o
      máximo das amostras) e R3 (resposta a degrau com constante de tempo ``1/λ`` frames).
    - J3: uma deriva lenta de ``h`` (velocidade ``v`` por frame) é seguida com defasagem
      estacionária ``v/λ``, abaixo do limiar do pulo se ``v/λ ≪ θ·s``.
    """

    def __init__(self, params: BaselineParams) -> None:
        self._params = params
        self._samples: list[float] = []
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Base atual (px) ou ``None`` até acumular ``K`` amostras."""
        return self._value

    def update(self, h: float, resting: bool) -> float | None:
        """Incorpora ``h`` (px) se ``resting`` e finito; devolve a base ou ``None`` (< K)."""
        if not resting or not math.isfinite(h):
            return self._value
        if self._value is None:
            self._samples.append(h)
            if len(self._samples) >= self._params.init_frames:
                self._value = sum(self._samples) / len(self._samples)
                self._samples.clear()
        else:
            lam = self._params.rate
            self._value = (1.0 - lam) * self._value + lam * h
        return self._value

    def reset(self) -> None:
        """Esquece a base e as amostras (volta a precisar de ``K`` frames de repouso)."""
        self._samples.clear()
        self._value = None


@dataclass(frozen=True)
class FrameContext:
    """Tudo o que os gestos de corpo precisam de um frame, calculado uma vez."""

    t: float
    pose: PoseFrame
    skeleton: NormalizedSkeleton | None
    shoulder_px: float | None
    hip_y_px: float | None
    baseline_px: float | None


def build_context(
    pose: PoseFrame,
    t: float,
    geometry: GeometryConfig,
    baseline: BodyBaseline,
    resting: bool,
) -> FrameContext:
    """Monta o contexto do frame: esqueleto normalizado (A6), ``s``, ``h`` e a base R7.

    ``h`` é a ordenada do centro do quadril em pixels (y cresce para baixo) e ``s`` a
    largura dos ombros em pixels. A base só é atualizada se houver esqueleto e ``resting``.
    Sem esqueleto (visibilidade ou ``s`` degenerado), ``shoulder_px`` e ``hip_y_px`` são
    ``None`` e a base não muda.
    """
    skeleton = normalize_skeleton(pose, geometry)
    if skeleton is None:
        return FrameContext(t, pose, None, None, None, baseline.value)
    px = pose.pixels()
    hip_y = float((px[_HIPS[0], 1] + px[_HIPS[1], 1]) / 2.0)
    base = baseline.update(hip_y, resting)
    return FrameContext(t, pose, skeleton, skeleton.shoulder_px, hip_y, base)


def forward_reach(pose: PoseFrame, side: Side, geometry: GeometryConfig) -> float | None:
    """A7/A8: alcance frontal em world, ``r = (z_ombro − z_pulso) / L``.

    ``L = ‖ombro − cotovelo‖ + ‖cotovelo − pulso‖`` (comprimento do braço).
    Calculado nos world landmarks (metros; ``z`` negativo = em direção à câmera).

    Propriedade ``|r| ≤ 1``: pela desigualdade triangular,
    ``‖ombro − pulso‖ ≤ ‖ombro − cotovelo‖ + ‖cotovelo − pulso‖`` e
    ``|z_ombro − z_pulso| ≤ ‖ombro − pulso‖``; logo ``|r| ≤ 1``. A igualdade (``r = 1``)
    vale se e somente se o cotovelo está no segmento ombro-pulso (braço reto) e esse
    segmento é paralelo ao eixo ``z`` apontando para a câmera. É adimensional (J1).

    Gate: com o pulso abaixo do centro do quadril (``y`` do world cresce para baixo) a
    feature vale ``0.0``. Devolve ``None`` sem world, com ombro/cotovelo/pulso/quadris
    abaixo do limiar de visibilidade, braço degenerado (comprimento 0) ou valores não finitos.
    """
    world = pose.world
    if world is None:
        return None
    sh, el, wr = _ARM[side]
    if not is_visible(pose, (sh, el, wr, *_HIPS), geometry.visibility_threshold):
        return None
    p_sh, p_el, p_wr = world[sh, :3], world[el, :3], world[wr, :3]
    length = float(np.linalg.norm(p_sh - p_el) + np.linalg.norm(p_el - p_wr))
    hip_y = float((world[_HIPS[0], 1] + world[_HIPS[1], 1]) / 2.0)
    if not (math.isfinite(length) and math.isfinite(hip_y)) or length == 0.0:
        return None
    if p_wr[1] > hip_y:
        return 0.0
    r = float(p_sh[2] - p_wr[2]) / length
    return r if math.isfinite(r) else None


def jump_height(ctx: FrameContext) -> float | None:
    """J1: ``j = (b − h)/s``, adimensional; positivo quando o quadril sobe acima da base.

    Invariante à escala (distância até a câmera) por ser dividida por ``s``.
    ``None`` sem esqueleto ou sem base (R7 ainda sem K amostras).
    """
    if ctx.baseline_px is None or ctx.hip_y_px is None or ctx.shoulder_px is None:
        return None
    return (ctx.baseline_px - ctx.hip_y_px) / ctx.shoulder_px


def crouch_depth(ctx: FrameContext) -> float | None:
    """RF4: ``(h − b)/s``, adimensional; positivo quando o quadril desce abaixo da base."""
    j = jump_height(ctx)
    return None if j is None else -j


def lean_forward(pose: PoseFrame, geometry: GeometryConfig) -> float | None:
    """A9 em profundidade: ``atan2(z_quadril − z_ombros, |y_quadril − y_ombros|)`` em world.

    Centros dos ombros e do quadril nos world landmarks; em rad, 0 = ereto. Positivo =
    ombros mais perto da câmera que o quadril (``z`` menor), isto é, tronco inclinado para
    a frente. ``None`` sem world, com ombros/quadris invisíveis ou tronco degenerado.
    """
    world = pose.world
    if world is None:
        return None
    if not is_visible(pose, (*_SHOULDERS, *_HIPS), geometry.visibility_threshold):
        return None
    sh = (world[_SHOULDERS[0], :3] + world[_SHOULDERS[1], :3]) / 2.0
    hip = (world[_HIPS[0], :3] + world[_HIPS[1], :3]) / 2.0
    dz, dy = float(hip[2] - sh[2]), abs(float(hip[1] - sh[1]))
    if not (math.isfinite(dz) and math.isfinite(dy)) or (dz == 0.0 and dy == 0.0):
        return None
    return math.atan2(dz, dy)
