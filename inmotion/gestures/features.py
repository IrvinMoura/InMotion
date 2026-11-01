"""Features dos gestos de corpo (spec 008): alcance frontal, altura do quadril e base R7.

Núcleo puro: sem relógio; o tempo entra como parâmetro. Todas as features são
adimensionais (divididas pela largura dos ombros ``s``) ou angulares (RNF1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
if TYPE_CHECKING:
    from inmotion.gestures.march import MarchState

_HIPS = (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)
_SHOULDERS = (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)


@dataclass(frozen=True)
class BaselineParams:
    """Parâmetros da base R7: ``K`` frames iniciais, taxa ``λ`` da EMA lenta (por frame) e
    ``rest_band``: desvio máximo ``|h − b|/s`` (larguras de ombro) para a base andar."""

    init_frames: int = 15
    rate: float = 0.02
    rest_band: float = 0.15

    def __post_init__(self) -> None:
        if self.init_frames < 1:
            raise ValueError("init_frames deve ser >= 1")
        if not (math.isfinite(self.rate) and 0.0 < self.rate <= 1.0):
            raise ValueError("rate deve estar em (0, 1]")
        if not (math.isfinite(self.rest_band) and self.rest_band > 0.0):
            raise ValueError("rest_band deve ser finito e > 0")


class BodyBaseline:
    """R7: base adaptativa de uma altura, ``b_n = (1−λ)·b_{n−1} + λ·h_n`` em repouso.

    Serve a qualquer altura em pixels: o quadril (pulo) e, na spec 017, os ombros (agachar).

    - A base inicial é a média de ``K`` amostras de repouso, nunca um frame único (RF3).
    - Fora do repouso (algum gesto de corpo fora de IDLE) a base fica congelada (J2).
    - É uma EMA sobre os frames de repouso, logo valem R2 (``b`` fica entre o mínimo e o
      máximo das amostras) e R3 (resposta a degrau com constante de tempo ``1/λ`` frames).
    - J3: uma deriva lenta de ``h`` (velocidade ``v`` por frame) é seguida com defasagem
      estacionária ``v/λ``, abaixo do limiar do pulo se ``v/λ ≪ θ·s``.
    - Portão de desvio (spec 017 T7): com a escala ``s`` informada, a amostra só entra se
      ``|h − b|/s < rest_band``. A descida para agachar ainda conta como repouso (``c`` <
      θ_on); sem o portão a base seguia o quadril e, de pé de novo, ``j`` passava do
      limiar do pulo (sessões guiadas de 01/11/2026). Com o portão, a deriva lenta (J3)
      continua absorvida, pois ``v/λ`` fica abaixo de ``rest_band·s``.
    """

    def __init__(self, params: BaselineParams) -> None:
        self._params = params
        self._samples: list[float] = []
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Base atual (px) ou ``None`` até acumular ``K`` amostras."""
        return self._value

    def update(self, h: float, resting: bool, scale: float | None = None) -> float | None:
        """Incorpora ``h`` (px) se ``resting``, finito e perto da base; devolve a base.

        ``scale`` é a largura dos ombros ``s`` (px) do frame; sem ela, não há portão de
        desvio. Devolve ``None`` enquanto houver menos de ``K`` amostras."""
        if not resting or not math.isfinite(h):
            return self._value
        if (
            self._value is not None
            and scale is not None
            and math.isfinite(scale)
            and scale > 0.0
            and abs(h - self._value) / scale >= self._params.rest_band
        ):
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
    # Spec 017: ombros medidos direto em pixels, sem exigir os quadris (agachar pelos ombros).
    shoulder_y_px: float | None = None
    shoulder_width_px: float | None = None
    shoulder_baseline_px: float | None = None
    # Spec 018: distância vertical joelho-quadril de cada lado (px) e sua base R7 (px).
    knee_dist_left_px: float | None = None
    knee_dist_right_px: float | None = None
    knee_base_left_px: float | None = None
    knee_base_right_px: float | None = None
    # Estado da marcha do frame (preenchido pelo pipeline, uma vez por frame, antes dos gestos).
    march: MarchState | None = None


def _shoulder_line(pose: PoseFrame, geometry: GeometryConfig) -> tuple[float, float] | None:
    """``(ȳ, s)`` dos ombros em pixels, sem exigir os quadris (spec 017 RF3).

    ``ȳ = (y₁₁ + y₁₂)/2`` e ``s = ‖p₁₁ − p₁₂‖`` (A6, a mesma escala do esqueleto). ``None`` se
    algum ombro está abaixo do limiar de visibilidade, se ``s < min_shoulder_px`` ou se os
    valores não são finitos.
    """
    if not is_visible(pose, _SHOULDERS, geometry.visibility_threshold):
        return None
    px = pose.pixels()
    left, right = px[_SHOULDERS[0]], px[_SHOULDERS[1]]
    s = float(np.hypot(left[0] - right[0], left[1] - right[1]))
    y = float((left[1] + right[1]) / 2.0)
    if not (math.isfinite(s) and math.isfinite(y)) or s < geometry.min_shoulder_px:
        return None
    return y, s


def build_context(
    pose: PoseFrame,
    t: float,
    geometry: GeometryConfig,
    baseline: BodyBaseline,
    resting: bool,
    shoulder_baseline: BodyBaseline | None = None,
    knee_baselines: tuple[BodyBaseline, BodyBaseline] | None = None,
) -> FrameContext:
    """Monta o contexto do frame: esqueleto normalizado (A6), ``s``, ``h`` e as bases R7.

    ``h`` é a ordenada do centro do quadril em pixels (y cresce para baixo) e ``s`` a
    largura dos ombros em pixels. A base do quadril só é atualizada se houver esqueleto e
    ``resting``. Sem esqueleto (visibilidade ou ``s`` degenerado), ``shoulder_px`` e
    ``hip_y_px`` são ``None`` e a base do quadril não muda.

    Spec 017: os ombros também são medidos direto em pixels (``shoulder_y_px``,
    ``shoulder_width_px``), **sem exigir os quadris**, e uma segunda base R7
    (``shoulder_baseline``, opcional) acompanha ``ȳ_ombros`` com a mesma regra de repouso:
    só anda com ``resting`` e ombros válidos, e fica congelada fora disso.
    """
    shoulders = _shoulder_line(pose, geometry)
    shoulder_y, shoulder_w = shoulders if shoulders is not None else (None, None)
    shoulder_base: float | None = None
    if shoulder_baseline is not None:
        shoulder_base = (
            shoulder_baseline.update(shoulder_y, resting, shoulder_w)
            if shoulder_y is not None
            else shoulder_baseline.value
        )
    skeleton = normalize_skeleton(pose, geometry)
    if skeleton is None:
        return FrameContext(
            t, pose, None, None, None, baseline.value, shoulder_y, shoulder_w, shoulder_base
        )
    px = pose.pixels()
    hip_y = float((px[_HIPS[0], 1] + px[_HIPS[1], 1]) / 2.0)
    base = baseline.update(hip_y, resting, skeleton.shoulder_px)
    return FrameContext(
        t, pose, skeleton, skeleton.shoulder_px, hip_y, base, shoulder_y, shoulder_w, shoulder_base
    )


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


def crouch_drop(ctx: FrameContext) -> float | None:
    """RF3: ``c = (ȳ_ombros − b_o)/s``, adimensional; positivo quando os ombros descem.

    ``ȳ_ombros = (y₁₁ + y₁₂)/2`` e ``s = ‖p₁₁ − p₁₂‖`` são medidos em pixels (A6, a mesma
    escala do esqueleto, mas sem exigir os quadris) e ``b_o`` é a base R7 dessa altura.
    Invariante a translação e escala: ``ȳ`` e ``b_o`` são ordenadas na mesma imagem, então
    uma translação vertical ``d`` cancela em ``ȳ − b_o`` e uma escala ``k`` multiplica
    numerador e denominador. Vale ≈ 0 na base. ``None`` sem ombros válidos ou sem base.
    """
    y, base, s = ctx.shoulder_y_px, ctx.shoulder_baseline_px, ctx.shoulder_width_px
    if y is None or base is None or s is None:
        return None
    return (y - base) / s


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


def knee_lift(ctx: FrameContext, side: Side) -> float | None:
    """RF1 (018): ``k = d − (ȳ_joelho − ȳ_quadril)/s``; ≈ 0 parado, > 0 com o joelho erguido."""
    raise NotImplementedError
