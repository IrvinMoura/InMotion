"""Features dos gestos de corpo (spec 008): alcance frontal, altura do quadril e base R7.

Núcleo puro: sem relógio; o tempo entra como parâmetro. Todas as features são
adimensionais (divididas pela largura dos ombros ``s``) ou angulares (RNF1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig, NormalizedSkeleton, is_visible
from inmotion.vision.landmarks import PoseLandmark, Side

_ARM: dict[Side, tuple[PoseLandmark, PoseLandmark, PoseLandmark]] = {
    Side.LEFT: (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    Side.RIGHT: (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}
_HIPS = (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)


@dataclass(frozen=True)
class BaselineParams:
    """Parâmetros da base R7: ``K`` frames iniciais e taxa ``λ`` da EMA lenta."""

    init_frames: int = 15
    rate: float = 0.02


class BodyBaseline:
    """R7: ``b ← (1−λ)·b + λ·h`` só em repouso (EMA); a inicial é a média de K frames."""

    def __init__(self, params: BaselineParams) -> None:
        self._params = params

    def update(self, h: float, resting: bool) -> float | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def crouch_depth(ctx: FrameContext) -> float | None:
    raise NotImplementedError


def lean_forward(pose: PoseFrame, geometry: GeometryConfig) -> float | None:
    raise NotImplementedError
