"""Features dos gestos de corpo (spec 008): alcance frontal, altura do quadril e base R7.

Núcleo puro: sem relógio; o tempo entra como parâmetro. Todas as features são
adimensionais (divididas pela largura dos ombros ``s``) ou angulares (RNF1).
"""

from __future__ import annotations

from dataclasses import dataclass

from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig, NormalizedSkeleton
from inmotion.vision.landmarks import Side


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
    raise NotImplementedError


def jump_height(ctx: FrameContext) -> float | None:
    raise NotImplementedError


def crouch_depth(ctx: FrameContext) -> float | None:
    raise NotImplementedError


def lean_forward(pose: PoseFrame, geometry: GeometryConfig) -> float | None:
    raise NotImplementedError
