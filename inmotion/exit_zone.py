"""Zona de saída temporária: segurar a mão num canto da imagem encerra o app. Núcleo puro.

Existe porque, com os gestos clicando o mouse, fica difícil chegar ao terminal. Um pulso
visível dentro do retângulo (coordenadas normalizadas da imagem, sem espelhamento) por
``hold_s`` segundos seguidos dispara a saída; sair do retângulo zera a contagem.
Provisória até a spec 004/009 (config em YAML e overlay completo).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from inmotion.types import PoseFrame
from inmotion.vision.landmarks import PoseLandmark

_WRISTS = (PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST)
_X, _Y, _VISIBILITY = 0, 1, 3


@dataclass(frozen=True)
class ExitZoneConfig:
    """Retângulo ``[x0, x1] × [y0, y1]`` em coordenadas normalizadas e tempo de espera.

    O padrão fica no canto superior esquerdo da janela, logo abaixo do texto de FPS.
    Como a janela não é espelhada (ADR 0007), esse canto fica do lado direito da pessoa.
    """

    x0: float = 0.02
    y0: float = 0.12
    x1: float = 0.22
    y1: float = 0.42
    hold_s: float = 4.0
    min_visibility: float = 0.5

    def __post_init__(self) -> None:
        coords = (self.x0, self.y0, self.x1, self.y1)
        if not (
            all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in coords)
            and self.x0 < self.x1
            and self.y0 < self.y1
        ):
            raise ValueError(f"retângulo inválido (x0<x1, y0<y1, em [0, 1]): {coords}")
        if not (math.isfinite(self.hold_s) and self.hold_s > 0):
            raise ValueError(f"hold_s deve ser finito e > 0: {self.hold_s}")
        if not (math.isfinite(self.min_visibility) and 0.0 <= self.min_visibility <= 1.0):
            raise ValueError(f"min_visibility deve estar em [0, 1]: {self.min_visibility}")


@dataclass(frozen=True)
class ExitZoneState:
    """Leitura da zona num frame: há mão dentro, progresso em [0, 1] e se disparou."""

    config: ExitZoneConfig
    inside: bool
    progress: float
    triggered: bool


class ExitZone:
    """Cronômetro da mão parada na zona; o tempo chega como parâmetro (``t``)."""

    def __init__(self, config: ExitZoneConfig) -> None:
        self._config = config
        self._since: float | None = None

    def _hand_inside(self, pose: PoseFrame | None) -> bool:
        if pose is None:
            return False
        cfg = self._config
        for wrist in _WRISTS:
            x, y, visibility = pose.landmarks[wrist, [_X, _Y, _VISIBILITY]]
            if visibility >= cfg.min_visibility and cfg.x0 <= x <= cfg.x1 and cfg.y0 <= y <= cfg.y1:
                return True
        return False

    def update(self, pose: PoseFrame | None, t: float) -> ExitZoneState:
        """Atualiza com a pose do frame (``None`` = sem pose, conta como fora da zona).

        progresso = min((t − t_entrada) / hold_s, 1); dispara quando chega a 1.
        """
        cfg = self._config
        if not self._hand_inside(pose):
            self._since = None
            return ExitZoneState(cfg, inside=False, progress=0.0, triggered=False)
        if self._since is None:
            self._since = t
        progress = min((t - self._since) / cfg.hold_s, 1.0)
        return ExitZoneState(cfg, inside=True, progress=progress, triggered=progress >= 1.0)
