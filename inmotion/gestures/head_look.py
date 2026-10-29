"""Olhar com a cabeça: curva de resposta não linear (C1-C4) da spec 014.

Converte o deslocamento normalizado do nariz (adimensional: deslocamento / distância
entre as orelhas) em velocidade do ponteiro (px/s), com zona morta, expoente e limite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


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
