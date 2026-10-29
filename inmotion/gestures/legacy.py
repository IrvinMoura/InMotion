"""Gestos do protótipo (``actions/``) portados para ``PoseFrame``. Núcleo puro, temporário.

O comportamento é o do legado, sem melhorias: a spec 008 substitui este módulo. As
razões são calculadas em **pixels** (A5: p = diag(W, H)·n), como no original, porque
a razão 2D de comprimentos muda com W ≠ H quando feita em coordenadas normalizadas.
"""

from __future__ import annotations

import math
from typing import Protocol

from inmotion.input.backend import KeyName
from inmotion.safety import TrackedInput
from inmotion.types import PoseFrame
from inmotion.vision.landmarks import PoseLandmark, Side


class LegacyGesture(Protocol):
    """Gesto do protótipo: lê a pose e aperta/solta teclas via ``TrackedInput``."""

    def update(self, pose: PoseFrame, out: TrackedInput) -> None: ...

    def reset(self) -> None: ...


_ARM = {
    Side.RIGHT: (
        PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW,
        PoseLandmark.RIGHT_WRIST,
        PoseLandmark.RIGHT_HIP,
    ),
    Side.LEFT: (
        PoseLandmark.LEFT_SHOULDER,
        PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST,
        PoseLandmark.LEFT_HIP,
    ),
}


class LegacyPunch:
    """Soco frontal (``actions/atacar.py`` e ``actions/usarItem.py``).

    Razão ``ρ = ‖e − w‖ / ‖s − e‖`` em pixels (A5); ativo quando ``ρ < ratio_threshold``
    e o pulso está acima do quadril (``y_w < y_h``, y cresce para baixo). Se
    ``‖s − e‖ = 0``, usa 1 no denominador (como o original).
    """

    def __init__(self, side: Side, key: KeyName, ratio_threshold: float = 0.60) -> None:
        if not (math.isfinite(ratio_threshold) and ratio_threshold > 0):
            raise ValueError(f"ratio_threshold deve ser finito e > 0: {ratio_threshold}")
        if not key:
            raise ValueError("key não pode ser vazio")
        self._side = side
        self._key = key
        self._threshold = ratio_threshold
        self._active = False

    def update(self, pose: PoseFrame, out: TrackedInput) -> None:
        px = pose.pixels()
        s, e, w, h = (px[int(i)] for i in _ARM[self._side])
        upper = math.hypot(float(s[0] - e[0]), float(s[1] - e[1]))
        if upper == 0:
            upper = 1.0
        ratio = math.hypot(float(e[0] - w[0]), float(e[1] - w[1])) / upper
        if bool(w[1] < h[1]) and ratio < self._threshold:
            if not self._active:
                out.press(self._key)
                self._active = True
        elif self._active:
            out.release(self._key)
            self._active = False

    def reset(self) -> None:
        """Esquece o estado (as teclas já foram soltas pelo ``SafetyManager``)."""
        self._active = False


class LegacyJump:
    """Pulo (``actions/pular.py``), com limiares em fração da altura ``H`` da imagem.

    Base ``b`` = y do centro dos quadris no primeiro frame. Pulo quando
    ``y < b − rise·H`` (y cresce para baixo). Ao soltar, se ``|y − b| < settle·H``,
    recalibra ``b = y`` (contra deriva). Equivale ao legado (50 px, 20 px) em 720 p.
    """

    def __init__(
        self, key: KeyName = "space", rise: float = 50 / 720, settle: float = 20 / 720
    ) -> None:
        for name, value in (("rise", rise), ("settle", settle)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} deve ser finito e > 0: {value}")
        if not key:
            raise ValueError("key não pode ser vazio")
        self._key = key
        self._rise = rise
        self._settle = settle
        self._base: float | None = None
        self._active = False

    def update(self, pose: PoseFrame, out: TrackedInput) -> None:
        px = pose.pixels()
        y = float((px[PoseLandmark.LEFT_HIP][1] + px[PoseLandmark.RIGHT_HIP][1]) / 2)
        height = pose.image_size[1]
        if self._base is None:
            self._base = y
            return
        if y < self._base - self._rise * height:
            if not self._active:
                out.press(self._key)
                self._active = True
        elif self._active:
            out.release(self._key)
            self._active = False
            if abs(y - self._base) < self._settle * height:
                self._base = y

    def reset(self) -> None:
        """Esquece a base e o estado: o próximo frame recalibra."""
        self._base = None
        self._active = False
