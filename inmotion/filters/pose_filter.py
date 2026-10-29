"""PoseFilter (spec 006, RF7): aplica um filtro temporal (R1-R4) aos landmarks da pose.

Premissa 1 do plano: filtra os landmarks BRUTOS, antes de ``M = S·R·T`` (A6), para que a
matriz (que depende dos ombros) fique estável. Núcleo puro: sem relógio, o tempo é ``pose.t``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from inmotion.filters import TemporalFilter
from inmotion.types import PoseFrame


class PoseFilter:
    """Dois filtros independentes (criados pela ``factory``): um para ``landmarks[:, 0:3]``
    e outro para ``world[:, 0:3]``, ambos de forma (33, 3).
    """

    def __init__(self, factory: Callable[[], TemporalFilter]) -> None:
        self._image_filter = factory()
        self._world_filter = factory()

    def __call__(self, pose: PoseFrame | None) -> PoseFrame | None:
        """Filtra x, y, z (e world x, y, z); visibility e presence passam inalteradas.

        ``None`` devolve ``None`` sem tocar no estado (quem reseta ao perder o tracking é
        o pipeline). Se a pose vier sem ``world``, o filtro de world é resetado para não
        misturar sessões distintas. Devolve um ``PoseFrame`` novo (``t`` e ``image_size``
        preservados).
        """
        if pose is None:
            return None
        landmarks = np.array(pose.landmarks, dtype=np.float64, copy=True)
        landmarks[:, 0:3] = self._image_filter.update(pose.landmarks[:, 0:3], pose.t)
        world: NDArray[np.float64] | None = None
        if pose.world is None:
            self._world_filter.reset()
        else:
            world = np.array(pose.world, dtype=np.float64, copy=True)
            world[:, 0:3] = self._world_filter.update(pose.world[:, 0:3], pose.t)
        return PoseFrame(t=pose.t, image_size=pose.image_size, landmarks=landmarks, world=world)

    def reset(self) -> None:
        """Esquece o estado dos dois filtros."""
        self._image_filter.reset()
        self._world_filter.reset()
