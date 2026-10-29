"""Formato de gravação de sessões de landmarks rotuladas (spec 011, ADR 0008).

Núcleo puro: sem arquivo, sem relógio. A hora de criação (``created``) vem do chamador.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray

from inmotion.input.backend import KeyName
from inmotion.types import N_LANDMARKS, PoseFrame

SCHEMA_VERSION: Final = 1


def _frozen(name: str, value: object, dtype: type[np.float64] | type[np.bool_]) -> NDArray[Any]:
    """Copia ``value`` para ``dtype`` e marca como somente leitura."""
    try:
        arr = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} inválido: {exc}") from exc
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True)
class RecordingMeta:
    """Metadados de uma sessão (pessoa anônima, condições de coleta, hash da config)."""

    schema_version: int
    created: str
    person: str
    distance_m: float | None
    lighting: Literal["clara", "escura", "outra"]
    camera: str
    config_hash: str


@dataclass(frozen=True)
class LabelInterval:
    """Intervalo [start, end] em que o gesto ``gesture`` estava sendo executado."""

    gesture: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.start) and math.isfinite(self.end)):
            raise ValueError("rótulo com início/fim não finito")
        if not self.start < self.end:
            raise ValueError(
                f"rótulo {self.gesture!r} exige start < end, veio [{self.start}, {self.end}]"
            )


class RecordingFormatError(ValueError):
    """Arquivo de gravação inválido, corrompido ou de versão incompatível."""


@dataclass(frozen=True, eq=False)
class Recording:
    """Sessão gravada. Arrays com ``N`` frames; NaN onde não há pose."""

    meta: RecordingMeta
    image_size: tuple[int, int]
    t: NDArray[np.float64]
    landmarks: NDArray[np.float64]
    world: NDArray[np.float64]
    present: NDArray[np.bool_]
    labels: tuple[LabelInterval, ...]

    def __post_init__(self) -> None:
        w, h = self.image_size
        if len(self.image_size) != 2 or not all(
            isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (w, h)
        ):
            raise ValueError(f"image_size deve ser (W, H) de inteiros > 0, veio {self.image_size}")
        t = _frozen("t", self.t, np.float64)
        if t.ndim != 1:
            raise ValueError(f"t deve ter forma (N,), veio {t.shape}")
        n = t.shape[0]
        landmarks = _frozen("landmarks", self.landmarks, np.float64)
        world = _frozen("world", self.world, np.float64)
        present = _frozen("present", self.present, np.bool_)
        if landmarks.shape != (n, N_LANDMARKS, 5):
            raise ValueError(
                f"landmarks deve ter forma ({n}, {N_LANDMARKS}, 5), veio {landmarks.shape}"
            )
        if world.shape != (n, N_LANDMARKS, 4):
            raise ValueError(f"world deve ter forma ({n}, {N_LANDMARKS}, 4), veio {world.shape}")
        if present.shape != (n,):
            raise ValueError(f"present deve ter forma ({n},), veio {present.shape}")
        if not np.isfinite(t).all():
            raise ValueError("t deve ser finito")
        if n > 1 and not (np.diff(t) > 0.0).all():
            raise ValueError("t deve ser estritamente crescente")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "landmarks", landmarks)
        object.__setattr__(self, "world", world)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "labels", tuple(self.labels))

    def poses(self) -> Iterator[tuple[float, PoseFrame | None]]:
        """Itera ``(t, PoseFrame)``; ``None`` nos frames sem pose (``present`` falso)."""
        for i in range(self.t.shape[0]):
            t = float(self.t[i])
            if not self.present[i]:
                yield t, None
                continue
            yield (
                t,
                PoseFrame(
                    t=t,
                    image_size=self.image_size,
                    landmarks=self.landmarks[i],
                    world=None if np.isnan(self.world[i]).all() else self.world[i],
                ),
            )

    @property
    def duration_s(self) -> float:
        """Duração D = t_{N−1} − t_0 (0 se há menos de 2 frames)."""
        if self.t.shape[0] < 2:
            return 0.0
        return float(self.t[-1] - self.t[0])


class LabelTracker:
    """Transforma "teclas de rótulo seguradas por frame" em intervalos.

    Uma tecla mapeada que aparece em ``keys`` abre um intervalo no instante ``t``; ao
    deixar de aparecer, fecha em ``t``. Teclas fora do mapa são ignoradas. Intervalos de
    duração zero são descartados (``start < end``).
    """

    def __init__(self, key_to_gesture: Mapping[KeyName, str]) -> None:
        self._map = dict(key_to_gesture)
        self._open: dict[KeyName, float] = {}
        self._done: list[LabelInterval] = []
        self._last_t = -math.inf

    def update(self, keys: frozenset[KeyName], t: float) -> None:
        """Registra as teclas seguradas no instante ``t`` (não decrescente)."""
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError(f"t deve ser finito e não decrescente, veio {t}")
        self._last_t = t
        for key, gesture in self._map.items():
            held = key in keys
            if held and key not in self._open:
                self._open[key] = t
            elif not held and key in self._open:
                self._finish(key, gesture, t)

    def close(self, t: float) -> tuple[LabelInterval, ...]:
        """Fecha as teclas ainda seguradas em ``t`` e devolve todos os intervalos por início."""
        if not math.isfinite(t) or t < self._last_t:
            raise ValueError(f"t deve ser finito e não decrescente, veio {t}")
        self._last_t = t
        for key in list(self._open):
            self._finish(key, self._map[key], t)
        return tuple(sorted(self._done, key=lambda lab: (lab.start, lab.gesture)))

    def _finish(self, key: KeyName, gesture: str, t: float) -> None:
        start = self._open.pop(key)
        if start < t:
            self._done.append(LabelInterval(gesture, start, t))
