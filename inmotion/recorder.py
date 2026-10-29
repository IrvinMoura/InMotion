"""Adaptador de arquivo das gravações: acumula frames e lê/escreve ``.npz`` (ADR 0008).

Layout do arquivo (um único ``.npz`` comprimido, sem pickle):

- ``t`` (N,), ``landmarks`` (N, 33, 5), ``world`` (N, 33, 4), ``present`` (N,),
  ``image_size`` (2,): arrays NumPy com NaN onde não há pose;
- ``meta_json``: texto JSON com ``schema_version``, ``meta`` (``RecordingMeta``) e
  ``labels`` (lista de ``{gesture, start, end}``).
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from inmotion.recording import (
    SCHEMA_VERSION,
    LabelInterval,
    Recording,
    RecordingFormatError,
    RecordingMeta,
)
from inmotion.types import N_LANDMARKS, PoseFrame

_ARRAY_KEYS = ("t", "landmarks", "world", "present", "image_size")


class Recorder:
    """Acumula frames em memória e salva um ``.npz`` comprimido."""

    def __init__(self, meta: RecordingMeta, image_size: tuple[int, int]) -> None:
        if meta.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"meta.schema_version deve ser {SCHEMA_VERSION}, veio {meta.schema_version}"
            )
        self._meta = meta
        self._image_size = image_size
        self._t: list[float] = []
        self._landmarks: list[Any] = []
        self._world: list[Any] = []
        self._present: list[bool] = []

    def add(self, t: float, pose: PoseFrame | None) -> None:
        """Acrescenta um frame; ``pose=None`` (sem detecção) grava NaN e ``present=False``."""
        if pose is not None and pose.image_size != self._image_size:
            raise ValueError(
                f"image_size da pose {pose.image_size} difere do da gravação {self._image_size}"
            )
        self._t.append(float(t))
        if pose is None:
            self._landmarks.append(np.full((N_LANDMARKS, 5), np.nan))
            self._world.append(np.full((N_LANDMARKS, 4), np.nan))
            self._present.append(False)
            return
        self._landmarks.append(pose.landmarks)
        self._world.append(np.full((N_LANDMARKS, 4), np.nan) if pose.world is None else pose.world)
        self._present.append(True)

    def save(self, path: Path, labels: Sequence[LabelInterval]) -> Path:
        """Valida (via ``Recording``) e grava o ``.npz``; devolve o caminho final.

        Acrescenta o sufixo ``.npz`` se faltar (comportamento do ``numpy``, explicitado).
        """
        n = len(self._t)
        recording = Recording(
            meta=self._meta,
            image_size=self._image_size,
            t=np.array(self._t, dtype=np.float64),
            landmarks=np.array(self._landmarks, dtype=np.float64).reshape(n, N_LANDMARKS, 5),
            world=np.array(self._world, dtype=np.float64).reshape(n, N_LANDMARKS, 4),
            present=np.array(self._present, dtype=np.bool_),
            labels=tuple(labels),
        )
        target = path if path.suffix == ".npz" else path.with_name(path.name + ".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        meta_json = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "meta": asdict(recording.meta),
                "labels": [asdict(lab) for lab in recording.labels],
            },
            ensure_ascii=False,
        )
        np.savez_compressed(
            target,
            t=recording.t,
            landmarks=recording.landmarks,
            world=recording.world,
            present=recording.present,
            image_size=np.array(recording.image_size, dtype=np.int64),
            meta_json=np.array(meta_json),
        )
        return target


def load_recording(path: Path) -> Recording:
    """Lê um ``.npz``; versão incompatível ou arquivo inválido → RecordingFormatError.

    Lê com ``allow_pickle=False``: nenhum objeto Python é desserializado.
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            if "meta_json" not in files:
                raise RecordingFormatError(f"{path}: campo 'meta_json' ausente")
            doc = json.loads(str(data["meta_json"]))
            if not isinstance(doc, dict) or "schema_version" not in doc:
                raise RecordingFormatError(f"{path}: 'schema_version' ausente em meta_json")
            version = doc["schema_version"]
            if version != SCHEMA_VERSION:
                raise RecordingFormatError(
                    f"{path}: schema_version {version} incompatível (suportada: {SCHEMA_VERSION})"
                )
            missing = [key for key in _ARRAY_KEYS if key not in files]
            if missing:
                raise RecordingFormatError(f"{path}: campos ausentes: {', '.join(missing)}")
            arrays = {key: np.array(data[key]) for key in _ARRAY_KEYS}
        width, height = (int(v) for v in arrays["image_size"])
        return Recording(
            meta=RecordingMeta(**doc["meta"]),
            image_size=(width, height),
            t=arrays["t"],
            landmarks=arrays["landmarks"],
            world=arrays["world"],
            present=arrays["present"],
            labels=tuple(LabelInterval(**lab) for lab in doc["labels"]),
        )
    except RecordingFormatError:
        raise
    except FileNotFoundError:
        raise
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, KeyError, TypeError) as exc:
        raise RecordingFormatError(f"{path}: arquivo de gravação inválido ({exc})") from exc
