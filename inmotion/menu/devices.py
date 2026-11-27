"""Dispositivos e perfis para o menu (spec 024): microfones, câmeras e jogos.

As bibliotecas (``sounddevice``, ``cv2``) entram por funções injetáveis, para os testes
usarem dublês sem hardware.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

GAMES_DIR = Path("config/games")
_MIC_HOSTAPI = "MME"  # um nome por microfone; os outros drivers repetem os mesmos aparelhos
_MIC_MAPPERS = (
    "mapeador de som",
    "sound mapper",
)  # o "microfone padrão" do Windows, não um aparelho


@dataclass(frozen=True)
class GameOption:
    """Perfil de jogo: ``name`` vai para ``--game``; ``label`` aparece no menu."""

    name: str
    label: str


def list_game_profiles(games_dir: Path = GAMES_DIR) -> list[GameOption]:
    """Um item por ``<jogo>.yaml``; o rótulo é o ``name`` do arquivo, com inicial maiúscula."""
    options = []
    for path in sorted(games_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        name = data.get("name") if isinstance(data, dict) else None
        label = str(name) if isinstance(name, str) and name else path.stem
        options.append(GameOption(path.stem, label[:1].upper() + label[1:]))
    return options


def list_microphones(
    query_devices: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    query_hostapis: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> list[tuple[int, str]]:
    """``(índice, nome)`` das entradas de áudio do driver MME, sem o mapeador do Windows.

    O índice é o que o ``sounddevice`` usa (``voice.device``).
    """
    if query_devices is None or query_hostapis is None:
        import sounddevice as sd

        query_devices = query_devices or (lambda: list(sd.query_devices()))
        query_hostapis = query_hostapis or (lambda: list(sd.query_hostapis()))
    apis = [str(api["name"]) for api in query_hostapis()]
    mics = []
    for index, dev in enumerate(query_devices()):
        name = str(dev["name"])
        if int(dev["max_input_channels"]) <= 0 or apis[int(dev["hostapi"])] != _MIC_HOSTAPI:
            continue
        if any(mapper in name.lower() for mapper in _MIC_MAPPERS):
            continue
        mics.append((index, name))
    return mics


def _open_dshow(index: int) -> Any:
    import cv2

    return cv2.VideoCapture(index, cv2.CAP_DSHOW)


def probe_cameras(
    open_camera: Callable[[int], Any] | None = None, max_index: int = 5
) -> list[tuple[int, NDArray[np.uint8]]]:
    """``(índice, miniatura BGR)`` dos índices ``0..max_index−1`` que entregam um frame.

    Cada câmera é aberta, lida uma vez e solta logo, para o jogo poder usá-la depois.
    """
    opener = open_camera or _open_dshow
    found: list[tuple[int, NDArray[np.uint8]]] = []
    for index in range(max_index):
        cap = opener(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append((index, np.asarray(frame, dtype=np.uint8)))
        finally:
            cap.release()
    return found
