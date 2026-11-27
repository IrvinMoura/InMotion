"""Dispositivos e perfis para o menu (spec 024): microfones, câmeras e jogos."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

GAMES_DIR = Path("config/games")


@dataclass(frozen=True)
class GameOption:
    """Perfil de jogo: ``name`` vai para ``--game``; ``label`` aparece no menu."""

    name: str
    label: str


def list_game_profiles(games_dir: Path = GAMES_DIR) -> list[GameOption]:
    raise NotImplementedError


def list_microphones(
    query_devices: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    query_hostapis: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
) -> list[tuple[int, str]]:
    raise NotImplementedError


def probe_cameras(
    open_camera: Callable[[int], Any] | None = None, max_index: int = 5
) -> list[tuple[int, NDArray[np.uint8]]]:
    raise NotImplementedError
