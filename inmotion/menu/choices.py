"""Escolhas do menu (spec 024): viram argumentos do app e ficam salvas em ``profiles/``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_PATH = Path("profiles/menu.yaml")


@dataclass(frozen=True)
class MenuChoices:
    """Câmera (``None`` = a do config), microfone (``None`` = sem voz), jogo e mão."""

    game: str = "minecraft"
    camera: int | None = None
    microphone: int | None = None
    handedness: Literal["right", "left"] = "right"

    def to_argv(self) -> list[str]:
        raise NotImplementedError


def load_choices(path: Path = DEFAULT_PATH) -> MenuChoices:
    raise NotImplementedError


def save_choices(choices: MenuChoices, path: Path = DEFAULT_PATH) -> None:
    raise NotImplementedError


def reconcile(
    choices: MenuChoices,
    cameras: Sequence[int],
    microphones: Sequence[int],
    games: Sequence[str],
) -> tuple[MenuChoices, list[str]]:
    raise NotImplementedError
