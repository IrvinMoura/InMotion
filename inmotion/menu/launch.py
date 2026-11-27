"""Botão Jogar (spec 024): roda o app com as escolhas e devolve o erro como texto."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from inmotion.menu.choices import MenuChoices


@dataclass(frozen=True)
class PlayResult:
    code: int
    error: str | None


def play(choices: MenuChoices, main: Callable[[Sequence[str]], int] | None = None) -> PlayResult:
    raise NotImplementedError
