"""Escolhas do menu (spec 024): viram argumentos do app e ficam salvas em ``profiles/``.

As escolhas viram exatamente os argumentos que a pessoa digitaria no terminal (``--game``,
``--camera``, ``--voice`` e ``--set``), então o jogo roda igual ao da linha de comando.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import yaml

DEFAULT_PATH = Path("profiles/menu.yaml")  # profiles/ fica fora do git
_HANDS = ("right", "left")


@dataclass(frozen=True)
class MenuChoices:
    """Câmera (``None`` = a do config), microfone (``None`` = sem voz), jogo e mão."""

    game: str = "minecraft"
    camera: int | None = None
    microphone: int | None = None
    handedness: Literal["right", "left"] = "right"

    def __post_init__(self) -> None:
        if not self.game:
            raise ValueError("game não pode ser vazio")
        for name in ("camera", "microphone"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"{name} deve ser um índice inteiro ou None")
            if isinstance(value, int) and value < 0:
                raise ValueError(f"{name} deve ser >= 0")
        if self.handedness not in _HANDS:
            raise ValueError(f"handedness deve ser right ou left, recebido {self.handedness!r}")

    def to_argv(self) -> list[str]:
        """Os argumentos da linha de comando equivalentes a estas escolhas."""
        argv = ["--game", self.game]
        if self.camera is not None:
            argv += ["--camera", str(self.camera)]
        if self.microphone is not None:
            argv += ["--voice", "--set", f"voice.device={self.microphone}"]
        argv += ["--set", f"handedness={self.handedness}"]
        return argv


def load_choices(path: Path = DEFAULT_PATH) -> MenuChoices:
    """Escolhas salvas; arquivo ausente, ilegível ou inválido devolve as padrão."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return MenuChoices()
        known = {k: v for k, v in data.items() if k in MenuChoices.__dataclass_fields__}
        return MenuChoices(**known)
    except (OSError, yaml.YAMLError, TypeError, ValueError):
        return MenuChoices()


def save_choices(choices: MenuChoices, path: Path = DEFAULT_PATH) -> None:
    """Grava as escolhas em YAML (cria a pasta se preciso)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Escolhas do menu inicial (spec 024). Gerado pelo menu; pode apagar.\n"
    text += yaml.safe_dump(asdict(choices), allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def reconcile(
    choices: MenuChoices,
    cameras: Sequence[int],
    microphones: Sequence[int],
    games: Sequence[str],
) -> tuple[MenuChoices, list[str]]:
    """Troca o que sumiu desde a última vez e explica cada troca.

    Câmera sumida vira a primeira encontrada; microfone sumido vira "sem voz"; jogo sumido
    vira o primeiro da lista. O que ainda existe fica como estava.
    """
    warnings: list[str] = []
    fixed = choices
    if choices.camera is not None and choices.camera not in cameras:
        first = cameras[0] if cameras else None
        warnings.append(f"A câmera {choices.camera} não foi encontrada; usando a {first}.")
        fixed = replace(fixed, camera=first)
    if choices.microphone is not None and choices.microphone not in microphones:
        warnings.append("O microfone salvo não foi encontrado; a voz ficou desligada.")
        fixed = replace(fixed, microphone=None)
    if games and choices.game not in games:
        warnings.append(f"O perfil {choices.game!r} não existe mais; usando {games[0]!r}.")
        fixed = replace(fixed, game=games[0])
    return fixed, warnings
