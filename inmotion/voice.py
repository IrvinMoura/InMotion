"""Comandos de voz para abrir menus e pausar os controles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Protocol

from inmotion.safety import SafetyManager


class VoiceCommand(Enum):
    INVENTORY = "inventory"
    MENU = "menu"
    BACK = "back"
    PAUSE = "pause"
    RESUME = "resume"


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = False
    model_dir: Path = Path("models/vosk-model-small-pt-0.3")
    device: int | None = None
    min_confidence: float = 0.90
    inventory: str = "jogo inventário"
    menu: str = "jogo menu"
    back: str = "jogo voltar"
    pause: str = "controle pausar"
    resume: str = "controle continuar"
    inventory_key: str = "e"
    menu_key: str = "esc"

    @property
    def vocabulary(self) -> dict[str, VoiceCommand]:
        return {" ".join(getattr(self, cmd.value).lower().split()): cmd for cmd in VoiceCommand}

    def __post_init__(self) -> None:
        phrases = self.vocabulary
        if len(phrases) != len(VoiceCommand) or any(
            not phrase or not all(word.isalpha() for word in phrase.split()) for phrase in phrases
        ):
            raise ValueError("use frases de voz distintas, formadas por palavras")
        if self.device is not None and self.device < 0:
            raise ValueError("device deve ser um índice >= 0")
        if not isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence deve estar entre 0 e 1")
        if not self.inventory_key or not self.menu_key or self.inventory_key == self.menu_key:
            raise ValueError("inventory_key e menu_key devem ser teclas distintas")


class VoiceSource(Protocol):
    def poll(self) -> VoiceCommand | None: ...
    def close(self) -> None: ...


def apply_command(
    command: VoiceCommand, config: VoiceConfig, safety: SafetyManager, t: float, hold_s: float
) -> None:
    if command in (VoiceCommand.PAUSE, VoiceCommand.RESUME):
        safety.set_paused(command is VoiceCommand.PAUSE)
    elif command is VoiceCommand.INVENTORY:
        safety.change_menu("inventario", config.inventory_key, t, hold_s)
    elif command is VoiceCommand.MENU:
        safety.change_menu("menu", config.menu_key, t, hold_s)
    elif command is VoiceCommand.BACK:
        safety.change_menu(None, config.menu_key, t, hold_s)
