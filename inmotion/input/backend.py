"""Interfaces de entrada do núcleo: quem aperta teclas e quem lê teclas globais.

Teclas e botões do mouse compartilham um único espaço de nomes (``KeyName``), para que
o conjunto de "pressionados" seja um só.
"""

from typing import Final, Protocol, TypeAlias

KeyName: TypeAlias = str
"""Nome de tecla ou botão: ``"space"``, ``"shift"``, ``"w"``, ``"f8"``, ``"mouse_left"``..."""

MOUSE_BUTTONS: Final[frozenset[KeyName]] = frozenset({"mouse_left", "mouse_right", "mouse_middle"})


class InputBackend(Protocol):
    """Envia input ao sistema operacional (teclado, mouse, gamepad)."""

    def press(self, key: KeyName) -> None: ...

    def release(self, key: KeyName) -> None: ...


class HotkeySource(Protocol):
    """Lê teclas globais, mesmo com outra janela (o jogo) em foco."""

    def poll(self) -> frozenset[KeyName]:
        """Teclas pressionadas agora, ex.: ``frozenset({"f8"})``."""
        ...
