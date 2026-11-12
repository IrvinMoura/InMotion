"""Interfaces de entrada do núcleo: quem aperta teclas e quem lê teclas globais.

Teclas e botões do mouse compartilham um único espaço de nomes (``KeyName``), para que
o conjunto de "pressionados" seja um só.
"""

from typing import Final, Protocol, TypeAlias, runtime_checkable

KeyName: TypeAlias = str
"""Nome de tecla ou botão: ``"space"``, ``"shift"``, ``"w"``, ``"f8"``, ``"mouse_left"``..."""

MOUSE_BUTTONS: Final[frozenset[KeyName]] = frozenset({"mouse_left", "mouse_right", "mouse_middle"})


class InputBackend(Protocol):
    """Envia input ao sistema operacional (teclado, mouse, gamepad)."""

    def press(self, key: KeyName) -> None: ...

    def release(self, key: KeyName) -> None: ...


@runtime_checkable
class PointerBackend(Protocol):
    """Move o ponteiro de forma relativa (spec 014: câmera pela cabeça).

    Port separado do ``InputBackend``: movimento não fica "pressionado", então não entra
    em P nem no ``release_all()``. Quem não move o mouse (ex.: o log da sessão guiada)
    simplesmente não implementa este port.
    """

    def move_rel(self, dx: int, dy: int) -> None: ...


class HotkeySource(Protocol):
    """Lê teclas globais, mesmo com outra janela (o jogo) em foco."""

    def poll(self) -> frozenset[KeyName]:
        """Teclas pressionadas agora, ex.: ``frozenset({"f8"})``."""
        ...
