"""Adaptador de teclas globais via GetAsyncKeyState (Windows, ADR 0003)."""

import ctypes
from typing import Any

from kinect.input.backend import KeyName

# Códigos de tecla virtual do Windows (winuser.h).
_VK_CODES: dict[KeyName, int] = {
    **{f"f{n}": 0x70 + n - 1 for n in range(1, 25)},  # VK_F1 = 0x70 … VK_F24 = 0x87
    "scrolllock": 0x91,
}
_PRESSED_NOW = 0x8000


class WindowsHotkeys:
    """``HotkeySource`` que lê o estado global do teclado, mesmo com o jogo em foco.

    Polling síncrono, uma vez por frame. Uma tecla conta como pressionada quando o bit
    ``0x8000`` de ``GetAsyncKeyState`` está ligado; o bit ``0x0001`` ("pressionada desde a
    última consulta") é ignorado por não ser confiável entre processos.

    Args:
        keys: nomes das teclas observadas (ex.: ``{"f8", "f9"}``).
        user32: ``ctypes.windll.user32`` ou um dublê; ``None`` usa o real (só no Windows).

    Raises:
        ValueError: se algum nome não tiver código de tecla virtual conhecido.
    """

    def __init__(self, keys: frozenset[KeyName], user32: Any = None) -> None:
        unknown = sorted(keys - _VK_CODES.keys())
        if unknown:
            raise ValueError(f"teclas sem código VK conhecido: {unknown}")
        self._codes = {key: _VK_CODES[key] for key in keys}
        # getattr: ``ctypes.windll`` só existe no Windows (mypy/ruff em outras plataformas).
        self._user32 = user32 if user32 is not None else getattr(ctypes, "windll").user32  # noqa: B009

    def poll(self) -> frozenset[KeyName]:
        return frozenset(
            key
            for key, vk in self._codes.items()
            if self._user32.GetAsyncKeyState(vk) & _PRESSED_NOW
        )
