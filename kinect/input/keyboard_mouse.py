"""Adaptador de teclado e mouse sobre o pydirectinput (Windows, SendInput)."""

from typing import Any

from kinect.input.backend import KeyName

_MOUSE_BUTTONS = {"mouse_left": "left", "mouse_right": "right", "mouse_middle": "middle"}


class KeyboardMouseBackend:
    """``InputBackend`` real: teclas por scancode e botões do mouse.

    Na construção:
    - ``PAUSE = 0``, porque o padrão de 100 ms por chamada trava o loop;
    - ``FAILSAFE = False`` (ADR 0004), porque com o cursor num canto o failsafe faria o
      próprio ``release_all()`` falhar na emergência.

    Args:
        api: módulo ``pydirectinput`` ou um dublê com a mesma interface. ``None`` importa
            o módulo real (só no Windows).
    """

    def __init__(self, api: Any = None) -> None:
        if api is None:
            # Import tardio: o módulo só existe/funciona no Windows.
            import pydirectinput

            api = pydirectinput
        api.PAUSE = 0
        api.FAILSAFE = False
        self._api = api

    def press(self, key: KeyName) -> None:
        if key in _MOUSE_BUTTONS:
            self._api.mouseDown(button=_MOUSE_BUTTONS[key])
        else:
            self._api.keyDown(self._checked(key))

    def release(self, key: KeyName) -> None:
        if key in _MOUSE_BUTTONS:
            self._api.mouseUp(button=_MOUSE_BUTTONS[key])
        else:
            self._api.keyUp(self._checked(key))

    def _checked(self, key: KeyName) -> KeyName:
        """Garante que a tecla tem scancode; o pydirectinput ignoraria em silêncio."""
        if self._api.KEYBOARD_MAPPING.get(key) is None:
            raise ValueError(f"tecla não suportada pelo pydirectinput: {key!r}")
        return key
