"""``python -m inmotion.menu``: abre o menu inicial (spec 024).

Aberto pelo ``InMotion.bat`` com ``pythonw`` (sem terminal), um erro na abertura vira
caixa de mensagem e vai para ``logs/menu.log``.
"""

from __future__ import annotations

import logging
from pathlib import Path


def main() -> int:
    try:
        from inmotion.menu.window import MenuWindow

        MenuWindow().run()
    except Exception as exc:
        Path("logs").mkdir(exist_ok=True)
        logging.basicConfig(filename="logs/menu.log", encoding="utf-8", level=logging.INFO)
        logging.getLogger("inmotion.menu").exception("O menu não abriu")
        from tkinter import messagebox

        messagebox.showerror("InMotion", f"O menu não abriu: {exc}\nDetalhes em logs/menu.log.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
