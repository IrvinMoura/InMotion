"""Botão Jogar (spec 024): roda o app com as escolhas e devolve o erro como texto.

O app roda no mesmo processo, pelo mesmo ``main`` da linha de comando, então a segurança
(F8, F9, ``release_all``) é a de sempre. Sem terminal, o log vai para ``logs/menu.log`` e a
última mensagem de erro do app vira o texto da caixa de mensagem do menu.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from inmotion.menu.choices import MenuChoices

LOG_FILE = "menu.log"  # vai para logs/ (pasta fora do git)
_APP_LOGGER = "inmotion"  # o basicConfig do app troca só os handlers da raiz, não os deste
_GENERIC_ERROR = "O jogo fechou com um erro. Veja os detalhes em logs/menu.log."


@dataclass(frozen=True)
class PlayResult:
    """Código de saída do app e, se falhou, a mensagem para mostrar."""

    code: int
    error: str | None


class _ErrorCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def play(choices: MenuChoices, main: Callable[[Sequence[str]], int] | None = None) -> PlayResult:
    """Roda o app com ``choices`` e espera ele terminar (F9 ou janela fechada)."""
    if main is None:
        from inmotion.__main__ import main as app_main

        main = app_main
    collector = _ErrorCollector()
    logger = logging.getLogger(_APP_LOGGER)
    logger.addHandler(collector)
    try:
        code = main([*choices.to_argv(), "--log-file", LOG_FILE])
    except SystemExit as exc:  # argparse recusou um argumento
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        logger.exception("Falha inesperada no jogo")
        return PlayResult(1, f"Erro inesperado: {exc}")
    finally:
        logger.removeHandler(collector)
    if code == 0:
        return PlayResult(0, None)
    return PlayResult(code, collector.messages[-1] if collector.messages else _GENERIC_ERROR)
