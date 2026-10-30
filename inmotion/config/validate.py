"""Validação e mesclagem de configuração (núcleo puro, ADR 0006). Stubs da T1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Erro de configuração, sempre com o caminho do campo (ex.: ``filters.ema.alpha``)."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


def build(cls: type[T], data: Mapping[str, object], path: str = "") -> T:
    raise NotImplementedError


def deep_merge(*layers: Mapping[str, object]) -> dict[str, object]:
    raise NotImplementedError


def parse_override(text: str) -> tuple[tuple[str, ...], str]:
    raise NotImplementedError
