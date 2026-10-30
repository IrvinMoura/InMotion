"""Validação e mesclagem de configuração (núcleo puro, ADR 0006).

``build`` converte o ``dict`` vindo do YAML em dataclasses imutáveis, percorrendo as
anotações de tipo; cada dataclass é dona da validação das suas faixas (``__post_init__``).
Todo erro sai como ``ConfigError`` com o caminho completo do campo.
"""

from __future__ import annotations

import dataclasses
import difflib
import re
import types
import typing
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

_UNION_ORIGINS = (typing.Union, types.UnionType)
_LEADING_WORD = re.compile(r"\s*(\w+)")


class ConfigError(ValueError):
    """Erro de configuração, sempre com o caminho do campo (ex.: ``filters.ema.alpha``)."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.path = path


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _describe(value: object) -> str:
    return f"{value!r} ({type(value).__name__})"


def _wrong(path: str, expected: str, value: object) -> ConfigError:
    return ConfigError(path, f"esperado {expected}, recebido {_describe(value)}")


def _convert(tp: Any, value: object, path: str) -> Any:
    """Converte ``value`` para o tipo anotado ``tp``; ``TypeError`` se ``tp`` não é suportado."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in _UNION_ORIGINS:
        members = [a for a in args if a is not type(None)]
        if len(members) != 1 or len(members) == len(args):
            raise TypeError(f"só X | None é suportado em {path or '<raiz>'}: {tp}")
        return None if value is None else _convert(members[0], value, path)

    if origin is typing.Literal:
        if not any(value == a and type(value) is type(a) for a in args):
            raise ConfigError(path, f"esperado um de {list(args)!r}, recebido {_describe(value)}")
        return value

    if origin in (frozenset, tuple):
        if origin is tuple and (len(args) != 2 or args[1] is not Ellipsis):
            raise TypeError(f"tupla suportada só como tuple[X, ...] em {path or '<raiz>'}: {tp}")
        if not isinstance(value, list | tuple):
            raise _wrong(path, "uma lista", value)
        items = [_convert(args[0], item, f"{path}[{i}]") for i, item in enumerate(value)]
        return frozenset(items) if origin is frozenset else tuple(items)

    if origin is dict:
        if len(args) != 2 or args[0] is not str:
            raise TypeError(f"dict suportado só como dict[str, X] em {path or '<raiz>'}: {tp}")
        if not isinstance(value, Mapping):
            raise _wrong(path, "um mapa (chave: valor)", value)
        return {str(k): _convert(args[1], v, _join(path, str(k))) for k, v in value.items()}

    if origin is not None:
        raise TypeError(f"tipo não suportado em {path or '<raiz>'}: {tp}")

    if tp is bool:
        if not isinstance(value, bool):
            raise _wrong(path, "bool (true/false)", value)
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _wrong(path, "int", value)
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _wrong(path, "float", value)
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise _wrong(path, "str", value)
        return value
    if tp is Path:
        if not isinstance(value, str):
            raise _wrong(path, "um caminho (str)", value)
        return Path(value)
    if isinstance(tp, type) and issubclass(tp, Enum):
        names = {member.name.lower(): member for member in tp}
        if not isinstance(value, str) or value.lower() not in names:
            raise ConfigError(
                path, f"esperado um de {sorted(names)!r}, recebido {_describe(value)}"
            )
        return names[value.lower()]
    if dataclasses.is_dataclass(tp):
        return build(tp, value, path)  # type: ignore[arg-type]  # `value` é validado em build
    raise TypeError(f"tipo não suportado em {path or '<raiz>'}: {tp}")


def _failed_field(error: ValueError, fields: set[str]) -> str | None:
    """Campo citado no início da mensagem do ``__post_init__`` (ex.: ``alpha deve ...``)."""
    match = _LEADING_WORD.match(str(error))
    return match.group(1) if match and match.group(1) in fields else None


def build(cls: type[T], data: Mapping[str, object], path: str = "") -> T:
    """Constrói a dataclass ``cls`` a partir de ``data`` (dict vindo do YAML).

    Suporta: bool, int, float (aceita int), str, Path, Enum (pelo nome, sem maiúsculas),
    Literal[...], X | None, frozenset[X]/tuple[X, ...] (de listas), dict[str, X] e
    dataclasses aninhadas. Chave desconhecida → ConfigError com sugestão (difflib). Campo
    ausente sem padrão → ConfigError. ValueError do __post_init__ → ConfigError com o
    caminho (do campo, quando a mensagem começa pelo nome dele).

    Raises:
        ConfigError: dado inválido (ausente, tipo, chave desconhecida, faixa).
        TypeError: ``cls`` não é dataclass ou usa um tipo que o construtor não suporta.
    """
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"build() exige uma dataclass, recebeu {cls!r}")
    if not isinstance(data, Mapping):
        raise _wrong(path, "um mapa (chave: valor)", data)

    fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
    hints = typing.get_type_hints(cls)
    for key in data:
        if key not in fields:
            close = difflib.get_close_matches(str(key), list(fields), n=1)
            hint = f" Você quis dizer {close[0]!r}?" if close else ""
            raise ConfigError(
                _join(path, str(key)),
                f"chave desconhecida {key!r}.{hint} Chaves válidas: {sorted(fields)}",
            )

    kwargs: dict[str, Any] = {}
    for name, fld in fields.items():
        if name in data:
            kwargs[name] = _convert(hints[name], data[name], _join(path, name))
        elif fld.default is dataclasses.MISSING and fld.default_factory is dataclasses.MISSING:
            raise ConfigError(_join(path, name), "campo obrigatório ausente")

    try:
        return cls(**kwargs)
    except ValueError as exc:
        failed = _failed_field(exc, set(fields))
        raise ConfigError(_join(path, failed) if failed else path, str(exc)) from exc


def deep_merge(*layers: Mapping[str, object]) -> dict[str, object]:
    raise NotImplementedError


def parse_override(text: str) -> tuple[tuple[str, ...], str]:
    raise NotImplementedError
