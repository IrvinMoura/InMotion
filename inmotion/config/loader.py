"""Carregador de configuração (adaptador): lê YAML com ``safe_load`` e monta as camadas.

Precedência (RF5): ``default`` ← ``jogo`` ← ``perfil do usuário`` ← ``--set``. As camadas
são mescladas como dicts e só então validadas por ``build``; um erro cita o arquivo da
camada que definiu o campo.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from inmotion.config.schema import AppConfig, CameraConfig, GameProfile
from inmotion.config.validate import ConfigError, build, deep_merge, parse_override

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
GAMES_DIR = CONFIG_DIR / "games"
# Chaves de topo que pertencem ao ``GameProfile``; o resto da camada é do ``AppConfig``.
PROFILE_KEYS = ("name", "bindings")
_SET_LABEL = "--set"


def _read_yaml(path: Path) -> dict[str, object]:
    """Lê um arquivo YAML (só ``safe_load``); arquivo vazio vale como ``{}``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(str(path), f"não foi possível ler o arquivo: {exc}") from exc
    try:
        data: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), f"YAML inválido ou não permitido: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(
            str(path), f"esperado um mapa (chave: valor) no topo, recebido {type(data).__name__}"
        )
    return dict(data)


def _nest(keys: Sequence[str], value: object) -> dict[str, object]:
    """``(('a', 'b'), v)`` → ``{'a': {'b': v}}``."""
    head, *rest = keys
    return {head: _nest(rest, value) if rest else value}


def _override_layer(overrides: Sequence[str]) -> dict[str, object]:
    """Converte ``--set a.b=v`` em uma camada ``{a: {b: v}}`` (``v`` lido como YAML)."""
    layers: list[dict[str, object]] = []
    for text in overrides:
        keys, raw = parse_override(text)
        try:
            value: object = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise ConfigError(_SET_LABEL, f"valor inválido em {text!r}: {exc}") from exc
        layers.append(_nest(keys, value))
    return deep_merge(*layers)


def _defines(layer: Mapping[str, object], path: str) -> bool:
    """Diz se a camada tem o caminho ``a.b.c`` (ignora ``[i]`` de listas)."""
    node: object = layer
    for key in path.split("."):
        key = key.split("[", 1)[0]
        if not isinstance(node, Mapping) or key not in node:
            return False
        node = node[key]
    return True


def _with_origin(
    error: ConfigError, layers: Sequence[tuple[str, Mapping[str, object]]]
) -> Exception:
    """Acrescenta ao erro a camada (arquivo ou ``--set``) que definiu o campo, se houver."""
    for label, layer in reversed(layers):
        if error.path and _defines(layer, error.path):
            return ConfigError(error.path, f"{error.message} (origem: {label})")
    return error


def load_config(
    default_path: Path = DEFAULT_CONFIG_PATH,
    game: str | None = None,
    profile_path: Path | None = None,
    overrides: Sequence[str] = (),
    games_dir: Path = GAMES_DIR,
) -> tuple[AppConfig, GameProfile | None]:
    """Carrega ``default`` ← ``<games_dir>/<game>.yaml`` ← ``profile_path`` ← ``overrides``.

    Sem ``game`` nem ``bindings`` em nenhuma camada, o perfil de jogo é ``None``.

    Raises:
        ConfigError: arquivo ilegível ou inválido, ``--set`` mal formado, campo inválido.
    """
    layers: list[tuple[str, Mapping[str, object]]] = [(str(default_path), _read_yaml(default_path))]
    if game is not None:
        game_path = games_dir / f"{game}.yaml"
        if not game_path.is_file():
            available = sorted(p.stem for p in games_dir.glob("*.yaml"))
            raise ConfigError(
                "--game",
                f"perfil de jogo {game!r} não encontrado em {games_dir}. Disponíveis: {available}",
            )
        layers.append((str(game_path), _read_yaml(game_path)))
    if profile_path is not None:
        layers.append((str(profile_path), _read_yaml(profile_path)))
    if overrides:
        layers.append((_SET_LABEL, _override_layer(overrides)))

    merged = deep_merge(*(layer for _, layer in layers))
    profile_data = {key: merged.pop(key) for key in PROFILE_KEYS if key in merged}
    try:
        config = build(AppConfig, merged)
        profile: GameProfile | None = None
        if game is not None or "bindings" in profile_data:
            profile_data.setdefault("name", game or "perfil")
            profile = build(GameProfile, profile_data)
    except ConfigError as exc:
        raise _with_origin(exc, layers) from exc
    return config, profile


def load_camera_config(
    index: int | None = None, width: int | None = None, height: int | None = None
) -> CameraConfig:
    """``camera`` do ``config/default.yaml``, com campos trocados (``None`` = manter o YAML).

    Fonte única do padrão da câmera para as ferramentas (``tools/``).
    """
    camera = load_config()[0].camera
    return replace(
        camera,
        index=camera.index if index is None else index,
        width=camera.width if width is None else width,
        height=camera.height if height is None else height,
    )
