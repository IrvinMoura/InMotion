"""Spec 024 — menu inicial: escolhas, dispositivos e o botão Jogar (sem abrir janela)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from inmotion.__main__ import build_parser, resolve_config
from inmotion.menu.choices import MenuChoices, load_choices, reconcile, save_choices
from inmotion.menu.devices import (
    GameOption,
    list_game_profiles,
    list_microphones,
    probe_cameras,
)
from inmotion.menu.launch import play
from tests.fakes import FakeCapture

# --- CA1: escolhas viram os mesmos argumentos da linha de comando ---


def test_choices_become_cli_arguments() -> None:
    choices = MenuChoices(game="minecraft", camera=3, microphone=2, handedness="left")
    config, profile = resolve_config(build_parser().parse_args(choices.to_argv()))
    assert profile is not None
    assert profile.name == "minecraft"
    assert config.camera.index == 3
    assert (config.voice.enabled, config.voice.device) == (True, 2)
    assert config.handedness == "left"


def test_no_microphone_means_no_voice_and_no_camera_keeps_default() -> None:
    config, _ = resolve_config(build_parser().parse_args(MenuChoices().to_argv()))
    default, _ = resolve_config(build_parser().parse_args([]))
    assert config.voice.enabled is False
    assert config.camera.index == default.camera.index


# --- CA2: escolhas salvas e dispositivos que sumiram ---


def test_choices_round_trip_and_bad_file_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "menu.yaml"
    assert load_choices(path) == MenuChoices()  # sem arquivo
    choices = MenuChoices(camera=1, microphone=9, handedness="left")
    save_choices(choices, path)
    assert load_choices(path) == choices
    path.write_text("camera: [isso nao e numero\n", encoding="utf-8")
    assert load_choices(path) == MenuChoices()
    path.write_text("handedness: ambidestro\n", encoding="utf-8")
    assert load_choices(path) == MenuChoices()


def test_reconcile_replaces_missing_devices_with_a_warning() -> None:
    saved = MenuChoices(game="sumiu", camera=4, microphone=7)
    fixed, warnings = reconcile(saved, cameras=[0, 1], microphones=[2, 9], games=["minecraft"])
    assert fixed == MenuChoices(game="minecraft", camera=0, microphone=None)
    assert len(warnings) == 3
    same, none = reconcile(MenuChoices(camera=1, microphone=9), [0, 1], [2, 9], ["minecraft"])
    assert same == MenuChoices(camera=1, microphone=9)
    assert none == []


# --- CA3: perfis de jogo ---


def test_game_profiles_come_from_the_games_folder(tmp_path: Path) -> None:
    (tmp_path / "minecraft.yaml").write_text("name: minecraft\nbindings: {}\n", encoding="utf-8")
    (tmp_path / "roblox.yaml").write_text("bindings: {}\n", encoding="utf-8")
    (tmp_path / "notas.txt").write_text("x", encoding="utf-8")
    assert list_game_profiles(tmp_path) == [
        GameOption("minecraft", "Minecraft"),
        GameOption("roblox", "Roblox"),
    ]


def test_real_games_folder_has_minecraft() -> None:
    assert GameOption("minecraft", "Minecraft") in list_game_profiles()


# --- Dispositivos (dublês das bibliotecas) ---


def test_microphones_are_listed_once_by_name_from_mme() -> None:
    devices = [
        {"name": "Mapeador de som da Microsoft - Input", "hostapi": 0, "max_input_channels": 2},
        {"name": "Microfone (HyperX DuoCast)", "hostapi": 0, "max_input_channels": 2},
        {"name": "Alto-falantes", "hostapi": 0, "max_input_channels": 0},
        {"name": "Voicemeeter Out B1 (VB-Audio Vo", "hostapi": 0, "max_input_channels": 8},
        {"name": "Microfone (HyperX DuoCast)", "hostapi": 1, "max_input_channels": 2},
    ]
    apis = [{"name": "MME"}, {"name": "Windows WASAPI"}]
    assert list_microphones(lambda: devices, lambda: apis) == [
        (1, "Microfone (HyperX DuoCast)"),
        (3, "Voicemeeter Out B1 (VB-Audio Vo"),
    ]


def test_probe_cameras_keeps_only_indices_that_deliver_a_frame() -> None:
    opened: list[Any] = []

    def open_camera(index: int) -> FakeCapture:
        cap = FakeCapture(width=64, height=48)
        cap.opened = index in (0, 2)
        opened.append(cap)
        return cap

    found = probe_cameras(open_camera, max_index=4)
    assert [index for index, _ in found] == [0, 2]
    assert all(isinstance(img, np.ndarray) and img.shape == (48, 64, 3) for _, img in found)
    assert all(not cap.opened for cap in opened)  # todas soltas: o jogo precisa da câmera


# --- CA4: Jogar; erro do app vira mensagem ---


def test_play_runs_the_app_with_the_choices_and_a_log_file() -> None:
    seen: list[Sequence[str]] = []

    def fake_main(argv: Sequence[str]) -> int:
        seen.append(list(argv))
        return 0

    result = play(MenuChoices(camera=1), main=fake_main)
    assert (result.code, result.error) == (0, None)
    assert seen[0][:2] == ["--game", "minecraft"]
    assert "--log-file" in seen[0]  # sem terminal, o log vai para logs/


def test_play_turns_an_app_error_into_a_message() -> None:
    def failing_main(argv: Sequence[str]) -> int:
        logging.getLogger("inmotion").error("Falha ao abrir a câmera: ocupada")
        return 2

    result = play(MenuChoices(), main=failing_main)
    assert result.code == 2
    assert result.error == "Falha ao abrir a câmera: ocupada"


def test_play_turns_an_exception_into_a_message() -> None:
    def crashing_main(argv: Sequence[str]) -> int:
        raise OSError("disco cheio")

    result = play(MenuChoices(), main=crashing_main)
    assert result.code == 1
    assert result.error is not None
    assert "disco cheio" in result.error


def test_ppm_bytes_resizes_and_converts_bgr_to_rgb() -> None:
    """A miniatura vira PPM (P6) que o ``tk.PhotoImage`` lê sem Pillow."""
    from inmotion.menu.window import ppm_bytes

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)  # azul em BGR
    data = ppm_bytes(image, (8, 6))
    header = b"P6 8 6 255 "
    assert data.startswith(header)
    pixels = np.frombuffer(data[len(header) :], dtype=np.uint8).reshape(6, 8, 3)
    assert (pixels == (0, 0, 255)).all()  # azul em RGB
