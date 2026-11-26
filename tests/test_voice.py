import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any

import pytest

from inmotion.__main__ import build_parser, resolve_config
from inmotion.capture import microphone
from inmotion.config.validate import ConfigError
from inmotion.safety import SafetyConfig, SafetyManager, SafetyState, TrackedInput
from inmotion.voice import VoiceCommand, VoiceConfig, apply_command
from tests.fakes import FakeClock, FakeInputBackend


def test_voice_is_optional_and_configurable() -> None:
    parser = build_parser()
    assert not resolve_config(parser.parse_args([]))[0].voice.enabled
    args = parser.parse_args(["--voice", "--set", "voice.menu=abrir menu"])
    voice = resolve_config(args)[0].voice
    assert voice.enabled
    assert voice.vocabulary["abrir menu"] is VoiceCommand.MENU
    with pytest.raises(ConfigError, match="distintas"):
        resolve_config(parser.parse_args(["--set", "voice.menu=jogo voltar"]))


def test_menu_requires_tracking_and_obeys_pause_and_watchdog() -> None:
    backend = FakeInputBackend()
    safety = SafetyManager(TrackedInput(backend), SafetyConfig())
    config = VoiceConfig()

    def command(action: VoiceCommand, t: float) -> None:
        apply_command(action, config, safety, t, 0.08)

    safety.on_frame(0, False)
    command(VoiceCommand.MENU, 0)
    assert not backend.events
    safety.on_frame(1, True)
    command(VoiceCommand.PAUSE, 1)
    command(VoiceCommand.PAUSE, 1)
    command(VoiceCommand.MENU, 1)
    assert safety.state is SafetyState.PAUSED
    assert not backend.pressed
    command(VoiceCommand.RESUME, 1)
    command(VoiceCommand.MENU, 1)
    assert backend.pressed == {"esc"}
    safety.on_stall()
    assert not backend.pressed
    safety.on_frame(2, False)
    command(VoiceCommand.BACK, 2)
    assert safety.menu == "menu"
    safety.on_frame(3, True)
    safety.input.press("w")
    safety.input.move_rel(3, 3)
    assert not backend.moves
    assert not backend.pressed
    safety.on_hotkeys(frozenset({"f8"}))
    assert safety.menu is None
    assert safety.state is SafetyState.PAUSED
    safety.on_hotkeys(frozenset())
    safety.on_hotkeys(frozenset({"f8"}))
    assert safety.state is SafetyState.ACTIVE


def test_audio_worker_only_delivers_recent_complete_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = Event()
    clock = FakeClock()

    class Stream:
        active = False

        def __init__(self, **options: Any) -> None:
            self.callback = options["callback"]

        def start(self) -> None:
            self.active = True

        def close(self) -> None:
            self.active = False

    class Decoder:
        phrase = ""

        def SetWords(self, enabled: bool) -> None:  # noqa: N802
            assert enabled

        def AcceptWaveform(self, block: bytes) -> bool:  # noqa: N802
            self.phrase = block.decode()
            if self.phrase == "barreira":
                processed.set()
                return False
            return self.phrase != "parcial"

        def Result(self) -> str:  # noqa: N802
            if self.phrase == "incerta":
                return json.dumps({"text": "jogo menu", "result": [{"conf": 1.0}, {"conf": 0.6}]})
            return json.dumps({"text": self.phrase, "result": [{"conf": 1.0}]})

    stream = None

    def open_stream(**options: Any) -> Stream:
        nonlocal stream
        stream = Stream(**options)
        return stream

    libs = {
        "sounddevice": SimpleNamespace(
            query_devices=lambda *args: {"default_samplerate": 16000}, RawInputStream=open_stream
        ),
        "vosk": SimpleNamespace(
            Model=lambda path: SimpleNamespace(vosk_model_find_word=lambda word: 0),
            KaldiRecognizer=lambda *args: Decoder(),
        ),
    }
    monkeypatch.setattr(microphone, "import_module", libs.__getitem__)
    source = microphone.MicrophoneCommands(VoiceConfig(model_dir=tmp_path), clock)
    assert stream is not None

    def deliver(phrase: str) -> None:
        processed.clear()
        stream.callback(phrase.encode(), 1600, None, False)
        stream.callback(b"barreira", 1600, None, False)
        assert processed.wait(2)

    try:
        deliver("jogo menu")
        assert source.poll() is VoiceCommand.MENU
        assert source.poll() is None
        for phrase in ("parcial", "qualquer frase", "menu", "incerta"):
            deliver(phrase)
            assert source.poll() is None
        deliver("jogo voltar")
        clock.advance(1)
        assert source.poll() is None
    finally:
        source.close()
    assert not stream.active
