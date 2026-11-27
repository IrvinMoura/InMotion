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
from inmotion.voice import (
    VoiceCommand,
    VoiceConfig,
    VoiceReception,
    apply_command,
    level_from_rms,
)
from tests.fakes import FakeClock, FakeInputBackend


def test_voice_is_optional_and_configurable() -> None:
    parser = build_parser()
    assert not resolve_config(parser.parse_args([]))[0].voice.enabled
    args = parser.parse_args(["--voice", "--set", "voice.menu=abrir menu"])
    voice = resolve_config(args)[0].voice
    assert voice.enabled
    assert voice.vocabulary["abrir menu"] is VoiceCommand.MENU
    with pytest.raises(ConfigError, match="distintas"):
        resolve_config(parser.parse_args(["--set", "voice.menu=voltar"]))


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
                return json.dumps({"text": "abrir menu", "result": [{"conf": 1.0}, {"conf": 0.6}]})
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
        deliver("abrir menu")
        assert source.poll() is VoiceCommand.MENU
        assert source.poll() is None
        for phrase in ("parcial", "qualquer frase", "menu", "incerta"):
            deliver(phrase)
            assert source.poll() is None
        deliver("voltar")
        clock.advance(1)
        assert source.poll() is None
    finally:
        source.close()
    assert not stream.active


# --- Indicador de recepção de voz (27/11/2026) ---


def test_level_from_rms_maps_dbfs_to_unit_range() -> None:
    """dBFS = 20·log10(rms/32768); nível = (dBFS − piso)/(−piso), recortado em [0, 1]."""
    assert level_from_rms(32768.0) == pytest.approx(1.0)
    assert level_from_rms(32768.0 * 10 ** (-30 / 20)) == pytest.approx(0.5)  # −30 dBFS
    assert level_from_rms(32768.0 * 10 ** (-60 / 20)) == pytest.approx(0.0)
    assert level_from_rms(1.0) == 0.0  # abaixo do piso
    assert level_from_rms(0.0) == 0.0
    assert level_from_rms(float("nan")) == 0.0
    assert level_from_rms(1e9) == 1.0


def test_reception_validates_level() -> None:
    assert VoiceReception(0.5).heard is None
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="level"):
            VoiceReception(bad)


def _scripted_microphone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, results: dict[bytes, dict[str, Any]]
) -> tuple[Any, Any, FakeClock, Event]:
    """Microfone com Vosk e sounddevice falsos: cada bloco de ``results`` fecha uma frase."""
    processed = Event()
    clock = FakeClock()
    opened: dict[str, Any] = {}

    class Stream:
        active = False

        def __init__(self, **options: Any) -> None:
            self.callback = options["callback"]
            opened["stream"] = self

        def start(self) -> None:
            self.active = True

        def close(self) -> None:
            self.active = False

    class Decoder:
        block = b""

        def SetWords(self, enabled: bool) -> None:  # noqa: N802
            pass

        def Reset(self) -> None:  # noqa: N802
            pass

        def AcceptWaveform(self, block: bytes) -> bool:  # noqa: N802
            self.block = block
            if block == b"fim":
                processed.set()
                return False
            return block in results

        def Result(self) -> str:  # noqa: N802
            return json.dumps(results[self.block])

    libs = {
        "sounddevice": SimpleNamespace(
            query_devices=lambda *args: {"default_samplerate": 16000}, RawInputStream=Stream
        ),
        "vosk": SimpleNamespace(
            Model=lambda path: SimpleNamespace(vosk_model_find_word=lambda word: 0),
            KaldiRecognizer=lambda *args: Decoder(),
        ),
    }
    monkeypatch.setattr(microphone, "import_module", libs.__getitem__)
    source = microphone.MicrophoneCommands(VoiceConfig(model_dir=tmp_path), clock)
    return source, opened["stream"], clock, processed


def test_reception_reports_level_and_last_phrase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    loud = np.full(1600, 16384, dtype=np.int16).tobytes()  # −6 dBFS
    results = {
        loud: {"text": "abrir menu", "result": [{"conf": 1.0}, {"conf": 0.97}]},
        b"duvida": {"text": "abrir inventário", "result": [{"conf": 1.0}, {"conf": 0.82}]},
        b"ruido": {"text": "[unk]", "result": [{"conf": 1.0}]},
    }
    source, stream, clock, processed = _scripted_microphone(tmp_path, monkeypatch, results)

    def deliver(block: bytes) -> None:
        processed.clear()
        stream.callback(block, 1600, None, False)
        stream.callback(b"fim", 1600, None, False)
        assert processed.wait(2)

    try:
        assert source.reception() == VoiceReception(0.0)
        stream.callback(loud, 1600, None, False)
        assert source.reception().level == pytest.approx(level_from_rms(16384.0))
        deliver(loud)
        rec = source.reception()
        assert (rec.heard, rec.heard_t, rec.accepted) == ("abrir menu", 0.0, True)
        assert (rec.confidence, rec.threshold) == (0.97, 0.90)  # a palavra mais fraca
        assert source.poll() is VoiceCommand.MENU
        clock.advance(1.0)
        deliver(b"duvida")  # frase do vocabulário com confiança baixa: ouvida, mas ignorada
        rec = source.reception()
        assert (rec.heard, rec.heard_t, rec.accepted) == ("abrir inventário", 1.0, False)
        assert rec.confidence == 0.82
        assert source.poll() is None
        deliver(b"ruido")  # fora do vocabulário
        assert source.reception().heard == "[unk]"
        assert source.reception().accepted is False
    finally:
        source.close()


def test_default_phrases() -> None:
    """Frases pedidas pelo usuário em 27/11/2026 (minúsculas: é como o Vosk devolve o texto)."""
    vocabulary = VoiceConfig().vocabulary
    assert vocabulary == {
        "abrir inventário": VoiceCommand.INVENTORY,
        "abrir menu": VoiceCommand.MENU,
        "voltar": VoiceCommand.BACK,
        "pausar controle": VoiceCommand.PAUSE,
        "continuar controle": VoiceCommand.RESUME,
    }


def test_reception_validates_confidence() -> None:
    assert VoiceReception(0.5, "voltar", 1.0, True, 0.95, 0.9).confidence == 0.95
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="confidence"):
            VoiceReception(0.5, "voltar", 1.0, True, bad, 0.9)


def test_each_closed_phrase_is_logged_with_confidence_and_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Rodada de calibração (27/11): cada frase fechada vai para o log, aceita ou não."""
    import logging

    import numpy as np

    loud = np.full(1600, 16384, dtype=np.int16).tobytes()
    results = {
        loud: {"text": "abrir menu", "result": [{"conf": 1.0}, {"conf": 0.97}]},
        b"duvida": {"text": "voltar", "result": [{"conf": 0.62}]},
        b"ruido": {"text": "[unk]", "result": [{"conf": 1.0}]},
    }
    source, stream, _, processed = _scripted_microphone(tmp_path, monkeypatch, results)

    def deliver(block: bytes) -> None:
        processed.clear()
        stream.callback(block, 1600, None, False)
        stream.callback(b"fim", 1600, None, False)
        assert processed.wait(2)

    caplog.set_level(logging.INFO, logger="inmotion.capture.microphone")
    try:
        deliver(loud)
        deliver(b"duvida")
        deliver(b"ruido")
    finally:
        source.close()
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("voz:")]
    # Nos blocos falsos o texto vira amostras int16; o pico de uma frase é o maior nível
    # desde a frase anterior (inclui o bloco "fim" que separa as frases).
    level = microphone.MicrophoneCommands._block_level
    peaks = [level(loud), max(level(b"fim"), level(b"duvida")), max(level(b"fim"), level(b"ruido"))]
    assert lines == [
        f'voz: frase="abrir menu" confianca=0.97 palavras=1.00,0.97 aceita=sim pico={peaks[0]:.2f}',
        f'voz: frase="voltar" confianca=0.62 palavras=0.62 aceita=nao pico={peaks[1]:.2f}',
        f'voz: frase="[unk]" confianca=1.00 palavras=1.00 aceita=nao pico={peaks[2]:.2f}',
    ]
    assert peaks[0] == pytest.approx(level_from_rms(16384.0))
