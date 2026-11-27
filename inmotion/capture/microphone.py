"""Captura de áudio e reconhecimento de voz offline com Vosk."""

from __future__ import annotations

import json
import logging
import threading
from collections import deque
from contextlib import suppress
from importlib import import_module
from queue import Empty, Full, Queue
from typing import Any

import numpy as np

from inmotion.ports import Clock
from inmotion.voice import VoiceCommand, VoiceConfig, VoiceReception, level_from_rms

_MAX_AGE_S = 0.8
_log = logging.getLogger(__name__)


class MicrophoneCommands:
    def __init__(self, config: VoiceConfig, clock: Clock) -> None:
        if not config.model_dir.is_dir():
            raise ValueError(f"Extraia o modelo de português em {config.model_dir} (veja o README)")
        try:
            vosk = import_module("vosk")
            audio = import_module("sounddevice")
        except ImportError as exc:
            raise RuntimeError(
                "Instale a voz com python -m pip install -r requirements-voice.txt"
            ) from exc
        self._vocabulary = config.vocabulary
        model = vosk.Model(str(config.model_dir))
        words = {word for phrase in self._vocabulary for word in phrase.split()}
        missing = sorted(word for word in words if model.vosk_model_find_word(word) < 0)
        if missing:
            raise ValueError(f"Palavras ausentes no modelo de voz: {', '.join(missing)}")
        rate = int(audio.query_devices(config.device, "input")["default_samplerate"])
        grammar = json.dumps([*self._vocabulary, "[unk]"], ensure_ascii=False)
        self._recognizer = vosk.KaldiRecognizer(model, rate, grammar)
        self._recognizer.SetWords(True)
        self._min_confidence = config.min_confidence
        self._clock = clock
        self._blocks: Queue[tuple[int, float, bytes]] = Queue(maxsize=8)
        self._sequence = 0
        self._commands: deque[tuple[float, VoiceCommand]] = deque(maxlen=1)
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._level = 0.0  # escrito pela thread do áudio; leitura de um float é atômica
        self._phrase_peak = 0.0  # maior nível desde a última frase fechada (só a thread do Vosk)
        self._heard: tuple[str, float, bool, float | None] | None = None
        self._worker = threading.Thread(target=self._listen, name="vosk", daemon=True)
        self._stream = audio.RawInputStream(
            device=config.device,
            samplerate=rate,
            blocksize=rate // 10,
            channels=1,
            dtype="int16",
            callback=self._capture,
        )
        try:
            self._stream.start()
            self._worker.start()
        except BaseException:
            self.close()
            raise

    def _capture(self, data: Any, frames: int, timing: Any, status: Any) -> None:
        # Um salto no contador faz o decodificador esquecer a frase interrompida.
        self._sequence += 2 if status else 1
        self._level = self._block_level(data)
        with suppress(Full):
            self._blocks.put_nowait((self._sequence, self._clock.now(), bytes(data)))

    def _listen(self) -> None:
        previous = 0
        try:
            while not self._stop.is_set():
                try:
                    sequence, t, data = self._blocks.get(timeout=0.1)
                except Empty:
                    continue
                if self._clock.now() - t > _MAX_AGE_S:
                    continue
                if sequence != previous + 1:
                    self._recognizer.Reset()
                previous = sequence
                self._phrase_peak = max(self._phrase_peak, self._block_level(data))
                if self._recognizer.AcceptWaveform(data):
                    result = json.loads(self._recognizer.Result())
                    command = self._vocabulary.get(result.get("text", ""))
                    words = result.get("result", [])
                    accepted = (
                        command is not None
                        and bool(words)
                        and all(word["conf"] >= self._min_confidence for word in words)
                    )
                    if command is not None and accepted:
                        self._commands.append((t, command))
                    text = result.get("text", "")
                    if text:
                        weakest = min((float(w["conf"]) for w in words), default=None)
                        self._heard = (text, t, accepted, weakest)
                        self._log_phrase(text, words, accepted)
                    self._phrase_peak = 0.0
        except Exception as exc:
            self._error = exc

    @staticmethod
    def _block_level(data: Any) -> float:
        """Nível de um bloco int16 (``level_from_rms`` do RMS das amostras)."""
        samples = np.frombuffer(bytes(data), dtype=np.int16, count=len(data) // 2)
        if not samples.size:
            return 0.0
        return level_from_rms(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))

    def _log_phrase(self, text: str, words: list[dict[str, Any]], accepted: bool) -> None:
        """Uma linha por frase fechada, para calibrar ``min_confidence`` com dados
        (``tools.voice_report`` lê estas linhas)."""
        confs = [float(w["conf"]) for w in words]
        _log.info(
            'voz: frase="%s" confianca=%s palavras=%s aceita=%s pico=%.2f',
            text,
            f"{min(confs):.2f}" if confs else "-",
            ",".join(f"{c:.2f}" for c in confs) or "-",
            "sim" if accepted else "nao",
            self._phrase_peak,
        )

    def poll(self) -> VoiceCommand | None:
        if self._error is not None or not self._stream.active:
            raise RuntimeError("O reconhecimento de voz foi interrompido") from self._error
        try:
            t, command = self._commands.popleft()
        except IndexError:
            return None
        return command if self._clock.now() - t <= _MAX_AGE_S else None

    def reception(self) -> VoiceReception:
        """Nível do último bloco e a última frase fechada (para o indicador da janela)."""
        heard = self._heard
        if heard is None:
            return VoiceReception(self._level)
        text, t, accepted, weakest = heard
        return VoiceReception(self._level, text, t, accepted, weakest, self._min_confidence)

    def close(self) -> None:
        self._stop.set()
        try:
            self._stream.close()
        finally:
            if self._worker.is_alive():
                self._worker.join(timeout=1.0)
