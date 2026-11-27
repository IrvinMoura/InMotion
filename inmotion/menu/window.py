"""Janela do menu inicial (spec 024), em tkinter.

Só monta a tela: as listas vêm de ``devices``, as escolhas de ``choices`` e o botão Jogar
chama ``launch.play``. A prévia da câmera e o medidor do microfone rodam em threads e são
fechados antes de o jogo começar, para o jogo poder abrir os mesmos dispositivos.
"""

from __future__ import annotations

import threading
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox, ttk
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from inmotion.menu.choices import MenuChoices, load_choices, reconcile, save_choices
from inmotion.menu.devices import GameOption, list_game_profiles, list_microphones, probe_cameras
from inmotion.menu.launch import PlayResult, play
from inmotion.voice import level_from_rms

THUMB_SIZE = (120, 68)  # miniaturas das câmeras
THUMB_COLUMNS = 3
PREVIEW_SIZE = (384, 216)  # prévia ao vivo da câmera escolhida
PREVIEW_MS = 66  # ~15 quadros/s na prévia: basta para reconhecer a imagem
METER_MS = 100
NO_VOICE = "Sem voz"
_FONT = "Segoe UI"


def ppm_bytes(image_bgr: NDArray[np.uint8], size: tuple[int, int]) -> bytes:
    """Imagem BGR redimensionada para ``size`` (largura, altura) em PPM (P6), que o
    ``tk.PhotoImage`` lê sem precisar do Pillow."""
    width, height = size
    rgb = cv2.cvtColor(cv2.resize(image_bgr, (width, height)), cv2.COLOR_BGR2RGB)
    return f"P6 {width} {height} 255 ".encode("ascii") + np.ascontiguousarray(rgb).tobytes()


class _CameraPreview:
    """Lê a câmera escolhida numa thread e guarda o último quadro."""

    def __init__(self, index: int) -> None:
        self._index = index
        self._stop = threading.Event()
        self.frame: NDArray[np.uint8] | None = None
        self._thread = threading.Thread(target=self._run, name="menu-preview", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
        try:
            while not self._stop.is_set() and cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    self.frame = np.asarray(frame, dtype=np.uint8)
        finally:
            cap.release()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class _MicMeter:
    """Abre o microfone escolhido e guarda o nível do último bloco (``level_from_rms``)."""

    def __init__(self, device: int) -> None:
        import sounddevice as sd

        self.level = 0.0
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        self._stream = sd.InputStream(
            device=device,
            channels=1,
            samplerate=rate,
            blocksize=rate // 10,
            dtype="int16",
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, data: Any, frames: int, timing: Any, status: Any) -> None:
        samples = np.asarray(data, dtype=np.float64)
        self.level = level_from_rms(float(np.sqrt(np.mean(samples**2)))) if samples.size else 0.0

    def close(self) -> None:
        self._stream.close()


def _dpi_aware() -> None:
    """Com a escala de tela acima de 100%, o Windows amplia janelas que não declaram DPI e
    elas ficam borradas; declarar faz o Tk desenhar nítido na escala certa."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass  # fora do Windows ou Windows antigo: segue sem


def _default_camera() -> int | None:
    """A câmera da configuração (``camera.index``), usada quando não há escolha salva."""
    try:
        from inmotion.config.loader import load_config

        return load_config()[0].camera.index
    except Exception:
        return None


class MenuWindow:
    """Menu: perfil, câmera (miniaturas + prévia), microfone (com medidor) e mão."""

    def __init__(self, player: Callable[[MenuChoices], PlayResult] = play) -> None:
        self._player = player
        _dpi_aware()
        self.root = tk.Tk()
        self.root.title("InMotion")
        self.root.resizable(False, False)
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Title.TLabel", font=(_FONT, 20, "bold"))
        style.configure("Hint.TLabel", font=(_FONT, 9), foreground="#666666")
        style.configure("Warn.TLabel", font=(_FONT, 9), foreground="#b35900")
        style.configure("TLabelframe.Label", font=(_FONT, 10, "bold"))
        style.configure("Play.TButton", font=(_FONT, 13, "bold"), padding=(30, 10))

        self._games: list[GameOption] = list_game_profiles()
        self._mics: list[tuple[int, str]] = []
        self._cameras: list[tuple[int, NDArray[np.uint8]]] = []
        self._thumbs: list[tk.PhotoImage] = []  # referências: o Tk não guarda as imagens
        self._preview: _CameraPreview | None = None
        self._meter: _MicMeter | None = None
        self._choices = load_choices()

        self._game = tk.StringVar()
        self._camera = tk.IntVar(value=-1)
        self._mic = tk.StringVar(value=NO_VOICE)
        self._hand = tk.StringVar(value=self._choices.handedness)
        self._status = tk.StringVar()
        self._build()
        self._load_microphones()
        self._search_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ tela
    def _build(self) -> None:
        body = ttk.Frame(self.root, padding=16)
        body.grid(sticky="nsew")
        ttk.Label(body, text="InMotion", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(body, text="Jogue com o corpo, usando só a câmera.", style="Hint.TLabel").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )
        left = ttk.Frame(body)
        left.grid(row=2, column=0, sticky="n")
        right = ttk.Frame(body)
        right.grid(row=2, column=1, sticky="n", padx=(14, 0))

        camera = ttk.LabelFrame(left, text="Câmera", padding=10)
        camera.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            camera,
            text="Escolha a imagem em que você aparece (1,5 a 2 m, corpo inteiro).",
            style="Hint.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self._thumb_grid = ttk.Frame(camera)
        self._thumb_grid.grid(row=1, column=0, sticky="w", pady=(6, 0))
        self._preview_label = ttk.Label(camera)
        self._preview_label.grid(row=2, column=0, pady=(8, 0))
        ttk.Button(camera, text="Procurar câmeras de novo", command=self._search_cameras).grid(
            row=3, column=0, pady=(8, 0), sticky="w"
        )

        game = ttk.LabelFrame(right, text="Perfil do jogo", padding=10)
        game.grid(row=0, column=0, sticky="ew")
        labels = [g.label for g in self._games] or ["(nenhum perfil em config/games)"]
        box = ttk.Combobox(game, textvariable=self._game, values=labels, state="readonly", width=30)
        box.grid(row=0, column=0, columnspan=2, sticky="w")
        saved = next((g.label for g in self._games if g.name == self._choices.game), labels[0])
        self._game.set(saved)
        ttk.Button(game, text="Adicionar perfil", state="disabled").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(game, text="em breve", style="Hint.TLabel").grid(
            row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
        )

        mic = ttk.LabelFrame(right, text="Microfone (comandos de voz)", padding=10)
        mic.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._mic_box = ttk.Combobox(mic, textvariable=self._mic, state="readonly", width=30)
        self._mic_box.grid(row=0, column=0, sticky="w")
        self._mic_box.bind("<<ComboboxSelected>>", lambda _e: self._restart_meter())
        ttk.Label(mic, text="Nível (fale para testar):", style="Hint.TLabel").grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        self._meter_bar = ttk.Progressbar(mic, length=220, maximum=100)
        self._meter_bar.grid(row=2, column=0, sticky="w")

        hand = ttk.LabelFrame(right, text="Mão dominante (ataque)", padding=10)
        hand.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        ttk.Radiobutton(hand, text="Destro", value="right", variable=self._hand).grid(
            row=0, column=0, padx=(0, 16)
        )
        ttk.Radiobutton(hand, text="Canhoto", value="left", variable=self._hand).grid(
            row=0, column=1
        )

        ttk.Label(right, textvariable=self._status, style="Warn.TLabel", wraplength=260).grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Button(right, text="Jogar", style="Play.TButton", command=self._on_play).grid(
            row=4, column=0, pady=(14, 4)
        )
        ttk.Label(
            right,
            text="No jogo: F8 pausa e retoma,\nF9 volta para este menu.",
            style="Hint.TLabel",
            justify="center",
        ).grid(row=5, column=0)

    # ------------------------------------------------------------ microfone
    def _load_microphones(self) -> None:
        try:
            self._mics = list_microphones()
        except Exception as exc:  # sem driver de áudio: a voz só fica indisponível
            self._mics = []
            self._warn(f"Não consegui listar os microfones: {exc}")
        self._mic_box.configure(values=[NO_VOICE] + [name for _, name in self._mics])
        saved = next((n for i, n in self._mics if i == self._choices.microphone), None)
        self._mic.set(saved or NO_VOICE)
        self._restart_meter()

    def _mic_index(self) -> int | None:
        name = self._mic.get()
        return next((i for i, n in self._mics if n == name), None)

    def _restart_meter(self) -> None:
        self._stop_meter()
        index = self._mic_index()
        if index is not None:
            try:
                self._meter = _MicMeter(index)
            except Exception as exc:
                self._warn(f"O microfone não abriu: {exc}")
        self._tick_meter()

    def _tick_meter(self) -> None:
        level = self._meter.level if self._meter is not None else 0.0
        self._meter_bar["value"] = round(level * 100)
        if self._meter is not None:
            self.root.after(METER_MS, self._tick_meter)

    def _stop_meter(self) -> None:
        if self._meter is not None:
            self._meter.close()
            self._meter = None

    # --------------------------------------------------------------- câmera
    def _search_cameras(self) -> None:
        self._stop_preview()
        for child in self._thumb_grid.winfo_children():
            child.destroy()
        ttk.Label(self._thumb_grid, text="Procurando câmeras...").grid()
        threading.Thread(target=self._probe_in_background, daemon=True).start()

    def _probe_in_background(self) -> None:
        try:
            found = probe_cameras()
        except Exception:
            found = []
        self.root.after(0, lambda: self._show_cameras(found))

    def _show_cameras(self, found: list[tuple[int, NDArray[np.uint8]]]) -> None:
        self._cameras = found
        for child in self._thumb_grid.winfo_children():
            child.destroy()
        self._thumbs = []
        if not found:
            ttk.Label(self._thumb_grid, text="Nenhuma câmera encontrada.").grid()
            return
        for col, (index, image) in enumerate(found):
            photo = tk.PhotoImage(data=ppm_bytes(image, THUMB_SIZE))
            self._thumbs.append(photo)
            ttk.Radiobutton(
                self._thumb_grid,
                text=f"Câmera {index}",
                image=photo,
                compound="top",
                value=index,
                variable=self._camera,
                command=self._start_preview,
            ).grid(row=col // THUMB_COLUMNS, column=col % THUMB_COLUMNS, padx=3, pady=3)
        fixed, warnings = reconcile(
            self._choices,
            [i for i, _ in found],
            [i for i, _ in self._mics],
            [g.name for g in self._games],
        )
        indices = [i for i, _ in found]
        preferred = fixed.camera if fixed.camera is not None else _default_camera()
        self._camera.set(preferred if preferred in indices else indices[0])
        for message in warnings:
            self._warn(message)
        self._start_preview()

    def _start_preview(self) -> None:
        self._stop_preview()
        if self._camera.get() >= 0:
            self._preview = _CameraPreview(self._camera.get())
            self._tick_preview()

    def _tick_preview(self) -> None:
        if self._preview is None:
            return
        frame = self._preview.frame
        if frame is not None:
            photo = tk.PhotoImage(data=ppm_bytes(frame, PREVIEW_SIZE))
            self._preview_label.configure(image=photo)
            self._preview_photo = photo
        self.root.after(PREVIEW_MS, self._tick_preview)

    def _stop_preview(self) -> None:
        if self._preview is not None:
            self._preview.close()
            self._preview = None

    # ---------------------------------------------------------------- jogar
    def current_choices(self) -> MenuChoices:
        game = next((g.name for g in self._games if g.label == self._game.get()), "minecraft")
        camera = self._camera.get()
        hand: Literal["right", "left"] = "left" if self._hand.get() == "left" else "right"
        return MenuChoices(game, camera if camera >= 0 else None, self._mic_index(), hand)

    def _warn(self, message: str) -> None:
        current = self._status.get()
        self._status.set(f"{current}\n{message}" if current else message)

    def _on_play(self) -> None:
        choices = self.current_choices()
        save_choices(choices)
        self._status.set("")
        self._stop_preview()
        self._stop_meter()
        self.root.withdraw()
        self.root.update()
        try:
            result = self._player(choices)
        finally:
            self.root.deiconify()
        if result.error is not None:
            messagebox.showerror("InMotion", result.error, parent=self.root)
        self._choices = choices
        self._start_preview()
        self._restart_meter()

    def _on_close(self) -> None:
        self._stop_preview()
        self._stop_meter()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
