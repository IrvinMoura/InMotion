"""Dublês de teste para os ports (sem hardware, sem Windows)."""

from collections.abc import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

from inmotion.types import N_LANDMARKS, FramePacket, PoseFrame


class SimulatedReleaseError(Exception):
    """Falha simulada ao soltar uma tecla (não herda de RuntimeError de propósito)."""


class FakeInputBackend:
    """Registra cada press/release; pode simular falha ao soltar certas teclas."""

    def __init__(self, fail_on_release: Iterable[str] = ()) -> None:
        self.events: list[tuple[str, str]] = []
        self.pressed: set[str] = set()
        self.fail_on_release = set(fail_on_release)

    def press(self, key: str) -> None:
        self.events.append(("press", key))
        self.pressed.add(key)

    def release(self, key: str) -> None:
        self.events.append(("release", key))
        if key in self.fail_on_release:
            raise SimulatedReleaseError(f"falha simulada ao soltar {key}")
        self.pressed.discard(key)

    def released(self) -> list[str]:
        return [key for kind, key in self.events if kind == "release"]


class FakeHotkeys:
    """Devolve uma sequência roteirizada de teclas globais; depois, nenhuma."""

    def __init__(self, script: Iterable[frozenset[str]]) -> None:
        self._script = list(script)

    def poll(self) -> frozenset[str]:
        return self._script.pop(0) if self._script else frozenset()


class FakeDirectInput:
    """Imita a API do pydirectinput usada pelo backend (sem enviar input real)."""

    def __init__(self) -> None:
        self.PAUSE = 0.1
        self.FAILSAFE = True
        self.KEYBOARD_MAPPING: dict[str, int | None] = {
            "space": 57,
            "shift": 42,
            "w": 17,
            "f13": None,  # existe no dicionário do pydirectinput, mas sem scancode
        }
        self.calls: list[tuple[str, str]] = []

    def keyDown(self, key: str) -> None:  # noqa: N802 - nome da API real
        self.calls.append(("keyDown", key))

    def keyUp(self, key: str) -> None:  # noqa: N802
        self.calls.append(("keyUp", key))

    def mouseDown(self, *, button: str) -> None:  # noqa: N802
        self.calls.append(("mouseDown", button))

    def mouseUp(self, *, button: str) -> None:  # noqa: N802
        self.calls.append(("mouseUp", button))


class FakeUser32:
    """Imita user32.GetAsyncKeyState: devolve o valor configurado para cada código VK."""

    def __init__(self, states: dict[int, int] | None = None) -> None:
        self.states = states or {}

    def GetAsyncKeyState(self, vk: int) -> int:  # noqa: N802 - nome da API do Windows
        return self.states.get(vk, 0)


# ---------------------------------------------------------------------------
# Dublês do pipeline (spec 002 T1): relógio, fonte de frames, estimador, captura.
# ---------------------------------------------------------------------------


class FakeClock:
    """Relógio manual (port ``Clock``): o tempo só anda com ``advance``."""

    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        if dt < 0:
            raise ValueError(f"dt deve ser >= 0, recebido {dt}")
        self.t += dt


class FakeFrameSource:
    """Port ``FrameSource`` roteirizado: devolve os itens em ordem; depois, ``None``."""

    def __init__(self, script: Iterable[FramePacket | None]) -> None:
        self._script = list(script)
        self.closed = False

    def read(self, timeout_s: float) -> FramePacket | None:
        return self._script.pop(0) if self._script else None

    def close(self) -> None:
        self.closed = True


class FakePoseEstimator:
    """Port ``PoseEstimator`` roteirizado.

    Cada item do roteiro é devolvido em ordem; um item ``Exception`` é lançado. Esgotado
    o roteiro, devolve ``None``. ``calls`` guarda o ``t`` de cada chamada.
    """

    def __init__(self, script: Iterable[PoseFrame | Exception | None]) -> None:
        self._script = list(script)
        self.calls: list[float] = []
        self.closed = False

    def estimate(self, image_bgr: NDArray[np.uint8], t: float) -> PoseFrame | None:
        self.calls.append(t)
        if not self._script:
            return None
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class FakeCapture:
    """Imita o ``cv2.VideoCapture`` usado pela ``Camera`` (spec 002 T7).

    ``read()`` devolve frames ``(height, width, 3)`` uint8 com o pixel ``[0, 0, 0]`` igual
    ao número do frame (mod 256); depois de ``max_frames`` frames, ``(False, None)``.
    ``set``/``get`` guardam as propriedades num dicionário (``props``).
    """

    def __init__(self, width: int = 64, height: int = 48, max_frames: int | None = None) -> None:
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.props: dict[int, float] = {}
        self.reads = 0
        self.frames = 0
        self.opened = True

    def isOpened(self) -> bool:  # noqa: N802 - nome da API do OpenCV
        return self.opened

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        self.reads += 1
        if not self.opened or (self.max_frames is not None and self.frames >= self.max_frames):
            return False, None
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[0, 0, 0] = self.frames % 256
        self.frames += 1
        return True, frame

    def release(self) -> None:
        self.opened = False


def make_pose(
    points_px: Mapping[int, tuple[float, float]],
    size: tuple[int, int] = (640, 480),
    t: float = 0.0,
    visibility: Mapping[int, float] | None = None,
    world: NDArray[np.float64] | None = None,
) -> PoseFrame:
    """Pose sintética: pontos dados em pixels (convertidos pela inversa de A5).

    Os pontos não informados ficam no centro da imagem. ``z = 0``; visibility e
    presence valem 1, exceto onde ``visibility`` sobrescreve.
    """
    w, h = size
    lm = np.zeros((N_LANDMARKS, 5), dtype=np.float64)
    lm[:, 0:2] = 0.5
    lm[:, 3:5] = 1.0
    for i, (x, y) in points_px.items():
        lm[i, 0] = x / w
        lm[i, 1] = y / h
    for i, v in (visibility or {}).items():
        lm[i, 3] = v
    return PoseFrame(t=t, image_size=size, landmarks=lm, world=world)
