"""Spec 002 T7 — Camera em thread: sempre o frame mais recente (CA4, CA5)."""

import itertools
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from inmotion.capture.camera import _BACKENDS, Camera
from inmotion.config.schema import CAMERA_BACKENDS, CameraConfig
from tests.fakes import FakeCapture, FakeClock


class CameraBoomError(Exception):
    """Falha simulada da captura (não herda de RuntimeError de propósito)."""


class PacedCapture(FakeCapture):
    """FakeCapture que leva ``period_s`` por leitura (imita uma câmera de 60 Hz)."""

    def __init__(self, period_s: float = 1 / 60, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.period_s = period_s

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        time.sleep(self.period_s)
        return super().read()


class BoomCapture(FakeCapture):
    """Lança ``CameraBoomError`` na leitura de número ``fail_at``."""

    def __init__(self, fail_at: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.fail_at = fail_at

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        if self.reads >= self.fail_at:
            raise CameraBoomError("falha simulada na captura")
        time.sleep(0.002)
        return super().read()


def make_camera(
    cap: FakeCapture, config: CameraConfig | None = None
) -> tuple[Camera, list[tuple[int, int]]]:
    calls: list[tuple[int, int]] = []

    def factory(index: int, api: int) -> FakeCapture:
        calls.append((index, api))
        return cap

    return Camera(config or CameraConfig(), FakeClock(), capture_factory=factory), calls


def drain(cam: Camera) -> None:
    for _ in range(50):
        cam.read(timeout_s=0.5)


def test_read_returns_latest_frame() -> None:
    cap = PacedCapture(period_s=1 / 60)
    cam, _ = make_camera(cap)
    cam.start()
    indices: list[int] = []
    try:
        for _ in range(10):  # consumidor ~20 Hz
            pkt = cam.read(timeout_s=1.0)
            assert pkt is not None
            assert int(pkt.image[0, 0, 0]) == pkt.index % 256
            indices.append(pkt.index)
            time.sleep(0.05)
    finally:
        cam.close()
    assert all(b > a for a, b in itertools.pairwise(indices))
    gaps = [b - a for a, b in itertools.pairwise(indices)]
    assert max(gaps) > 1  # produtor mais rápido: frames intermediários descartados


def test_read_returns_none_on_timeout_without_new_frame() -> None:
    cap = FakeCapture(max_frames=1)
    cam, _ = make_camera(cap)
    cam.start()
    try:
        first = cam.read(timeout_s=1.0)
        assert first is not None
        assert cam.read(timeout_s=0.05) is None  # nada novo
    finally:
        cam.close()


def test_close_joins_quickly() -> None:
    cap = PacedCapture(period_s=1 / 60)
    cam, _ = make_camera(cap)
    cam.start()
    assert cam.read(timeout_s=1.0) is not None
    t0 = time.perf_counter()
    cam.close()
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0
    assert not cap.opened  # captura solta
    assert all(t.name != "inmotion-camera" for t in threading.enumerate())
    cam.close()  # idempotente


def test_thread_exception_is_reraised() -> None:
    cam, _ = make_camera(BoomCapture(fail_at=3))
    cam.start()
    try:
        with pytest.raises(CameraBoomError, match="simulada"):
            drain(cam)
    finally:
        cam.close()


def test_settings_applied() -> None:
    cap = FakeCapture()
    config = CameraConfig(
        index=2, backend="msmf", width=800, height=600, fps=30, fourcc="MJPG", exposure=-6.0
    )
    cam, calls = make_camera(cap, config)
    cam.start()
    try:
        assert calls == [(2, cv2.CAP_MSMF)]
        assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 800
        assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 600
        assert cap.props[cv2.CAP_PROP_FPS] == 30
        assert cap.props[cv2.CAP_PROP_EXPOSURE] == -6.0
        assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter.fourcc(*"MJPG")
        assert cam.actual == (800, 600, 30.0)
    finally:
        cam.close()


def test_exposure_not_set_when_none_and_backend_mapping() -> None:
    cap = FakeCapture()
    cam, calls = make_camera(cap, CameraConfig(backend="any"))
    cam.start()
    try:
        assert calls == [(CameraConfig().index, cv2.CAP_ANY)]
        assert cv2.CAP_PROP_EXPOSURE not in cap.props
    finally:
        cam.close()


def test_start_fails_if_capture_not_opened() -> None:
    cap = FakeCapture()
    cap.opened = False
    cam, _ = make_camera(cap)
    with pytest.raises(OSError, match="câmera"):
        cam.start()
    cam.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"index": -1},
        {"width": 0},
        {"height": -5},
        {"fps": 0},
        {"fourcc": "MJP"},
        {"backend": "xyz"},
        {"exposure": float("nan")},
    ],
)
def test_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"deve|inválido"):
        CameraConfig(**kwargs)


def test_default_camera_index_is_shared_by_app_and_tools() -> None:
    """Padrão provisório: índice 1 (o 0 era uma câmera virtual); um só lugar (``CameraConfig``)."""
    from inmotion.__main__ import build_parser as app_parser
    from tools.bench_pose import _build_parser as bench_parser
    from tools.record import _build_parser as record_parser

    default = CameraConfig().index
    assert default == 1
    assert app_parser().parse_args([]).camera == default
    assert bench_parser().parse_args([]).camera == default
    record_args = record_parser().parse_args(["--person", "p", "--out", "x.npz"])
    assert record_args.camera == default


def test_backend_names_match_schema() -> None:
    """O mapa para as constantes do OpenCV cobre exatamente os nomes aceitos pelo esquema."""
    assert set(_BACKENDS) == set(CAMERA_BACKENDS)
