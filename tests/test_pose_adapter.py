"""Testes do adaptador MediaPipe com landmarker falso (spec 003, CA2 a CA4)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from inmotion.types import PoseFrame
from inmotion.vision.pose import (
    POSE_CONNECTIONS,
    MediaPipePoseEstimator,
    ModelNotFoundError,
    PoseConfig,
)


def _point(i: int) -> SimpleNamespace:
    """Landmark imitando ``NormalizedLandmark``/``Landmark`` do mediapipe Tasks."""
    return SimpleNamespace(
        x=i / 100.0,
        y=i / 50.0,
        z=-i / 200.0,
        visibility=1.0 - i / 100.0,
        presence=0.5 + i / 100.0,
    )


def fake_result(people: int = 1) -> SimpleNamespace:
    """Imita ``PoseLandmarkerResult``: listas (uma por pessoa) de 33 landmarks."""
    return SimpleNamespace(
        pose_landmarks=[[_point(i) for i in range(33)] for _ in range(people)],
        pose_world_landmarks=[[_point(i) for i in range(33)] for _ in range(people)],
        segmentation_masks=None,
    )


class FakeLandmarker:
    """Dublê de ``PoseLandmarker`` no modo VIDEO: registra timestamps e devolve o resultado."""

    def __init__(self, result: SimpleNamespace | None = None) -> None:
        self.result = result if result is not None else fake_result()
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image: Any, timestamp_ms: int) -> SimpleNamespace:
        self.timestamps.append(timestamp_ms)
        return self.result

    def close(self) -> None:
        self.closed = True


def make_estimator(tmp_path: Path, landmarker: FakeLandmarker) -> MediaPipePoseEstimator:
    return MediaPipePoseEstimator(PoseConfig(model_dir=tmp_path), lambda _cfg: landmarker)


def test_result_maps_to_pose_frame(tmp_path: Path) -> None:
    est = make_estimator(tmp_path, FakeLandmarker())
    img = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = est.estimate(img, 1.5)
    assert frame is not None
    assert frame.t == 1.5
    assert frame.image_size == (64, 48)
    assert frame.landmarks.shape == (33, 5)
    assert frame.world is not None
    assert frame.world.shape == (33, 4)
    expected = np.array([[p.x, p.y, p.z, p.visibility, p.presence] for p in map(_point, range(33))])
    np.testing.assert_allclose(frame.landmarks, expected)
    np.testing.assert_allclose(frame.world, expected[:, :4])


def test_no_pose_returns_none(tmp_path: Path) -> None:
    empty = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)
    est = make_estimator(tmp_path, FakeLandmarker(empty))
    assert est.estimate(np.zeros((8, 8, 3), dtype=np.uint8), 0.0) is None


def test_timestamps_strictly_increasing(tmp_path: Path) -> None:
    fake = FakeLandmarker()
    est = make_estimator(tmp_path, fake)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    for t in [0.0, 0.033, 0.033, 0.020, 0.066]:
        est.estimate(img, t)
    assert fake.timestamps == [0, 33, 34, 35, 66]


def test_missing_model_raises_with_instructions(tmp_path: Path) -> None:
    config = PoseConfig(model_dir=tmp_path / "vazio", variant="lite")
    with pytest.raises(ModelNotFoundError) as info:
        MediaPipePoseEstimator(config)
    message = str(info.value)
    assert "pose_landmarker_lite.task" in message
    assert "tools/download_model.py" in message
    assert "--variant lite" in message


def test_estimate_does_not_mutate_input_and_sets_size_from_image(tmp_path: Path) -> None:
    est = make_estimator(tmp_path, FakeLandmarker())
    img = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    before = img.copy()
    frame = est.estimate(img, 0.0)
    assert frame is not None
    assert frame.image_size == (6, 4)
    np.testing.assert_array_equal(img, before)


def test_uses_first_person_only(tmp_path: Path) -> None:
    result = fake_result(people=2)
    for p in result.pose_landmarks[1]:
        p.x = 0.99
    est = make_estimator(tmp_path, FakeLandmarker(result))
    frame = est.estimate(np.zeros((8, 8, 3), dtype=np.uint8), 0.0)
    assert frame is not None
    assert frame.landmarks[0, 0] == pytest.approx(0.0)


def test_timestamp_warning_at_most_once_per_second(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    est = make_estimator(tmp_path, FakeLandmarker())
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    with caplog.at_level(logging.WARNING, logger="inmotion.vision.pose"):
        est.estimate(img, 1.0)
        for _ in range(5):  # cinco repetições dentro de 1 s de t
            est.estimate(img, 1.0)
        assert len(caplog.records) == 1
        est.estimate(img, 2.5)
        est.estimate(img, 2.5)  # repetição 1,5 s depois do primeiro aviso
        assert len(caplog.records) == 2


def test_close_is_idempotent_and_context_manager(tmp_path: Path) -> None:
    fake = FakeLandmarker()
    with make_estimator(tmp_path, fake) as est:
        assert not fake.closed
    assert fake.closed
    est.close()  # segunda chamada não falha


def test_injected_factory_skips_model_check(tmp_path: Path) -> None:
    config = PoseConfig(model_dir=tmp_path / "inexistente")
    assert not config.model_path.exists()
    MediaPipePoseEstimator(config, lambda _cfg: FakeLandmarker())  # não levanta


def test_pose_config_model_path_and_validation() -> None:
    assert PoseConfig(variant="lite").model_path == Path("models") / "pose_landmarker_lite.task"
    with pytest.raises(ValueError, match="variant"):
        PoseConfig(variant="ultra")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_tracking_confidence"):
        PoseConfig(min_tracking_confidence=1.5)
    with pytest.raises(ValueError, match="min_detection_confidence"):
        PoseConfig(min_detection_confidence=float("nan"))


def test_pose_connections_are_valid_index_pairs() -> None:
    assert len(set(POSE_CONNECTIONS)) == len(POSE_CONNECTIONS) > 0
    assert all(0 <= a < 33 and 0 <= b < 33 and a != b for a, b in POSE_CONNECTIONS)


def test_pose_connections_match_mediapipe_topology() -> None:
    from mediapipe.tasks.python import vision

    official = {(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS}
    assert set(POSE_CONNECTIONS) == official


# --- Hardware (spec 003 T6): webcam + modelo real; rodar com `pytest -m hardware -s` ---

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
_HW_WAIT_S = 15.0  # tempo para a pessoa se posicionar
_HW_READ_TIMEOUT_S = 1.0
_HW_EMPTY_HOLD_S = 2.0  # cena vazia: sem pessoa por este tempo seguido
_HW_MIN_FRAMES = 10  # mínimo de frames para checar que o vídeo é ao vivo
# Sobrescreve o índice padrão; câmeras virtuais (Snap Camera, NVIDIA Broadcast) ocupam índices.
_HW_CAMERA_ENV = "INMOTION_CAMERA"
_SHOULDERS = (11, 12)
_MIN_SHOULDER_VISIBILITY = 0.5


def _wait_for_pose(
    capsys: pytest.CaptureFixture[str], prompt: str, want_person: bool
) -> PoseFrame | None:
    """Lê a webcam até a condição valer (ou o prazo acabar); devolve a última pose lida.

    Com ``want_person=False`` a cena precisa ficar vazia por ``_HW_EMPTY_HOLD_S``. Falha
    se todos os frames forem idênticos: imagem parada de câmera virtual não vale.
    """
    import os

    from inmotion.capture.camera import DEFAULT_CAMERA_INDEX, Camera, CameraConfig
    from inmotion.clock import MonotonicClock

    config = PoseConfig(model_dir=_MODEL_DIR)
    if not config.model_path.is_file():
        pytest.skip(f"modelo ausente: {config.model_path}")
    index = int(os.environ.get(_HW_CAMERA_ENV, str(DEFAULT_CAMERA_INDEX)))
    clock = MonotonicClock()
    with capsys.disabled():
        print(f"\n>>> {prompt} (câmera {index}, até {_HW_WAIT_S:.0f} s)")
    camera = Camera(CameraConfig(index=index), clock)
    camera.start()
    pose: PoseFrame | None = None
    first: np.ndarray[Any, Any] | None = None
    live = False
    frames = 0
    empty_since: float | None = None
    try:
        with MediaPipePoseEstimator(config) as estimator:
            deadline = clock.now() + _HW_WAIT_S
            while clock.now() < deadline:
                packet = camera.read(timeout_s=_HW_READ_TIMEOUT_S)
                if packet is None:
                    continue
                frames += 1
                if first is None:
                    first = packet.image.copy()
                elif not live:
                    live = not np.array_equal(first, packet.image)
                pose = estimator.estimate(packet.image, packet.t)
                if want_person:
                    done = pose is not None
                else:
                    empty_since = None if pose is not None else (empty_since or packet.t)
                    done = empty_since is not None and packet.t - empty_since >= _HW_EMPTY_HOLD_S
                if done and frames >= _HW_MIN_FRAMES:
                    break
    finally:
        camera.close()
    assert live, (
        f"câmera {index} entregou só frames idênticos (câmera virtual?); "
        f"defina {_HW_CAMERA_ENV} com o índice da webcam real"
    )
    return pose


@pytest.mark.hardware
def test_real_model_detects_person(capsys: pytest.CaptureFixture[str]) -> None:
    """CA5: pessoa em pé -> 33 landmarks e ombros com visibility > 0,5."""
    pose = _wait_for_pose(capsys, "Fique em pé, de frente para a câmera", want_person=True)
    assert pose is not None
    assert pose.landmarks.shape == (33, 5)
    visibility = pose.landmarks[list(_SHOULDERS), 3]
    with capsys.disabled():
        print(f"visibility dos ombros: {visibility}")
    assert (visibility > _MIN_SHOULDER_VISIBILITY).all()


@pytest.mark.hardware
def test_real_model_empty_scene(capsys: pytest.CaptureFixture[str]) -> None:
    """CA6: cena sem pessoa -> None."""
    pose = _wait_for_pose(capsys, "Saia do quadro da câmera", want_person=False)
    assert pose is None
