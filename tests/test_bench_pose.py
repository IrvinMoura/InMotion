"""Testes do benchmark de pose com fakes (spec 003 T7)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from inmotion.types import FramePacket, PoseFrame
from tests.fakes import FakeClock, make_pose
from tools.bench_pose import BenchResult, bench_variant


class EndlessSource:
    """Fonte que entrega um frame novo a cada leitura, avançando o relógio 10 ms."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self._index = 0
        self._image = np.zeros((4, 4, 3), dtype=np.uint8)

    def read(self, timeout_s: float) -> FramePacket | None:
        self._clock.advance(0.010)
        self._index += 1
        return FramePacket(self._clock.now(), self._index, self._image)

    def close(self) -> None:
        pass


class TimedEstimator:
    """Estimador que gasta 20 ms de relógio falso e detecta em frames pares."""

    def __init__(self, clock: FakeClock) -> None:
        self._clock = clock
        self.calls = 0

    def estimate(self, image_bgr: NDArray[np.uint8], t: float) -> PoseFrame | None:
        self.calls += 1
        self._clock.advance(0.020)
        return make_pose({}, t=t) if self.calls % 2 == 0 else None

    def close(self) -> None:
        pass


def test_bench_variant_measures_fps_inference_and_detection() -> None:
    clock = FakeClock(t0=0.0)
    estimator = TimedEstimator(clock)
    result = bench_variant(
        "lite", EndlessSource(clock), estimator, clock, duration_s=2.995, warmup_frames=2
    )
    assert isinstance(result, BenchResult)
    assert result.frames == 100  # 30 ms por frame; 2,995 s evita a fronteira em ponto flutuante
    assert result.fps == pytest.approx(1 / 0.030, rel=0.02)
    assert result.infer_mean_ms == pytest.approx(20.0)
    assert result.infer_p95_ms == pytest.approx(20.0)
    assert result.detection_rate == pytest.approx(0.5)
    assert estimator.calls == 102  # 2 de aquecimento ficam fora
    assert result.row().startswith("| lite | 100 |")


def test_bench_variant_rejects_non_positive_duration() -> None:
    clock = FakeClock(t0=0.0)
    with pytest.raises(ValueError, match="duration_s"):
        bench_variant("lite", EndlessSource(clock), TimedEstimator(clock), clock, 0.0)


def test_benchmark_without_frames_times_out_during_warmup() -> None:
    clock = FakeClock()

    class NoFrames(EndlessSource):
        def read(self, timeout_s: float) -> FramePacket | None:
            clock.advance(1.0)
            return None

    with pytest.raises(OSError, match="nenhum frame"):
        bench_variant("lite", NoFrames(clock), TimedEstimator(clock), clock, 1.0)
