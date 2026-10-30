"""Mede o FPS do laço câmera -> PoseLandmarker para cada variante (spec 003 T7, CA7).

Uso::

    python -m tools.bench_pose --seconds 10

Fique em pé na frente da câmera durante a medição. Imprime uma tabela Markdown
(variante, frames, FPS, inferência média e p95, taxa de detecção) para colar na spec.

Ferramenta de composição: monta os adaptadores reais; a medição testável fica em
``bench_variant``, que recebe os ports por injeção.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from inmotion.capture.camera import DEFAULT_CAMERA_INDEX, Camera, CameraConfig
from inmotion.clock import MonotonicClock
from inmotion.ports import Clock, FrameSource, PoseEstimator
from inmotion.vision.pose import MediaPipePoseEstimator, ModelNotFoundError, PoseConfig

logger = logging.getLogger("bench_pose")

VARIANTS: tuple[str, ...] = ("lite", "full", "heavy")
READ_TIMEOUT_S = 0.5
WARMUP_FRAMES = 5  # a primeira inferência inclui inicialização do XNNPACK
_MS_PER_S = 1000.0
_P95_QUANTILES = 20


@dataclass(frozen=True)
class BenchResult:
    """Resultado de uma variante. FPS = frames processados / tempo de parede."""

    variant: str
    frames: int
    fps: float
    infer_mean_ms: float
    infer_p95_ms: float
    detection_rate: float

    def row(self) -> str:
        """Linha da tabela Markdown."""
        return (
            f"| {self.variant} | {self.frames} | {self.fps:.1f} | {self.infer_mean_ms:.1f} | "
            f"{self.infer_p95_ms:.1f} | {self.detection_rate:.0%} |"
        )


TABLE_HEADER = (
    "| Variante | Frames | FPS | Inferência média (ms) | Inferência p95 (ms) | Detecção |\n"
    "|---|---|---|---|---|---|"
)


def bench_variant(
    variant: str,
    source: FrameSource,
    estimator: PoseEstimator,
    clock: Clock,
    duration_s: float,
    warmup_frames: int = WARMUP_FRAMES,
) -> BenchResult:
    """Processa frames por ``duration_s`` (após o aquecimento) e mede FPS e inferência."""
    if duration_s <= 0:
        raise ValueError(f"duration_s deve ser > 0: {duration_s}")
    infer_ms: list[float] = []
    detected = 0
    warm = 0
    start: float | None = None
    end = 0.0
    while start is None or clock.now() - start < duration_s:
        packet = source.read(READ_TIMEOUT_S)
        if packet is None:
            continue
        before = clock.now()
        pose = estimator.estimate(packet.image, packet.t)
        after = clock.now()
        if warm < warmup_frames:
            warm += 1
            if warm == warmup_frames:
                start = after
            continue
        if start is None:
            start = before
        infer_ms.append((after - before) * _MS_PER_S)
        detected += pose is not None
        end = after
    frames = len(infer_ms)
    elapsed = end - start if frames else 0.0
    return BenchResult(
        variant=variant,
        frames=frames,
        fps=frames / elapsed if elapsed > 0 else 0.0,
        infer_mean_ms=statistics.fmean(infer_ms) if infer_ms else 0.0,
        infer_p95_ms=(statistics.quantiles(infer_ms, n=_P95_QUANTILES)[-1] if frames >= 2 else 0.0),
        detection_rate=detected / frames if frames else 0.0,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark de FPS das variantes do modelo.")
    parser.add_argument("--seconds", type=float, default=10.0, help="duração por variante")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="índice da câmera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Roda o benchmark de cada variante e imprime a tabela. 0 = ok, 2 = modelo ausente."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    clock = MonotonicClock()
    camera_config = CameraConfig(index=args.camera, width=args.width, height=args.height)
    results: list[BenchResult] = []
    for variant in args.variants:
        try:
            estimator = MediaPipePoseEstimator(
                PoseConfig(model_dir=args.models_dir, variant=variant)
            )
        except ModelNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        camera = Camera(camera_config, clock)
        try:
            camera.start()
            logger.info("medindo %s por %.0f s (câmera %s)", variant, args.seconds, camera.actual)
            results.append(bench_variant(variant, camera, estimator, clock, args.seconds))
        finally:
            camera.close()
            estimator.close()
    lines = [TABLE_HEADER, *(r.row() for r in results)]
    logger.info("resultado:\n%s", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
