"""Verifica se o ambiente está pronto para rodar o InMotion (spec 015).

Uso::

    python -m tools.check_env [--skip-camera] [--model-variant full]

Confere:
1. Python 3.10 (``>=3.10, <3.11``);
2. Dependências obrigatórias (``cv2``, ``mediapipe``, ``numpy``, ``yaml``, ``pydirectinput``);
3. Modelo ``.task`` do MediaPipe existente e não vazio;
4. Sistema operacional / backend de input Windows (``pydirectinput``);
5. Acesso à webcam configurada (com fallback automático entre índices 1 e 0).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from inmotion.capture.camera import Camera
from inmotion.clock import MonotonicClock
from inmotion.config.loader import DEFAULT_CONFIG_PATH, load_config
from inmotion.config.schema import (
    CAMERA_BACKENDS,
    POSE_VARIANTS,
    CameraBackend,
    CameraConfig,
    PoseConfig,
    Variant,
)

logger = logging.getLogger("check_env")

REQUIRED_MODULES: tuple[str, ...] = ("cv2", "mediapipe", "numpy", "yaml", "pydirectinput")


@dataclass(frozen=True)
class CheckResult:
    """Resultado individual de uma checagem de ambiente."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class EnvHooks:
    """Pontos de injeção para testar ``check_env`` sem hardware real."""

    version_info: tuple[int, int, int] = (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    platform: str = sys.platform
    import_module: Callable[[str], object] = importlib.import_module
    probe_camera: Callable[[CameraConfig], tuple[int, tuple[int, int, float]]] | None = None


def _default_probe_camera(cfg: CameraConfig) -> tuple[int, tuple[int, int, float]]:
    cam = Camera(cfg, MonotonicClock())
    cam.start()
    try:
        return cam.resolved_index, cam.actual
    finally:
        cam.close()


def check_python_version(version_info: tuple[int, int, int]) -> CheckResult:
    """Exige Python 3.10.x (compatível com MediaPipe e deps fixadas)."""
    major, minor, micro = version_info
    ok = (major, minor) == (3, 10)
    detail = f"Python {major}.{minor}.{micro}"
    if not ok:
        detail += " (esperado 3.10.x)"
    return CheckResult("python", ok, detail)


def check_dependencies(
    import_fn: Callable[[str], object],
    modules: Sequence[str] = REQUIRED_MODULES,
) -> CheckResult:
    """Tenta importar cada biblioteca obrigatória sem alterar estado."""
    missing: list[str] = []
    for mod in modules:
        try:
            import_fn(mod)
        except Exception as exc:
            missing.append(f"{mod} ({exc})")
    if missing:
        return CheckResult("dependencias", False, "faltando/erro: " + "; ".join(missing))
    return CheckResult("dependencias", True, f"ok ({', '.join(modules)})")


def check_platform_input(platform: str, import_fn: Callable[[str], object]) -> CheckResult:
    """Garante Windows + PyDirectInput disponível para injeção de teclas/mouse."""
    if platform != "win32":
        return CheckResult(
            "input_windows",
            False,
            f"plataforma {platform!r}; injeção real requer Windows (win32)",
        )
    try:
        import_fn("pydirectinput")
    except Exception as exc:
        return CheckResult("input_windows", False, f"pydirectinput indisponível: {exc}")
    return CheckResult("input_windows", True, "Windows + PyDirectInput prontos")


def check_model_file(pose_cfg: PoseConfig) -> CheckResult:
    """Confere presença e tamanho não nulo do modelo ``.task`` configurado."""
    path = pose_cfg.model_path
    if not path.is_file():
        hint = f"python tools/download_model.py --variant {pose_cfg.variant}"
        return CheckResult("modelo_pose", False, f"{path} ausente (execute: {hint})")
    size = path.stat().st_size
    if size <= 0:
        return CheckResult("modelo_pose", False, f"{path} está vazio (0 bytes)")
    mb = size / (1024 * 1024)
    return CheckResult("modelo_pose", True, f"{path} ({mb:.1f} MB)")


def check_camera(
    cam_cfg: CameraConfig,
    probe: Callable[[CameraConfig], tuple[int, tuple[int, int, float]]],
) -> CheckResult:
    """Tenta abrir a câmera (com fallback 1<->0 do adaptador) e lê dimensões aceitas."""
    try:
        resolved_index, (w, h, fps) = probe(cam_cfg)
    except Exception as exc:
        return CheckResult("camera", False, f"falha ao abrir câmera ({exc})")
    fallback_note = f" [fallback de {cam_cfg.index}]" if resolved_index != cam_cfg.index else ""
    return CheckResult(
        "camera",
        True,
        f"índice {resolved_index} ({cam_cfg.backend}): {w}x{h}@{fps:.0f}fps{fallback_note}",
    )


def run_checks(
    pose_cfg: PoseConfig,
    cam_cfg: CameraConfig,
    *,
    skip_camera: bool = False,
    hooks: EnvHooks | None = None,
) -> list[CheckResult]:
    """Executa todas as verificações e devolve a lista estruturada."""
    h = hooks if hooks is not None else EnvHooks()
    results = [
        check_python_version(h.version_info),
        check_dependencies(h.import_module),
        check_platform_input(h.platform, h.import_module),
        check_model_file(pose_cfg),
    ]
    if skip_camera:
        results.append(CheckResult("camera", True, "pulada (--skip-camera)"))
    else:
        probe = h.probe_camera if h.probe_camera is not None else _default_probe_camera
        results.append(check_camera(cam_cfg, probe))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verifica pré-requisitos do InMotion.")
    parser.add_argument("--config", type=Path, default=None, help="YAML base opcional")
    parser.add_argument("--model-dir", type=Path, default=None, help="pasta dos modelos .task")
    parser.add_argument("--model-variant", choices=POSE_VARIANTS, default=None, help="variante")
    parser.add_argument("--camera", type=int, default=None, help="índice da câmera")
    parser.add_argument("--camera-backend", choices=CAMERA_BACKENDS, default=None, help="backend")
    parser.add_argument(
        "--skip-camera",
        action="store_true",
        help="não tenta abrir a webcam (útil em ambiente sem hardware)",
    )
    return parser


def main(argv: Sequence[str] | None = None, *, hooks: EnvHooks | None = None) -> int:
    """Executa ``check_env``; devolve 0 se tudo passou ou 1 se algo falhou."""
    args = _build_parser().parse_args(argv)
    config_path: Path = args.config if args.config is not None else DEFAULT_CONFIG_PATH
    base_cfg, _ = load_config(default_path=config_path)
    variant: Variant = (
        cast(Variant, args.model_variant)
        if args.model_variant is not None
        else base_cfg.pose.variant
    )
    pose_cfg = PoseConfig(
        model_dir=args.model_dir if args.model_dir is not None else base_cfg.pose.model_dir,
        variant=variant,
        min_detection_confidence=base_cfg.pose.min_detection_confidence,
        min_presence_confidence=base_cfg.pose.min_presence_confidence,
        min_tracking_confidence=base_cfg.pose.min_tracking_confidence,
    )
    backend: CameraBackend | None = cast(CameraBackend | None, args.camera_backend)
    cam_cfg = replace(
        base_cfg.camera,
        index=args.camera if args.camera is not None else base_cfg.camera.index,
        backend=backend if backend is not None else base_cfg.camera.backend,
    )
    results = run_checks(pose_cfg, cam_cfg, skip_camera=args.skip_camera, hooks=hooks)
    for item in results:
        tag = "OK" if item.ok else "FALHOU"
        level = logging.INFO if item.ok else logging.ERROR
        logger.log(level, "[%s] %-14s %s", tag, item.name, item.detail)
    return 0 if all(item.ok for item in results) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
