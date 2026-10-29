"""Pipeline puro por frame: segurança, gestos e FPS. Núcleo puro (tempo via ``t``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inmotion.fps import FpsMeter
from inmotion.gestures.legacy import LegacyGesture
from inmotion.safety import SafetyManager, SafetyState
from inmotion.types import PoseFrame


@dataclass(frozen=True)
class StepResult:
    """O que o frame produziu; o overlay só lê, nunca decide."""

    t: float
    fps: float | None
    safety_state: SafetyState
    pose: PoseFrame | None


class Pipeline:
    """Um passo por frame: ``on_frame`` → gestos (só em ACTIVE) → FPS."""

    def __init__(
        self, safety: SafetyManager, gestures: Sequence[LegacyGesture], fps: FpsMeter
    ) -> None:
        self._safety = safety
        self._gestures = tuple(gestures)
        self._fps = fps
        self._was_active = True  # gestos nascem zerados

    def step(self, pose: PoseFrame | None, t: float) -> StepResult:
        """on_frame → (se ACTIVE) gestos → FPS. Ao voltar para ACTIVE, reset() nos gestos.

        ``tracking_ok`` é ``pose is not None``. Os gestos só rodam com pose e em ACTIVE;
        na volta de LOST/PAUSED para ACTIVE o estado deles é zerado antes do primeiro
        ``update``, pois o ``release_all()`` já soltou as teclas que eles achavam apertadas.
        """
        self._safety.on_frame(t, pose is not None)
        state = self._safety.state
        active = state is SafetyState.ACTIVE
        if active and not self._was_active:
            for gesture in self._gestures:
                gesture.reset()
        self._was_active = active
        if active and pose is not None:
            for gesture in self._gestures:
                gesture.update(pose, self._safety.input)
        fps = self._fps.update(t)
        return StepResult(t=t, fps=fps, safety_state=state, pose=pose)
