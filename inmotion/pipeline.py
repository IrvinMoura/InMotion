"""Pipeline puro por frame: segurança, gestos e FPS. Núcleo puro (tempo via ``t``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inmotion.exit_zone import ExitZoneState
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.legacy import LegacyGesture
from inmotion.safety import SafetyManager, SafetyState
from inmotion.types import PoseFrame
from inmotion.vision.geometry import is_visible
from inmotion.vision.landmarks import PoseLandmark

# Ombros e quadris: sem eles visíveis a pose é fantasma/parcial e não conta como tracking.
TORSO: tuple[int, ...] = (
    PoseLandmark.LEFT_SHOULDER,
    PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_HIP,
)


@dataclass(frozen=True)
class StepResult:
    """O que o frame produziu; o overlay só lê, nunca decide."""

    t: float
    fps: float | None
    safety_state: SafetyState
    pose: PoseFrame | None
    exit_zone: ExitZoneState | None = None


class Pipeline:
    """Um passo por frame: ``on_frame`` → filtro → gestos (só em ACTIVE) → FPS.

    O ``pose_filter`` opcional (spec 006) suaviza a pose antes dos gestos; o ``StepResult``
    carrega a pose filtrada.
    """

    def __init__(
        self,
        safety: SafetyManager,
        gestures: Sequence[LegacyGesture],
        fps: FpsMeter,
        pose_filter: PoseFilter | None = None,
    ) -> None:
        self._safety = safety
        self._gestures = tuple(gestures)
        self._fps = fps
        self._pose_filter = pose_filter
        self._was_active = True  # gestos nascem zerados
        self._tracked_since: float | None = None  # início da sequência contínua de pose

    def step(self, pose: PoseFrame | None, t: float) -> StepResult:
        """on_frame → (se ACTIVE) gestos → FPS. Ao voltar para ACTIVE, reset() nos gestos.

        ``tracking_ok`` exige pose com o tronco visível (``min_torso_visibility`` em
        ombros e quadris): ao sair do quadro o MediaPipe pode devolver uma pose
        "fantasma" de baixa visibilidade, que não pode segurar teclas. Os gestos só
        rodam com pose rastreada e em ACTIVE; o ``StepResult`` ainda leva a pose crua
        não rastreada, para o overlay.

        Persistência (debounce R5 no tempo): a pose só vale depois de ``min_track_s``
        de detecção contínua, ``t − t₀ ≥ min_track_s``, com ``t₀`` o início da sequência
        atual; um quadro sem pose zera ``t₀``. Fantasmas piscam e nunca chegam lá.
        na volta de LOST/PAUSED para ACTIVE o estado deles é zerado antes do primeiro
        ``update``, pois o ``release_all()`` já soltou as teclas que eles achavam apertadas.
        """
        raw = pose
        config = self._safety.config
        if pose is not None and not is_visible(pose, TORSO, config.min_torso_visibility):
            pose = None
        if pose is None:
            self._tracked_since = None
        else:
            if self._tracked_since is None:
                self._tracked_since = t
            if t - self._tracked_since < config.min_track_s:
                pose = None
        self._safety.on_frame(t, pose is not None)
        state = self._safety.state
        active = state is SafetyState.ACTIVE
        if active and not self._was_active:
            for gesture in self._gestures:
                gesture.reset()
        self._was_active = active
        if self._pose_filter is not None:
            if pose is not None and active:
                pose = self._pose_filter(pose)
            else:
                self._pose_filter.reset()  # sem pose ou fora de ACTIVE: recomeça limpo
        if active and pose is not None:
            for gesture in self._gestures:
                gesture.update(pose, self._safety.input)
        fps = self._fps.update(t)
        return StepResult(t=t, fps=fps, safety_state=state, pose=raw if pose is None else pose)
