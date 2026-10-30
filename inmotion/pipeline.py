"""Pipeline puro por frame: segurança, gestos, árbitro, mapeamento e FPS (tempo via ``t``)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inmotion.config.schema import GameProfile
from inmotion.exit_zone import ExitZoneState
from inmotion.filters.pose_filter import PoseFilter
from inmotion.fps import FpsMeter
from inmotion.gestures.body import BodyGesture, all_idle
from inmotion.gestures.factory import build_arbiter
from inmotion.gestures.features import BaselineParams, BodyBaseline, build_context
from inmotion.mapping import ActionMapper, MapperParams
from inmotion.safety import SafetyManager, SafetyState
from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig, is_visible
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
    """Um passo por frame: ``on_frame`` → filtro → contexto → gestos → árbitro → mapper → FPS.

    O ``pose_filter`` opcional (spec 006) suaviza a pose antes dos gestos; o ``StepResult``
    carrega a pose filtrada. O contexto (spec 008) calcula uma vez por frame o esqueleto
    normalizado (A6) e a base adaptativa R7; os gestos de corpo são features escalares com
    histerese (R5, R6); o ``GestureArbiter`` aplica exclusividade e o combo ``both_arms``;
    o ``ActionMapper`` converte os eventos em input pelo ``profile``.
    """

    def __init__(
        self,
        safety: SafetyManager,
        gestures: Sequence[BodyGesture],
        fps: FpsMeter,
        pose_filter: PoseFilter | None = None,
        *,
        profile: GameProfile,
        mapper_params: MapperParams | None = None,
        baseline_params: BaselineParams | None = None,
        geometry: GeometryConfig | None = None,
    ) -> None:
        self._safety = safety
        self._gestures = tuple(gestures)
        self._fps = fps
        self._pose_filter = pose_filter
        self._geometry = geometry if geometry is not None else GeometryConfig()
        self._baseline = BodyBaseline(baseline_params or BaselineParams())
        self._arbiter, self._combo = build_arbiter([g.name for g in self._gestures])
        self._mapper = ActionMapper(profile, safety.input, mapper_params or MapperParams())
        self._was_active = True  # gestos nascem zerados
        self._tracked_since: float | None = None  # início da sequência contínua de pose

    def step(self, pose: PoseFrame | None, t: float) -> StepResult:
        """on_frame → (se ACTIVE) gestos → árbitro → mapper → FPS.

        ``tracking_ok`` exige pose com o tronco visível (``min_torso_visibility`` em
        ombros e quadris): ao sair do quadro o MediaPipe pode devolver uma pose
        "fantasma" de baixa visibilidade, que não pode segurar teclas. Os gestos só
        rodam com pose rastreada e em ACTIVE; o ``StepResult`` ainda leva a pose crua
        não rastreada, para o overlay.

        Persistência (debounce R5 no tempo): a pose só vale depois de ``min_track_s``
        de detecção contínua, ``t − t₀ ≥ min_track_s``, com ``t₀`` o início da sequência
        atual; um quadro sem pose zera ``t₀``. Fantasmas piscam e nunca chegam lá. Pose
        rejeitada não chega a nenhum gesto nem à base R7.

        Na volta de LOST/PAUSED para ACTIVE o estado de gestos, árbitro, mapper e base é
        zerado antes do primeiro ``update``, pois o ``release_all()`` já soltou as teclas
        que eles achavam apertadas. Fora de ACTIVE o mapper esquece os taps pendentes; em
        ACTIVE sem pose válida ele ainda solta os taps vencidos por tempo.
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
            self._reset_all()
        self._was_active = active
        if not active:
            self._mapper.reset()
        if self._pose_filter is not None:
            if pose is not None and active:
                pose = self._pose_filter(pose)
            else:
                self._pose_filter.reset()  # sem pose ou fora de ACTIVE: recomeça limpo
        if active:
            if pose is not None:
                self._run_gestures(pose, t)
            else:
                self._mapper.update(t)
        fps = self._fps.update(t)
        return StepResult(t=t, fps=fps, safety_state=state, pose=raw if pose is None else pose)

    def _reset_all(self) -> None:
        for gesture in self._gestures:
            gesture.reset()
        self._arbiter.reset()
        if self._combo is not None:
            self._combo.reset()
        self._mapper.reset()
        self._baseline.reset()

    def _run_gestures(self, pose: PoseFrame, t: float) -> None:
        """Repouso (estado do frame anterior) → contexto → gestos → árbitro → mapper."""
        resting = all_idle(self._gestures)
        ctx = build_context(pose, t, self._geometry, self._baseline, resting)
        events = [e for g in self._gestures if (e := g.update(ctx)) is not None]
        if self._combo is not None:
            events.extend(self._combo.derive(events, t))
        self._mapper.handle(self._arbiter.resolve(events, t), t)
