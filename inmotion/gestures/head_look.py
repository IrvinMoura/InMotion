"""Olhar com a cabeça (spec 014): orientação pelos pontos do rosto e câmera por velocidade.

Converte o deslocamento normalizado do nariz (adimensional: deslocamento / distância
entre as orelhas) em velocidade do ponteiro (px/s), com zona morta, expoente e limite
(C1-C4), e integra ``v·Δt`` com resto fracionário (C5). Núcleo puro: o tempo é ``t``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import numpy as np

from inmotion.types import PoseFrame
from inmotion.vision.geometry import GeometryConfig, is_visible, to_pixels
from inmotion.vision.landmarks import PoseLandmark


@dataclass(frozen=True)
class ResponseCurve:
    """Curva ``v = sign(x)·min(k·max(0, |x| − d)^gamma, v_max)``.

    Propriedades: C1 ímpar, ``v(−x) = −v(x)``; C2 ``v = 0`` em ``|x| ≤ d`` e contínua em
    ``|x| = d``; C3 ``|v|`` não decrescente em ``|x|``; C4 ``|v| ≤ v_max``.

    Entradas não finitas: ``NaN`` vira ``0.0`` (dado inválido nunca move a câmera) e
    ``±inf`` vira ``±v_max``.
    """

    gain: float = 1200.0  # padrões da T0 (12/11/2026); os valores reais vêm do YAML
    dead_zone: float = 0.15
    gamma: float = 1.5
    v_max: float = 800.0

    def __post_init__(self) -> None:
        """Valida: ganho > 0, zona morta ≥ 0, gamma ≥ 1, v_max > 0, todos finitos."""
        values = (self.gain, self.dead_zone, self.gamma, self.v_max)
        if not all(math.isfinite(v) for v in values):
            raise ValueError("parâmetros da curva devem ser finitos")
        if self.gain <= 0 or self.v_max <= 0:
            raise ValueError("gain e v_max devem ser > 0")
        if self.dead_zone < 0:
            raise ValueError("dead_zone deve ser >= 0")
        if self.gamma < 1:
            raise ValueError("gamma deve ser >= 1")

    def __call__(self, x: float) -> float:
        """Velocidade (px/s) para o deslocamento normalizado ``x``."""
        if math.isnan(x):
            return 0.0
        mag = abs(x) - self.dead_zone
        if mag <= 0.0:
            return 0.0
        try:
            speed = min(self.gain * mag**self.gamma, self.v_max)
        except OverflowError:
            speed = self.v_max
        return math.copysign(speed, x)


@dataclass(frozen=True)
class HeadPose:
    """Orientação da cabeça, adimensional (deslocamento do nariz / distância das orelhas).

    ``yaw > 0``: cabeça virada para a direita da pessoa. ``pitch > 0``: para cima.
    """

    yaw: float
    pitch: float


_FACE = (
    PoseLandmark.NOSE,
    PoseLandmark.LEFT_EYE,
    PoseLandmark.RIGHT_EYE,
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
)


def head_pose(pose: PoseFrame, geometry: GeometryConfig, min_ear_px: float) -> HeadPose | None:
    """RF1: orientação da cabeça pelos pontos do rosto, em pixels (A5: ``p = diag(W, H, 1)·n``).

    - ``yaw = −(x_nariz − x̄_orelhas) / e``: o sinal é trocado porque, na imagem não
      espelhada (ADR 0007), virar para a direita da pessoa leva o nariz para a esquerda;
    - ``pitch = (ȳ_olhos − y_nariz) / e`` (o ``y`` da imagem cresce para baixo);
    - ``e = ‖orelha_e − orelha_d‖₂`` (A8).

    C6: as duas razões são adimensionais, invariantes a translação e escala na imagem
    (distância à câmera). ``None`` se algum ponto do rosto está abaixo de
    ``geometry.visibility_threshold`` ou se ``e < min_ear_px`` (de perfil, as orelhas se
    sobrepõem e a razão explodiria).
    """
    if not is_visible(pose, _FACE, geometry.visibility_threshold):
        return None
    w, h = pose.image_size
    px = to_pixels(pose.landmarks[list(_FACE), 0:2], w, h)
    nose, left_eye, right_eye, left_ear, right_ear = px
    ears = float(np.linalg.norm(left_ear - right_ear))
    if not (math.isfinite(ears) and ears >= min_ear_px):
        return None
    yaw = -(nose[0] - (left_ear[0] + right_ear[0]) / 2.0) / ears
    pitch = ((left_eye[1] + right_eye[1]) / 2.0 - nose[1]) / ears
    return HeadPose(yaw=float(yaw), pitch=float(pitch))


@dataclass(frozen=True)
class HeadLookParams:
    """Parâmetros da câmera (spec 014 T0): curvas por eixo, neutro e pitch.

    ``enabled``: liga a câmera pela cabeça no app; ``neutral_s``: segundos de amostras
    para o neutro (RF2); ``min_ear_px``: distância
    mínima entre as orelhas para a pose valer; ``pitch_enabled`` desligado por padrão (T0,
    decisão 2); ``invert_pitch`` troca o sentido vertical. Os valores reais vêm do YAML.
    """

    enabled: bool = True
    neutral_s: float = 1.0
    min_ear_px: float = 5.0
    yaw: ResponseCurve = ResponseCurve()
    pitch: ResponseCurve = ResponseCurve()
    pitch_enabled: bool = False
    invert_pitch: bool = False

    def __post_init__(self) -> None:
        """Valida: ``neutral_s`` e ``min_ear_px`` finitos e > 0."""
        if not (math.isfinite(self.neutral_s) and self.neutral_s > 0.0):
            raise ValueError(f"neutral_s deve ser finito e > 0: {self.neutral_s}")
        if not (math.isfinite(self.min_ear_px) and self.min_ear_px > 0.0):
            raise ValueError(f"min_ear_px deve ser finito e > 0: {self.min_ear_px}")


class HeadLook:
    """Câmera em controle de velocidade a partir da ``HeadPose`` de cada frame.

    - **Neutro (RF2):** depois de criado ou de ``reset()``, as poses dos primeiros
      ``neutral_s`` segundos são guardadas e o neutro ``(yaw₀, pitch₀)`` é a mediana delas;
      até lá devolve ``(0, 0)``.
    - **Velocidade (RF3):** ``v = curva(pose − neutro)`` por eixo (C1-C4). No eixo
      vertical o sinal é trocado (cabeça para cima → ponteiro para cima, ``dy < 0``),
      salvo ``invert_pitch``; com ``pitch_enabled`` falso, ``dy = 0``.
    - **Integração (C5):** ``Δ = v·Δt + r`` e ``r ← Δ − trunc(Δ)``, de modo que a soma dos
      ``(dx, dy)`` inteiros num intervalo fica a menos de 1 px de ``∫v dt``, qualquer que
      seja o FPS.
    - **Segurança (RF5, RF6):** sem pose (``None``) ou com ``suppress`` (agachar em curso),
      devolve ``(0, 0)`` e zera o resto; sem pose, o próximo frame recomeça com ``Δt = 0``,
      então o tempo sem pose nunca vira movimento.
    """

    def __init__(self, params: HeadLookParams) -> None:
        self._params = params
        self._samples: list[HeadPose] = []
        self._t0: float | None = None
        self._neutral: HeadPose | None = None
        self._last_t: float | None = None
        self._rest = [0.0, 0.0]

    @property
    def neutral(self) -> HeadPose | None:
        """O neutro medido, ou ``None`` enquanto ainda está medindo."""
        return self._neutral

    def update(self, head: HeadPose | None, t: float, *, suppress: bool = False) -> tuple[int, int]:
        """``(dx, dy)`` inteiros em pixels para o frame em ``t`` (segundos)."""
        if head is None:
            self._rest = [0.0, 0.0]
            self._last_t = None
            return (0, 0)
        if suppress:
            self._rest = [0.0, 0.0]
            self._last_t = t
            return (0, 0)
        if self._neutral is None:
            if self._t0 is None:
                self._t0 = t
            if t - self._t0 < self._params.neutral_s or not self._samples:
                self._samples.append(head)
                self._last_t = t
                return (0, 0)
            self._neutral = HeadPose(
                yaw=statistics.median(s.yaw for s in self._samples),
                pitch=statistics.median(s.pitch for s in self._samples),
            )
            self._samples = []
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        p = self._params
        vx = p.yaw(head.yaw - self._neutral.yaw)
        vy = 0.0
        if p.pitch_enabled:
            vy = p.pitch(head.pitch - self._neutral.pitch)
            vy = vy if p.invert_pitch else -vy
        return self._integrate(0, vx * dt), self._integrate(1, vy * dt)

    def reset(self) -> None:
        """Esquece neutro, amostras, resto e último ``t`` (volta a ACTIVE: mede de novo)."""
        self._samples = []
        self._t0 = None
        self._neutral = None
        self._last_t = None
        self._rest = [0.0, 0.0]

    def _integrate(self, axis: int, delta: float) -> int:
        """C5: soma o resto do eixo, devolve a parte inteira e guarda a fração."""
        total = delta + self._rest[axis]
        whole = math.trunc(total)
        self._rest[axis] = total - whole
        return whole
