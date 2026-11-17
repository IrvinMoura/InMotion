"""Poses sintéticas com world landmarks para os testes dos gestos de corpo (spec 008).

Convenção do world do MediaPipe (origem no centro do quadril, metros): x cresce para a
direita da imagem (o ombro ESQUERDO da pessoa tem x maior), y cresce para baixo e z
cresce para longe da câmera (negativo = à frente do quadril, em direção à câmera).
Os pixels vêm do world por projeção ortográfica: ``px = c + k·(x, y)`` com
``k = largura_dos_ombros_px / SHOULDER_WIDTH_M``, de modo que ``s`` em pixels vale
exatamente ``100·scale``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from inmotion.types import N_LANDMARKS, PoseFrame
from inmotion.vision.landmarks import PoseLandmark

Vec3 = tuple[float, float, float]

SHOULDER_WIDTH_M = 0.40
HIP_WIDTH_M = 0.20
TORSO_M = 0.60
UPPER_ARM_M = 0.30
FOREARM_M = 0.27
THIGH_M = 0.45  # quadril -> joelho em pé (world, eixo y)
SHOULDER_PX_AT_SCALE_1 = 100.0


@dataclass(frozen=True)
class Arm:
    """Braço por direções (world) do braço e do antebraço; o comprimento é fixo."""

    upper: Vec3
    fore: Vec3


ARM_STRAIGHT_FORWARD = Arm((0.0, 0.0, -1.0), (0.0, 0.0, -1.0))  # reto para a câmera
ARM_HANGING = Arm((0.0, 1.0, 0.0), (0.0, 1.0, 0.0))  # estendido para baixo
ARM_BENT_DOWN = Arm((0.0, 1.0, 0.0), (0.0, 0.2, -1.0))  # cotovelo para baixo, antebraço à frente
ARM_RAISED_SIDE = Arm((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))  # aberto para o lado (sem alcance z)


def _unit(v: Vec3) -> NDArray[np.float64]:
    a = np.asarray(v, dtype=np.float64)
    return a / float(np.linalg.norm(a))


def make_body(
    *,
    t: float = 0.0,
    scale: float = 1.0,
    hip_x_px: float = 320.0,
    hip_y_px: float = 300.0,
    lean: float = 0.0,
    lateral_lean: float = 0.0,
    left_arm: Arm = ARM_HANGING,
    right_arm: Arm = ARM_HANGING,
    size: tuple[int, int] = (640, 480),
    visibility: float = 1.0,
    with_world: bool = True,
    hips_visible: bool = True,
    shoulder_dy_px: float = 0.0,
    knee_lift_left: float = 0.0,
    knee_lift_right: float = 0.0,
    knees_visible: bool = True,
) -> PoseFrame:
    """Pose sintética de uma pessoa de frente para a câmera.

    ``scale`` multiplica só os pixels (aproximar da câmera); o world é métrico e não muda.
    ``lean`` (rad) inclina o tronco para frente: os ombros vão em direção à câmera (z < 0).
    ``lateral_lean`` (rad) inclina para os lados: positivo = esquerda da pessoa (+x na
    imagem não espelhada), negativo = direita.
    ``hips_visible=False`` tira os quadris do quadro (visibilidade 0); ``shoulder_dy_px``
    desce os ombros na imagem (pixels, y cresce para baixo), como no agachar (spec 017).
    Spec 018: ``knee_lift_left/right`` sobem o joelho do lado em larguras de ombro (``k`` da
    RF1: com a base em repouso, ``k`` vale exatamente esse valor); ``knees_visible=False``
    tira os joelhos do quadro (visibilidade 0).
    """
    world = np.zeros((N_LANDMARKS, 4), dtype=np.float64)
    world[:, 3] = visibility
    up = np.array(
        [
            math.cos(lean) * math.sin(lateral_lean),
            -math.cos(lean) * math.cos(lateral_lean),
            -math.sin(lean),
        ]
    )  # quadril -> ombros
    sh_center = TORSO_M * up
    half_sh = np.array([SHOULDER_WIDTH_M / 2.0, 0.0, 0.0])
    left_sh, right_sh = sh_center + half_sh, sh_center - half_sh
    world[PoseLandmark.LEFT_SHOULDER, :3] = left_sh
    world[PoseLandmark.RIGHT_SHOULDER, :3] = right_sh
    world[PoseLandmark.LEFT_HIP, :3] = (HIP_WIDTH_M / 2.0, 0.0, 0.0)
    world[PoseLandmark.RIGHT_HIP, :3] = (-HIP_WIDTH_M / 2.0, 0.0, 0.0)
    arms = (
        (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, left_sh, left_arm),
        (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, right_sh, right_arm),
    )
    for el, wr, shoulder, arm in arms:
        elbow = shoulder + UPPER_ARM_M * _unit(arm.upper)
        world[el, :3] = elbow
        world[wr, :3] = elbow + FOREARM_M * _unit(arm.fore)
    for knee, hip_x, lift in (
        (PoseLandmark.LEFT_KNEE, HIP_WIDTH_M / 2.0, knee_lift_left),
        (PoseLandmark.RIGHT_KNEE, -HIP_WIDTH_M / 2.0, knee_lift_right),
    ):
        world[knee, :3] = (hip_x, THIGH_M - lift * SHOULDER_WIDTH_M, 0.0)
    # Pontos não usados ficam no quadril (já em 0); só importa o que os gestos leem.
    k = SHOULDER_PX_AT_SCALE_1 * scale / SHOULDER_WIDTH_M
    w_img, h_img = size
    lm = np.zeros((N_LANDMARKS, 5), dtype=np.float64)
    lm[:, 0] = (hip_x_px + k * world[:, 0]) / w_img
    lm[:, 1] = (hip_y_px + k * world[:, 1]) / h_img
    lm[:, 2] = world[:, 2]
    lm[:, 3] = visibility
    lm[:, 4] = 1.0
    for shoulder in (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER):
        lm[shoulder, 1] += shoulder_dy_px / h_img
    if not knees_visible:
        for knee in (PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE):
            lm[knee, 3] = 0.0
            if with_world:
                world[knee, 3] = 0.0
    if not hips_visible:
        for hip in (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP):
            lm[hip, 3] = 0.0
            if with_world:
                world[hip, 3] = 0.0
    return PoseFrame(t=t, image_size=size, landmarks=lm, world=world if with_world else None)


def with_face(
    pose: PoseFrame,
    *,
    yaw: float = 0.0,
    pitch: float = 0.0,
    center_px: tuple[float, float] | None = None,
    ear_px: float | None = None,
    visibility: float = 1.0,
) -> PoseFrame:
    """Põe nariz, olhos e orelhas (0, 2, 5, 7, 8) na imagem com yaw/pitch exatos (spec 014).

    As orelhas ficam em ``center ± ear_px/2`` na horizontal; o nariz vai a ``−yaw·ear_px``
    do meio delas (``yaw > 0`` = direita da pessoa, que na imagem não espelhada fica à
    esquerda) e ``pitch·s`` acima do meio dos ombros, com ``s`` a largura dos ombros em
    pixels (pitch da revisão de 12/11/2026). Assim ``head_pose`` devolve exatamente
    ``(yaw, pitch)``. Sem ``center_px``/``ear_px``, a cabeça acompanha o corpo: centro
    ``0,6·s`` acima dos ombros e ``ear_px = 0,4·s``.
    """
    w_img, h_img = pose.image_size
    lm = np.array(pose.landmarks, copy=True)
    shoulders = lm[[PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER], 0:2] * (w_img, h_img)
    sh_x, sh_y = shoulders.mean(axis=0)
    s = float(np.linalg.norm(shoulders[0] - shoulders[1]))
    cx, cy = center_px if center_px is not None else (sh_x, sh_y - 0.6 * s)
    ear = ear_px if ear_px is not None else 0.4 * s
    eye_y = cy - ear / 4.0
    points = {
        PoseLandmark.LEFT_EAR: (cx + ear / 2.0, cy),  # orelha esquerda da pessoa: x maior
        PoseLandmark.RIGHT_EAR: (cx - ear / 2.0, cy),
        PoseLandmark.LEFT_EYE: (cx + ear / 4.0, eye_y),
        PoseLandmark.RIGHT_EYE: (cx - ear / 4.0, eye_y),
        PoseLandmark.NOSE: (cx - yaw * ear, sh_y - pitch * s),
    }
    for idx, (x, y) in points.items():
        lm[idx, 0] = x / w_img
        lm[idx, 1] = y / h_img
        lm[idx, 3] = visibility
    return PoseFrame(t=pose.t, image_size=pose.image_size, landmarks=lm, world=pose.world)
