"""Montagem dos gestos de corpo e do árbitro (spec 008). Núcleo puro.

Os nomes são os do perfil de jogo: ``attack`` e ``use`` (socos), ``jump``, ``crouch``,
``walk`` e ``sprint`` (marcha no lugar, spec 018) e ``lean`` (inclinar o tronco). Só existem
os gestos que o perfil mapeia: um gesto sem
binding não pode influenciar o repouso (R7) nem os grupos exclusivos.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from inmotion.gestures.arbiter import DEFAULT_ARBITER, ArbiterConfig, GestureArbiter
from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.combo import ComboGesture
from inmotion.gestures.crouch import DEFAULT_CROUCH_PARAMS, CrouchFeature, crouch_gesture
from inmotion.gestures.jump import DEFAULT_JUMP_PARAMS, jump_gesture
from inmotion.gestures.lean import DEFAULT_LEAN_PARAMS, lean_gesture
from inmotion.gestures.march import (
    DEFAULT_SPRINT_PARAMS,
    DEFAULT_WALK_PARAMS,
    MarchParams,
    sprint_gesture,
    walk_gesture,
)
from inmotion.gestures.punch import DEFAULT_PUNCH_PARAMS, opposite, punch_gesture
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side

BODY_GESTURE_NAMES: frozenset[str] = frozenset(
    {"attack", "use", "jump", "crouch", "walk", "sprint", "lean"}
)

DEFAULT_GESTURE_PARAMS: Mapping[str, GestureParams] = {
    "attack": DEFAULT_PUNCH_PARAMS,
    "use": DEFAULT_PUNCH_PARAMS,
    "jump": DEFAULT_JUMP_PARAMS,
    "crouch": DEFAULT_CROUCH_PARAMS,
    "walk": DEFAULT_WALK_PARAMS,
    "sprint": DEFAULT_SPRINT_PARAMS,
    "lean": DEFAULT_LEAN_PARAMS,
}


def build_body_gestures(
    names: Collection[str],
    handedness: Side = Side.RIGHT,
    params: Mapping[str, GestureParams] | None = None,
    geometry: GeometryConfig | None = None,
    crouch_feature: CrouchFeature = "shoulders",
    march: MarchParams | None = None,
) -> list[FeatureGesture]:
    """Gestos de corpo para os ``names`` do perfil, na ordem de prioridade.

    O braço dominante (``handedness``) ataca e o outro usa item (CA2). ``params`` sobrescreve
    os padrões por nome; ``crouch_feature`` escolhe o sinal do agachar (ombros ou quadril).
    ``march`` traz o
    ``θ_on`` do passo, que o pulo usa para não disparar com o joelho erguido (RF6b da 018).
    """
    geometry = geometry if geometry is not None else GeometryConfig()
    merged = {**DEFAULT_GESTURE_PARAMS, **(params or {})}
    wanted = set(names)
    gestures: list[FeatureGesture] = []
    if "attack" in wanted:
        gestures.append(punch_gesture("attack", handedness, merged["attack"], geometry))
    if "use" in wanted:
        gestures.append(punch_gesture("use", opposite(handedness), merged["use"], geometry))
    if "jump" in wanted:
        knee_gate = (march or MarchParams()).step.theta_on
        gestures.append(jump_gesture(merged["jump"], knee_gate))
    if "crouch" in wanted:
        gestures.append(crouch_gesture(merged["crouch"], crouch_feature))
    if "walk" in wanted:
        gestures.append(walk_gesture(merged["walk"]))
    if "sprint" in wanted:
        gestures.append(sprint_gesture(merged["sprint"]))
    if "lean" in wanted:
        gestures.append(lean_gesture(merged["lean"], geometry))
    return gestures


def build_arbiter(
    names: Collection[str], config: ArbiterConfig = DEFAULT_ARBITER
) -> tuple[GestureArbiter, ComboGesture | None]:
    """Árbitro dos gestos presentes, lido da ``config`` (seção ``arbiter`` do perfil, RF2).

    Nomes da config sem gesto montado são ignorados:

    - a prioridade mantém só os gestos presentes (e os combos que se formam);
    - cada grupo exclusivo vale pelos membros presentes, e só se restarem dois ou mais;
    - um combo só existe se todos os seus membros estão presentes (``both_arms`` exige
      ``attack`` e ``use``). O pipeline trata um combo por vez: mais de um é ``ValueError``.
    """
    ranked = set(config.priority)
    present = set(names) & ranked
    formed = {n: m for n, m in config.combos.items() if n in ranked and m <= present}
    if len(formed) > 1:
        raise ValueError(f"arbiter: só um combo por perfil é suportado, recebido {sorted(formed)}")
    combo = ComboGesture(*next(iter(formed.items()))) if formed else None
    groups = [g & present for g in config.exclusive]
    priority = [n for n in config.priority if n in present or n in formed]
    return GestureArbiter(priority, [g for g in groups if len(g) >= 2], formed or None), combo
