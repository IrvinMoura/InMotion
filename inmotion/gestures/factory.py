"""Montagem dos gestos de corpo e do árbitro (spec 008). Núcleo puro.

Os nomes são os do perfil de jogo: ``attack`` e ``use`` (socos), ``jump``, ``crouch`` e
``walk`` (inclinar o tronco). Só existem os gestos que o perfil mapeia: um gesto sem
binding não pode influenciar o repouso (R7) nem os grupos exclusivos.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from inmotion.gestures.arbiter import ArbiterConfig, GestureArbiter
from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.gestures.combo import ComboGesture
from inmotion.gestures.crouch import DEFAULT_CROUCH_PARAMS, crouch_gesture
from inmotion.gestures.jump import DEFAULT_JUMP_PARAMS, jump_gesture
from inmotion.gestures.lean import DEFAULT_LEAN_PARAMS, lean_gesture
from inmotion.gestures.punch import DEFAULT_PUNCH_PARAMS, opposite, punch_gesture
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side

BODY_GESTURE_NAMES: frozenset[str] = frozenset({"attack", "use", "jump", "crouch", "walk"})
BOTH_ARMS = "both_arms"
# Prioridade do árbitro (antes = mais prioritário); o combo suprime os membros.
_PRIORITY: tuple[str, ...] = (BOTH_ARMS, "attack", "use", "jump", "crouch", "walk")

DEFAULT_GESTURE_PARAMS: Mapping[str, GestureParams] = {
    "attack": DEFAULT_PUNCH_PARAMS,
    "use": DEFAULT_PUNCH_PARAMS,
    "jump": DEFAULT_JUMP_PARAMS,
    "crouch": DEFAULT_CROUCH_PARAMS,
    "walk": DEFAULT_LEAN_PARAMS,
}


def build_body_gestures(
    names: Collection[str],
    handedness: Side = Side.RIGHT,
    params: Mapping[str, GestureParams] | None = None,
    geometry: GeometryConfig | None = None,
) -> list[FeatureGesture]:
    """Gestos de corpo para os ``names`` do perfil, na ordem de prioridade.

    O braço dominante (``handedness``) ataca e o outro usa item (CA2). ``params`` sobrescreve
    os padrões por nome.
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
        gestures.append(jump_gesture(merged["jump"]))
    if "crouch" in wanted:
        gestures.append(crouch_gesture(merged["crouch"]))
    if "walk" in wanted:
        gestures.append(lean_gesture(merged["walk"], geometry))
    return gestures


def build_arbiter(
    names: Collection[str], config: ArbiterConfig | None = None
) -> tuple[GestureArbiter, ComboGesture | None]:
    """Árbitro do Minecraft para os gestos presentes e o combo ``both_arms``, se couber.

    - exclusivo ``{jump, crouch}`` (só vale se os dois existem);
    - combo ``both_arms = {attack, use}``: só suprime os dois braços (nenhuma ação).
    """
    if config is not None:
        raise NotImplementedError("spec 017 T2: árbitro a partir do ArbiterConfig")
    present = set(names)
    groups = [frozenset({"jump", "crouch"})] if {"jump", "crouch"} <= present else []
    combo = (
        ComboGesture(BOTH_ARMS, frozenset({"attack", "use"}))
        if {"attack", "use"} <= present
        else None
    )
    priority = [n for n in _PRIORITY if n in present or (n == BOTH_ARMS and combo is not None)]
    combos = {BOTH_ARMS: combo.members} if combo is not None else None
    return GestureArbiter(priority, groups, combos), combo
