"""Gesto de pulo (RF2): trigger sobre ``j = (b − h)/s``."""

from __future__ import annotations

from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture


def jump_gesture(params: GestureParams) -> FeatureGesture:
    raise NotImplementedError
