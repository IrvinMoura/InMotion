"""Gesto de inclinar o tronco para frente (RF5, A9): hold para andar."""

from __future__ import annotations

from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.vision.geometry import GeometryConfig


def lean_gesture(params: GestureParams, geometry: GeometryConfig) -> FeatureGesture:
    raise NotImplementedError
