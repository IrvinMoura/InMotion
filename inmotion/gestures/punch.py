"""Gesto de soco (RF1): histerese sobre o alcance frontal ``r``."""

from __future__ import annotations

from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture
from inmotion.vision.geometry import GeometryConfig
from inmotion.vision.landmarks import Side


def punch_gesture(
    name: str, side: Side, params: GestureParams, geometry: GeometryConfig
) -> FeatureGesture:
    raise NotImplementedError
