"""Gesto de agachar (RF4): hold sobre ``(h − b)/s``."""

from __future__ import annotations

from inmotion.gestures.base import GestureParams
from inmotion.gestures.body import FeatureGesture


def crouch_gesture(params: GestureParams) -> FeatureGesture:
    raise NotImplementedError
