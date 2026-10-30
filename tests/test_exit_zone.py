"""Zona de saída temporária: mão parada no canto por ``hold_s`` encerra o app."""

from __future__ import annotations

import pytest

from inmotion.exit_zone import ExitZone, ExitZoneConfig
from inmotion.vision.landmarks import PoseLandmark
from tests.fakes import make_pose

SIZE = (640, 480)
CFG = ExitZoneConfig(x0=0.0, y0=0.1, x1=0.2, y1=0.4, hold_s=4.0, min_visibility=0.5)
INSIDE_PX = (64.0, 120.0)  # (0,1; 0,25) normalizado
OUTSIDE_PX = (320.0, 240.0)


def _pose(wrist_px: tuple[float, float], t: float, visibility: float = 1.0, right: bool = True):
    wrist = PoseLandmark.RIGHT_WRIST if right else PoseLandmark.LEFT_WRIST
    other = PoseLandmark.LEFT_WRIST if right else PoseLandmark.RIGHT_WRIST
    return make_pose(
        {wrist: wrist_px, other: OUTSIDE_PX}, size=SIZE, t=t, visibility={wrist: visibility}
    )


def test_hold_for_hold_s_triggers() -> None:
    zone = ExitZone(CFG)
    states = [zone.update(_pose(INSIDE_PX, t), t) for t in (10.0, 12.0, 13.99)]
    assert all(s.inside for s in states)
    assert not any(s.triggered for s in states)
    assert states[1].progress == pytest.approx(0.5)
    final = zone.update(_pose(INSIDE_PX, 14.0), 14.0)
    assert final.triggered
    assert final.progress == pytest.approx(1.0)


def test_either_wrist_counts() -> None:
    zone = ExitZone(CFG)
    zone.update(_pose(INSIDE_PX, 0.0, right=False), 0.0)
    assert zone.update(_pose(INSIDE_PX, 4.0, right=False), 4.0).triggered


def test_leaving_the_zone_resets_the_timer() -> None:
    zone = ExitZone(CFG)
    zone.update(_pose(INSIDE_PX, 0.0), 0.0)
    zone.update(_pose(INSIDE_PX, 3.0), 3.0)
    out = zone.update(_pose(OUTSIDE_PX, 3.5), 3.5)
    assert not out.inside
    assert out.progress == 0.0
    zone.update(_pose(INSIDE_PX, 4.0), 4.0)
    assert not zone.update(_pose(INSIDE_PX, 7.9), 7.9).triggered
    assert zone.update(_pose(INSIDE_PX, 8.0), 8.0).triggered


def test_lost_pose_and_low_visibility_count_as_outside() -> None:
    zone = ExitZone(CFG)
    zone.update(_pose(INSIDE_PX, 0.0), 0.0)
    assert not zone.update(None, 2.0).inside
    assert not zone.update(_pose(INSIDE_PX, 4.0, visibility=0.2), 4.0).inside
    assert not zone.update(_pose(INSIDE_PX, 6.0), 6.0).triggered  # recomeçou em 6,0


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="retângulo"):
        ExitZoneConfig(x0=0.3, x1=0.2)
    with pytest.raises(ValueError, match="retângulo"):
        ExitZoneConfig(y0=-0.1)
    with pytest.raises(ValueError, match="hold_s"):
        ExitZoneConfig(hold_s=0.0)
    with pytest.raises(ValueError, match="min_visibility"):
        ExitZoneConfig(min_visibility=float("nan"))


def test_default_zone_is_top_left_below_text() -> None:
    cfg = ExitZoneConfig()
    assert cfg.x0 < cfg.x1 <= 0.25
    assert 0.1 <= cfg.y0 < cfg.y1 <= 0.5
    assert cfg.hold_s == 4.0
