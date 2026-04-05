"""
tests/test_trajectory.py - Phase 12: Trajectory Engine Tests

Tests:
  1. test_velocity_computation
  2. test_zone_entry_prediction
  3. test_offscreen_trajectory_case
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from trajectory import (
    compute_ball_velocity,
    predict_trajectory,
    will_enter_zone,
)


def _make_ball_track(positions, start=0):
    """Build ball_tracks from list of (cx, cy)."""
    entries = [
        {"frame_id": start + i, "track_id": 1,
         "bbox": [cx-5, cy-5, cx+5, cy+5]}
        for i, (cx, cy) in enumerate(positions)
    ]
    return {1: entries}


# ---- Test 1 -----------------------------------------------------------------

def test_velocity_computation():
    """Velocity from known constant-motion positions must match dx/dy."""
    positions = [(float(i * 10), 200.0) for i in range(10)]
    tracks = _make_ball_track(positions)

    vel = compute_ball_velocity(1, tracks, frame_window=5)

    assert abs(vel["vx"] - 10.0) < 1.0, f"Expected vx~10, got {vel['vx']}"
    assert abs(vel["vy"])        < 1.0, f"Expected vy~0, got {vel['vy']}"
    assert abs(vel["speed"] - 10.0) < 1.0
    print("PASS  test_velocity_computation")


# ---- Test 2 -----------------------------------------------------------------

def test_zone_entry_prediction():
    """Ball moving toward a zone should be predicted to enter it."""
    # Ball at (0, 300) moving right at 20px/frame; zone at x=400-500, y=250-350
    vel       = {"vx": 20.0, "vy": 0.0, "speed": 20.0, "heading_deg": 0.0}
    predicted = predict_trajectory((0.0, 300.0), vel, frames_ahead=30)
    zone_bbox = [400, 250, 500, 350]

    result = will_enter_zone(predicted, zone_bbox)

    assert result["will_score"],       "Expected ball to enter zone"
    assert result["predicted_frame"] > 0
    assert result["confidence"]       > 0.0
    print("PASS  test_zone_entry_prediction")


# ---- Test 3 -----------------------------------------------------------------

def test_offscreen_trajectory_case():
    """
    Ball moving AWAY from zone should NOT predict a score.
    Simulates ball flying off screen / wrong direction.
    """
    # Ball moving left (negative vx), zone is to the right
    vel       = {"vx": -15.0, "vy": 0.0, "speed": 15.0, "heading_deg": 180.0}
    predicted = predict_trajectory((640.0, 360.0), vel, frames_ahead=30)
    zone_bbox = [900, 300, 1000, 420]   # far right

    result = will_enter_zone(predicted, zone_bbox)

    assert not result["will_score"], (
        f"Expected NO zone entry for ball moving away; got {result}"
    )
    print("PASS  test_offscreen_trajectory_case")


# ---- Runner -----------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_velocity_computation,
        test_zone_entry_prediction,
        test_offscreen_trajectory_case,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"PHASE 12 trajectory -- {passed}/{len(tests)} tests passed")
    print("=" * 50)
    import sys; sys.exit(0 if failed == 0 else 1)
