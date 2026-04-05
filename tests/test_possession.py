"""
tests/test_possession.py - Phase 12: Possession Engine Tests

Tests:
  1. test_proximity_assignment
  2. test_velocity_filter_prevents_false_positive
  3. test_handoff_detection
  4. test_reacquisition_after_occlusion
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from possession import (
    assign_possession,
    build_possession_log,
    get_last_possessor,
    detect_handoff,
)

RULES = {
    "proximity_threshold_px":     80,
    "possession_min_frames":       3,
    "release_threshold_px":       150,
    "velocity_handoff_threshold":   5,
}


def _make_track(track_id, positions, team="TEST", start=0):
    """Build a track dict from list of (cx, cy) positions."""
    entries = []
    for i, (cx, cy) in enumerate(positions):
        entries.append({
            "frame_id":   start + i,
            "track_id":   track_id,
            "bbox":       [cx-20, cy-20, cx+20, cy+20],
            "team_number": team,
        })
    return {track_id: entries}


def _merge(*track_dicts):
    merged = {}
    for d in track_dicts:
        merged.update(d)
    return merged


# ---- Test 1 -----------------------------------------------------------------

def test_proximity_assignment():
    """Ball within proximity threshold of a slow robot -> possessed."""
    ball_tracks  = _make_track(1, [(100.0, 100.0)] * 5)
    robot_tracks = _make_track(10, [(115.0, 100.0)] * 5, team="1234")

    result = assign_possession(ball_tracks, robot_tracks, frame_id=2, rules=RULES)

    assert 1 in result, "Expected ball track 1 to be possessed"
    assert result[1]["team_number"] == "1234"
    assert result[1]["confidence"] > 0
    print("PASS  test_proximity_assignment")


# ---- Test 2 -----------------------------------------------------------------

def test_velocity_filter_prevents_false_positive():
    """
    Ball moving fast near a robot should NOT be assigned possession.
    (ball in flight, not held)
    """
    # Ball moving 20px/frame (well above vel_threshold=5)
    positions = [(float(i * 20), 100.0) for i in range(5)]
    ball_tracks  = _make_track(1, positions)
    # Robot stationary right next to ball at frame 4
    robot_tracks = _make_track(10, [(70.0, 100.0)] * 5, team="5678")

    result = assign_possession(ball_tracks, robot_tracks, frame_id=4, rules=RULES)

    assert 1 not in result, (
        f"Fast-moving ball should NOT be possessed; got {result}"
    )
    print("PASS  test_velocity_filter_prevents_false_positive")


# ---- Test 3 -----------------------------------------------------------------

def test_handoff_detection():
    """Ball held by Robot A then Robot B within frame_window -> handoff detected."""
    # Robot A possesses frames 0-4, Robot B possesses frames 6-10
    log = {
        1: (
            [{"frame_id": f, "robot_track_id": 10, "team_number": "1234",
              "confidence": 0.9} for f in range(5)]
            +
            [{"frame_id": f, "robot_track_id": 20, "team_number": "5678",
              "confidence": 0.9} for f in range(6, 11)]
        )
    }

    handoff = detect_handoff(1, log, frame_window=15)

    assert handoff is not None, "Expected a handoff to be detected"
    assert handoff["from_robot"] == 10
    assert handoff["to_robot"]   == 20
    assert handoff["from_team"]  == "1234"
    assert handoff["to_team"]    == "5678"
    print("PASS  test_handoff_detection")


# ---- Test 4 -----------------------------------------------------------------

def test_reacquisition_after_occlusion():
    """
    Ball disappears (no entries) for several frames then reappears
    near same robot. get_last_possessor must still return correct robot.
    """
    log = {
        1: [
            {"frame_id": 10, "robot_track_id": 10,
             "team_number": "9999", "confidence": 0.85},
            {"frame_id": 11, "robot_track_id": 10,
             "team_number": "9999", "confidence": 0.85},
            # frames 12-19: occlusion (no entries)
            {"frame_id": 20, "robot_track_id": 10,
             "team_number": "9999", "confidence": 0.80},
        ]
    }

    last = get_last_possessor(1, before_frame=25, possession_log=log)
    assert last is not None,              "Expected a last possessor"
    assert last["team_number"] == "9999", f"Expected 9999, got {last['team_number']}"
    assert last["frame_id"] == 20,        f"Expected frame 20, got {last['frame_id']}"
    print("PASS  test_reacquisition_after_occlusion")


# ---- Runner -----------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_proximity_assignment,
        test_velocity_filter_prevents_false_positive,
        test_handoff_detection,
        test_reacquisition_after_occlusion,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"PHASE 12 possession -- {passed}/{len(tests)} tests passed")
    print("=" * 50)
    import sys; sys.exit(0 if failed == 0 else 1)
