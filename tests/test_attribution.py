"""
tests/test_attribution.py - Phase 12: Attribution Engine Tests

Tests:
  1. test_case_1_visible_robot_at_goal
  2. test_case_2_robot_moved_before_score
  3. test_case_3_robot_never_visible
  4. test_case_4_scoreboard_only_change
  5. test_case_5_ambiguous_dual_robot
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from inference_engine import attribute_score, build_score_timeline, compute_final_scores


ZONES = {"red_goal": [600, 200, 700, 400]}

def _robot_tracks_at(track_id, frame_id, cx, cy, team=None):
    entry = {"frame_id": frame_id, "track_id": track_id,
             "bbox": [cx-25, cy-25, cx+25, cy+25]}
    if team:
        entry["team_number"] = team
    return {track_id: [entry]}

def _identity(track_id, team):
    return {track_id: {"team_number": team, "confidence": 0.9}}


# ---- Test 1 -----------------------------------------------------------------

def test_case_1_visible_robot_at_goal():
    """
    Robot visible at zone, trajectory origin and last possessor agree.
    Must return confidence >= 0.90 with method=trajectory_origin.
    """
    event = {
        "event_frame": 100,
        "zone": "red_goal",
        "ball_track_id": 1,
        "last_possessor": "1234",
        "last_possessor_frame": 95,
        "trajectory_origin_robot": "1234",
        "confidence": 0.7,
    }
    # Robot 10 = team 1234, positioned at zone centre (650, 300)
    robot_tracks = _robot_tracks_at(10, 100, 650, 300, "1234")
    identity     = _identity(10, "1234")

    result = attribute_score(event, {1: []}, robot_tracks, {}, ZONES, identity)

    assert result["case"] == 1,                  f"Expected case 1, got {result['case']}"
    assert result["confidence"] >= 0.90,         f"Expected conf>=0.90, got {result['confidence']}"
    assert result["method"] == "trajectory_origin"
    print("PASS  test_case_1_visible_robot_at_goal")


# ---- Test 2 -----------------------------------------------------------------

def test_case_2_robot_moved_before_score():
    """
    Trajectory + last_possessor agree but robot is NOT near zone at event frame.
    Must return confidence >= 0.80, case 2.
    """
    event = {
        "event_frame": 100,
        "zone": "red_goal",
        "ball_track_id": 1,
        "last_possessor": "5678",
        "last_possessor_frame": 85,
        "trajectory_origin_robot": "5678",
        "confidence": 0.65,
    }
    # Robot far from zone (opposite side of field)
    robot_tracks = _robot_tracks_at(20, 100, 100, 100, "5678")
    identity     = _identity(20, "5678")

    result = attribute_score(event, {1: []}, robot_tracks, {}, ZONES, identity)

    assert result["case"] == 2,           f"Expected case 2, got {result['case']}"
    assert result["confidence"] >= 0.80,  f"Expected conf>=0.80, got {result['confidence']}"
    print("PASS  test_case_2_robot_moved_before_score")


# ---- Test 3 -----------------------------------------------------------------

def test_case_3_robot_never_visible():
    """
    No trajectory origin, no robot near zone.
    Must attribute to last_possessor with moderate confidence (case 3).
    """
    event = {
        "event_frame": 100,
        "zone": "red_goal",
        "ball_track_id": 1,
        "last_possessor": "9999",
        "last_possessor_frame": 80,
        "trajectory_origin_robot": None,
        "confidence": 0.55,
    }

    result = attribute_score(event, {1: []}, {}, {}, ZONES, None)

    assert result["case"] == 3,                   f"Expected case 3, got {result}"
    assert result["team_number"] == "9999"
    assert 0.50 <= result["confidence"] <= 0.80
    print("PASS  test_case_3_robot_never_visible")


# ---- Test 4 -----------------------------------------------------------------

def test_case_4_scoreboard_only_change():
    """
    No last_possessor, no trajectory; robot near zone is the only signal.
    Must use proximity_to_zone with low confidence, flag INFERRED-LOW-CONF.
    """
    event = {
        "event_frame": 150,
        "zone": "red_goal",
        "ball_track_id": 2,
        "last_possessor": None,
        "last_possessor_frame": None,
        "trajectory_origin_robot": None,
        "confidence": 0.30,
    }
    robot_tracks = _robot_tracks_at(30, 150, 640, 300, "2468")
    identity     = _identity(30, "2468")

    result = attribute_score(event, {2: []}, robot_tracks, {}, ZONES, identity)

    assert result["case"] == 4,                        f"Expected case 4, got {result}"
    assert result["method"] == "proximity_to_zone"
    assert result["flag"] == "INFERRED-LOW-CONF"
    assert result["confidence"] < 0.65
    print("PASS  test_case_4_scoreboard_only_change")


# ---- Test 5 -----------------------------------------------------------------

def test_case_5_ambiguous_dual_robot():
    """
    Two robots equally close to zone simultaneously -> AMBIGUOUS flag.
    """
    event = {
        "event_frame": 200,
        "zone": "red_goal",
        "ball_track_id": 3,
        "last_possessor": None,
        "last_possessor_frame": None,
        "trajectory_origin_robot": None,
        "confidence": 0.25,
    }
    # Both robots equidistant from zone centre (650, 300)
    from inference_engine import attribute_score as _attr
    robot_tracks = {}
    robot_tracks.update(_robot_tracks_at(40, 200, 650, 260, "1111"))
    robot_tracks.update(_robot_tracks_at(50, 200, 650, 340, "2222"))
    identity = {40: {"team_number": "1111"}, 50: {"team_number": "2222"}}

    result = _attr(event, {3: []}, robot_tracks, {}, ZONES, identity)

    assert result["flag"] == "AMBIGUOUS-MANUAL-REVIEW", (
        f"Expected AMBIGUOUS flag, got {result}"
    )
    print("PASS  test_case_5_ambiguous_dual_robot")


# ---- Runner -----------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_case_1_visible_robot_at_goal,
        test_case_2_robot_moved_before_score,
        test_case_3_robot_never_visible,
        test_case_4_scoreboard_only_change,
        test_case_5_ambiguous_dual_robot,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"PHASE 12 attribution -- {passed}/{len(tests)} tests passed")
    print("=" * 50)
    import sys; sys.exit(0 if failed == 0 else 1)
