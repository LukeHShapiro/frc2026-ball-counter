"""
tests/test_driving.py — Phase 13: Driving Analysis Tests

Tests:
  1.  test_velocity_computation_accuracy
  2.  test_collision_detection_no_false_positives
  3.  test_shadowing_detection_min_duration
  4.  test_path_repetition_identical_paths
  5.  test_path_repetition_random_paths
  6.  test_classification_defensive_profile
  7.  test_classification_reckless_profile
  8.  test_classification_smooth_profile
  9.  test_classification_defence_proof_profile
  10. test_dual_style_assignment
  11. test_thresholds_load_from_config
"""

import sys
import os
import math

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from driving_analysis import (
    compute_robot_velocity,
    detect_collisions,
    compute_path_repetition,
    detect_shadowing_events,
    classify_driving_style,
    _load_config,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def _make_track(positions: list[tuple[float, float]], start_frame: int = 0) -> list[dict]:
    """Build a robot track from a list of (cx, cy) centre positions."""
    return [
        {
            "frame_id": start_frame + i,
            "bbox": [cx - 20, cy - 20, cx + 20, cy + 20],
            "team_number": "TEST",
        }
        for i, (cx, cy) in enumerate(positions)
    ]


def _make_robot_tracks(positions_by_id: dict[int, list[tuple]]) -> dict:
    return {tid: _make_track(pos) for tid, pos in positions_by_id.items()}


# ── Test 1 — velocity computation accuracy ────────────────────────────────────

def test_velocity_computation_accuracy():
    """Velocity computed from known positions must match expected dx/dy."""
    # Robot moves 10px right per frame
    positions = [(float(i * 10), 100.0) for i in range(6)]
    robot_tracks = _make_robot_tracks({1: positions})

    vels = compute_robot_velocity(1, robot_tracks, frame_window=1)

    # All vx after first frame should be ≈ 10
    for v in vels[1:]:
        assert abs(v["vx"] - 10.0) < 1e-6, f"Expected vx≈10, got {v['vx']}"
        assert abs(v["vy"]) < 1e-6, f"Expected vy≈0, got {v['vy']}"
        assert abs(v["speed"] - 10.0) < 1e-6, f"Expected speed≈10, got {v['speed']}"

    print("PASS  test_velocity_computation_accuracy")


# ── Test 2 — no false-positive collisions when robots are far apart ───────────

def test_collision_detection_no_false_positives():
    """
    Robots that slow down with NO nearby opponent must not register collisions.
    """
    # Robot 1: decelerates sharply
    pos1 = [(float(i * 20), 100.0) for i in range(5)] + \
           [(float(5 * 20 + i), 100.0) for i in range(5)]  # slows to 1px/frame

    # Robot 2: very far away (x=900+)
    pos2 = [(900.0 + float(i), 100.0) for i in range(10)]

    robot_tracks = _make_robot_tracks({1: pos1, 2: pos2})

    collisions = detect_collisions(
        1, robot_tracks, robot_tracks,
        velocity_drop_threshold=0.4,
        proximity_threshold_px=100.0,
    )

    assert len(collisions) == 0, (
        f"Expected 0 collisions (opponent too far), got {len(collisions)}"
    )
    print("PASS  test_collision_detection_no_false_positives")


# ── Test 3 — shadowing detection enforces minimum duration ───────────────────

def test_shadowing_detection_min_duration():
    """
    A robot that follows an opponent for fewer frames than follow_duration_frames
    must NOT be flagged as shadowing.
    """
    # Robot 1 and Robot 2 move together for only 5 frames (< 20 required)
    base = [(float(i * 5), 200.0) for i in range(5)]
    pos1 = base
    pos2 = [(x + 30, y) for x, y in base]  # 30px apart — within threshold

    robot_tracks = _make_robot_tracks({1: pos1, 2: pos2})

    events = detect_shadowing_events(
        1, robot_tracks, robot_tracks,
        follow_duration_frames=20,
        proximity_threshold_px=120.0,
    )

    assert len(events) == 0, (
        f"Expected 0 shadowing events (too short), got {len(events)}"
    )
    print("PASS  test_shadowing_detection_min_duration")


# ── Test 4 — path repetition: identical paths score 1.0 ──────────────────────

def test_path_repetition_identical_paths():
    """A robot that always takes the exact same route must score ≈1.0."""
    loop = [(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)] * 5
    robot_tracks = _make_robot_tracks({1: loop})

    score = compute_path_repetition(1, robot_tracks, grid_resolution=50)

    assert score > 0.70, f"Expected high repetition score, got {score:.3f}"
    print("PASS  test_path_repetition_identical_paths")


# ── Test 5 — path repetition: random paths score low ─────────────────────────

def test_path_repetition_random_paths():
    """A robot that always moves to new positions scores close to 0.0."""
    import random
    rng = random.Random(42)
    positions = [(rng.uniform(0, 1280), rng.uniform(0, 720)) for _ in range(40)]
    robot_tracks = _make_robot_tracks({1: positions})

    score = compute_path_repetition(1, robot_tracks, grid_resolution=50)

    assert score < 0.60, f"Expected low repetition score for random path, got {score:.3f}"
    print("PASS  test_path_repetition_random_paths")


# ── Test 6 — defensive profile classification ────────────────────────────────

def test_classification_defensive_profile():
    """
    A robot with many shadowing events, time in opponent half, and low velocity
    must classify as DEFENSIVE.
    """
    metrics = {
        "avg_velocity_px_per_frame":        1.5,
        "velocity_variance":                0.5,
        "max_velocity":                     4.0,
        "velocity_recovery_rate":           0.1,
        "collision_count":                  1,
        "collision_rate_per_minute":        0.3,
        "angular_velocity_variance":        20.0,
        "path_repetition_score":            0.55,
        "arc_smoothness":                   0.80,
        "time_in_opponent_half_pct":        0.55,   # > 0.30 threshold
        "shadowing_events":                 6,      # > 3 threshold
        "avg_distance_to_nearest_opponent": 80.0,
        "scoring_under_pressure_rate":      0.10,
        "escape_success_rate":              0.20,
        "path_variety_score":               0.45,
    }
    result = classify_driving_style(metrics)
    assert result["primary_style"] == "DEFENSIVE", (
        f"Expected DEFENSIVE, got {result['primary_style']} "
        f"(scores: {result['style_scores']})"
    )
    print("PASS  test_classification_defensive_profile")


# ── Test 7 — reckless profile classification ─────────────────────────────────

def test_classification_reckless_profile():
    """
    A robot with high collision rate, high max velocity, and erratic heading
    must classify as RECKLESS.
    """
    metrics = {
        "avg_velocity_px_per_frame":        14.0,
        "velocity_variance":                60.0,
        "max_velocity":                     25.0,
        "velocity_recovery_rate":           0.5,
        "collision_count":                  8,
        "collision_rate_per_minute":        3.5,    # > 1.5 threshold
        "angular_velocity_variance":        350.0,  # > 200 (high)
        "path_repetition_score":            0.30,
        "arc_smoothness":                   0.30,
        "time_in_opponent_half_pct":        0.10,
        "shadowing_events":                 0,
        "avg_distance_to_nearest_opponent": 200.0,
        "scoring_under_pressure_rate":      0.20,
        "escape_success_rate":              0.30,
        "path_variety_score":               0.70,
    }
    result = classify_driving_style(metrics)
    assert result["primary_style"] == "RECKLESS", (
        f"Expected RECKLESS, got {result['primary_style']} "
        f"(scores: {result['style_scores']})"
    )
    print("PASS  test_classification_reckless_profile")


# ── Test 8 — smooth profile classification ───────────────────────────────────

def test_classification_smooth_profile():
    """
    A robot with low velocity variance, low collision rate, and high arc smoothness
    must classify as SMOOTH.
    """
    metrics = {
        "avg_velocity_px_per_frame":        5.0,
        "velocity_variance":                1.2,   # low
        "max_velocity":                     8.0,
        "velocity_recovery_rate":           0.2,
        "collision_count":                  0,
        "collision_rate_per_minute":        0.1,   # < 0.5 threshold
        "angular_velocity_variance":        15.0,  # low
        "path_repetition_score":            0.35,
        "arc_smoothness":                   0.90,  # high
        "time_in_opponent_half_pct":        0.05,
        "shadowing_events":                 0,
        "avg_distance_to_nearest_opponent": 300.0,
        "scoring_under_pressure_rate":      0.30,
        "escape_success_rate":              0.40,
        "path_variety_score":               0.65,
    }
    result = classify_driving_style(metrics)
    assert result["primary_style"] == "SMOOTH", (
        f"Expected SMOOTH, got {result['primary_style']} "
        f"(scores: {result['style_scores']})"
    )
    print("PASS  test_classification_smooth_profile")


# ── Test 9 — defence_proof profile classification ────────────────────────────

def test_classification_defence_proof_profile():
    """
    A robot that scores under pressure, escapes defenders quickly, and varies
    its paths must classify as DEFENCE_PROOF.
    """
    metrics = {
        "avg_velocity_px_per_frame":        6.0,
        "velocity_variance":                8.0,
        "max_velocity":                     12.0,
        "velocity_recovery_rate":           1.2,
        "collision_count":                  2,
        "collision_rate_per_minute":        0.8,
        "angular_velocity_variance":        80.0,
        "path_repetition_score":            0.15,  # < 0.40 threshold
        "arc_smoothness":                   0.65,
        "time_in_opponent_half_pct":        0.20,
        "shadowing_events":                 1,
        "avg_distance_to_nearest_opponent": 150.0,
        "scoring_under_pressure_rate":      0.75,  # > 0.60 threshold
        "escape_success_rate":              0.80,  # > 0.50 threshold
        "path_variety_score":               0.85,
    }
    result = classify_driving_style(metrics)
    assert result["primary_style"] == "DEFENCE_PROOF", (
        f"Expected DEFENCE_PROOF, got {result['primary_style']} "
        f"(scores: {result['style_scores']})"
    )
    print("PASS  test_classification_defence_proof_profile")


# ── Test 10 — dual style assignment ──────────────────────────────────────────

def test_dual_style_assignment():
    """
    A robot scoring in both the SMOOTH and DEFENCE_PROOF ranges should receive
    a secondary style label.
    """
    metrics = {
        "avg_velocity_px_per_frame":        5.5,
        "velocity_variance":                1.8,    # SMOOTH signal
        "max_velocity":                     9.0,
        "velocity_recovery_rate":           1.0,
        "collision_count":                  1,
        "collision_rate_per_minute":        0.2,    # SMOOTH signal
        "angular_velocity_variance":        25.0,   # SMOOTH signal
        "path_repetition_score":            0.20,   # DEFENCE_PROOF signal
        "arc_smoothness":                   0.88,   # SMOOTH signal
        "time_in_opponent_half_pct":        0.10,
        "shadowing_events":                 0,
        "avg_distance_to_nearest_opponent": 250.0,
        "scoring_under_pressure_rate":      0.65,   # DEFENCE_PROOF signal
        "escape_success_rate":              0.70,   # DEFENCE_PROOF signal
        "path_variety_score":               0.80,   # DEFENCE_PROOF signal
    }
    result = classify_driving_style(metrics)
    assert result["secondary_style"] is not None, (
        f"Expected a secondary style, got None "
        f"(primary={result['primary_style']}, scores={result['style_scores']})"
    )
    valid_styles = {"DEFENSIVE", "RECKLESS", "SMOOTH", "DEFENCE_PROOF"}
    assert result["primary_style"]   in valid_styles
    assert result["secondary_style"] in valid_styles
    assert result["primary_style"] != result["secondary_style"]
    print("PASS  test_dual_style_assignment")


# ── Test 11 — thresholds load from config ─────────────────────────────────────

def test_thresholds_load_from_config():
    """
    classify_driving_style() must use thresholds from configs/field_config.json,
    not hardcoded values. Verify the config is present and contains all required keys.
    """
    config = _load_config()

    assert "driving_classification" in config, (
        "driving_classification block missing from field_config.json"
    )
    dc = config["driving_classification"]

    for section in ["defensive", "reckless", "smooth", "defence_proof"]:
        assert section in dc, f"Missing section '{section}' in driving_classification"

    assert "min_time_opponent_half_pct" in dc["defensive"]
    assert "min_shadowing_events"       in dc["defensive"]
    assert "max_personal_score_rate"    in dc["defensive"]

    assert "min_collision_rate_per_min"        in dc["reckless"]
    assert "max_collision_rate_per_min"        in dc["smooth"]
    assert "min_scoring_under_pressure_rate"   in dc["defence_proof"]
    assert "min_escape_success_rate"           in dc["defence_proof"]
    assert "max_path_repetition_score"         in dc["defence_proof"]

    print("PASS  test_thresholds_load_from_config")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_velocity_computation_accuracy,
        test_collision_detection_no_false_positives,
        test_shadowing_detection_min_duration,
        test_path_repetition_identical_paths,
        test_path_repetition_random_paths,
        test_classification_defensive_profile,
        test_classification_reckless_profile,
        test_classification_smooth_profile,
        test_classification_defence_proof_profile,
        test_dual_style_assignment,
        test_thresholds_load_from_config,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL  {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"PHASE 13 COMPLETE — {passed}/{len(tests)} tests passed")
    if failed:
        print(f"  {failed} FAILED — fix before proceeding.")
    print("="*50)
    sys.exit(0 if failed == 0 else 1)
