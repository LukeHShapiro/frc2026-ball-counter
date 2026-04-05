"""
tests/test_count.py - Phase 12: Score Aggregation Tests

Tests:
  1. test_alliance_sum_matches_scoreboard
  2. test_confidence_bucketing
  3. test_unattributed_flagging
  4. test_discrepancy_report_triggers
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from count import aggregate_scores, generate_accuracy_report


def _event(team, conf, flag=""):
    return {
        "frame_id": 1, "timestamp_s": 0.0,
        "team_number": team, "method": "test",
        "confidence": conf, "zone": "red_goal",
        "notes": "", "case": 1, "flag": flag,
        "ball_track_id": 0,
    }


# ---- Test 1 -----------------------------------------------------------------

def test_alliance_sum_matches_scoreboard():
    """
    Sum of attributed robot scores must equal scoreboard totals
    when there are no gaps.
    """
    timeline = [
        _event("1234", 0.95),
        _event("1234", 0.90),
        _event("5678", 0.88),
        _event("9999", 0.92),
    ]
    scores = aggregate_scores(timeline)

    assert scores["1234"]["score"] == 2
    assert scores["5678"]["score"] == 1
    assert scores["9999"]["score"] == 1
    assert "UNATTRIBUTED" not in scores
    print("PASS  test_alliance_sum_matches_scoreboard")


# ---- Test 2 -----------------------------------------------------------------

def test_confidence_bucketing():
    """Events must land in the correct confidence bucket."""
    timeline = [
        _event("1234", 0.95),   # high
        _event("1234", 0.70),   # med
        _event("1234", 0.30),   # low
    ]
    scores = aggregate_scores(timeline)

    assert scores["1234"]["high_conf"] == 1, f"Got {scores['1234']}"
    assert scores["1234"]["med_conf"]  == 1
    assert scores["1234"]["low_conf"]  == 1
    print("PASS  test_confidence_bucketing")


# ---- Test 3 -----------------------------------------------------------------

def test_unattributed_flagging():
    """Events flagged UNATTRIBUTED must go to the UNATTRIBUTED bucket."""
    timeline = [
        _event("1234",         0.95),
        _event("UNATTRIBUTED", 0.0, flag="UNATTRIBUTED"),
        _event("UNATTRIBUTED", 0.0, flag="UNATTRIBUTED"),
    ]
    scores = aggregate_scores(timeline)

    assert "UNATTRIBUTED" in scores,             "Expected UNATTRIBUTED bucket"
    assert scores["UNATTRIBUTED"]["unattributed"] == 2
    assert scores["1234"]["score"] == 1
    print("PASS  test_unattributed_flagging")


# ---- Test 4 -----------------------------------------------------------------

def test_discrepancy_report_triggers():
    """
    When scoreboard_validation shows a gap, discrepancy must be True
    and discrepancy_details must be non-empty.
    """
    timeline = [_event("1234", 0.9), _event("5678", 0.9)]
    scores   = aggregate_scores(timeline)

    # Simulate: scoreboard says red=5, we only attributed 2
    validation = {
        "red_attributed":  2, "red_scoreboard":  5, "red_gap":  3,
        "blue_attributed": 1, "blue_scoreboard": 1, "blue_gap": 0,
        "balanced": False,
    }
    report = generate_accuracy_report(scores, scoreboard_validation=validation)

    assert report["discrepancy"],                   "Expected discrepancy=True"
    assert report["discrepancy_details"] != "",     "Expected non-empty details"
    assert "Red" in report["discrepancy_details"]
    print("PASS  test_discrepancy_report_triggers")


# ---- Runner -----------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_alliance_sum_matches_scoreboard,
        test_confidence_bucketing,
        test_unattributed_flagging,
        test_discrepancy_report_triggers,
    ]
    passed = failed = 0
    for fn in tests:
        try:
            fn(); passed += 1
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"FAIL  {fn.__name__}: {e}"); failed += 1
    print(f"\n{'='*50}")
    print(f"PHASE 12 count -- {passed}/{len(tests)} tests passed")
    print("=" * 50)
    import sys; sys.exit(0 if failed == 0 else 1)
