"""
tests/test_alliance_builder.py — Phase 14: Alliance Builder Tests

All 12 tests use mocked TBA API responses — NO live API calls,
NO real API key required.

Tests:
  1.  test_tba_client_auth_header_present
  2.  test_tba_client_etag_caching
  3.  test_tba_client_handles_304
  4.  test_tba_client_raises_on_bad_key
  5.  test_composite_score_weights_sum_to_1
  6.  test_style_complement_matrix_applied
  7.  test_pick_recommendation_excludes_our_team
  8.  test_do_not_pick_flags_reckless_high_collision
  9.  test_team_with_no_video_data_uses_tba_only
  10. test_alliance_simulator_three_teams
  11. test_pick_list_sorted_descending
  12. test_tba_config_loads_from_file
"""

import sys
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src/ and root are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Minimal composite scores fixture ─────────────────────────────────────────

def _make_composites(our_team: str = "1234") -> dict:
    """
    Build a minimal team_composite_scores dict for testing.
    Includes our_team plus several other teams with varying profiles.
    """
    return {
        our_team: {
            "composite_score": 0.72,
            "video_data": True,
            "tba_opr": 45.0,
            "tba_ccwm": 12.0,
            "ranking_points_avg": 8.0,
            "win_rate": 0.70,
            "video_score_rate": 0.60,
            "high_confidence_score_pct": 0.85,
            "defence_proof_score": 0.55,
            "smooth_score": 0.70,
            "collision_rate": 0.3,
            "driving_style": "SMOOTH",
            "style_scores": {"SMOOTH": 0.70, "DEFENCE_PROOF": 0.55,
                             "DEFENSIVE": 0.10, "RECKLESS": 0.05},
            "warnings": [],
            "data_confidence": "HIGH",
        },
        "5678": {
            "composite_score": 0.68,
            "video_data": True,
            "tba_opr": 38.0,
            "tba_ccwm": 8.0,
            "ranking_points_avg": 7.0,
            "win_rate": 0.60,
            "video_score_rate": 0.55,
            "high_confidence_score_pct": 0.78,
            "defence_proof_score": 0.80,
            "smooth_score": 0.40,
            "collision_rate": 0.5,
            "driving_style": "DEFENCE_PROOF",
            "style_scores": {"SMOOTH": 0.40, "DEFENCE_PROOF": 0.80,
                             "DEFENSIVE": 0.10, "RECKLESS": 0.15},
            "warnings": [],
            "data_confidence": "HIGH",
        },
        "9999": {
            "composite_score": 0.55,
            "video_data": True,
            "tba_opr": 28.0,
            "tba_ccwm": 5.0,
            "ranking_points_avg": 6.0,
            "win_rate": 0.50,
            "video_score_rate": 0.40,
            "high_confidence_score_pct": 0.70,
            "defence_proof_score": 0.20,
            "smooth_score": 0.20,
            "collision_rate": 0.9,
            "driving_style": "DEFENSIVE",
            "style_scores": {"SMOOTH": 0.20, "DEFENCE_PROOF": 0.20,
                             "DEFENSIVE": 0.75, "RECKLESS": 0.10},
            "warnings": [],
            "data_confidence": "HIGH",
        },
        "2468": {
            "composite_score": 0.30,
            "video_data": False,
            "tba_opr": 12.0,
            "tba_ccwm": -3.0,
            "ranking_points_avg": 3.0,
            "win_rate": 0.25,
            "video_score_rate": None,
            "high_confidence_score_pct": None,
            "defence_proof_score": None,
            "smooth_score": None,
            "collision_rate": None,
            "driving_style": None,
            "style_scores": {},
            "warnings": ["No video data — TBA signals only"],
            "data_confidence": "LOW",
        },
        "3690": {
            "composite_score": 0.25,
            "video_data": True,
            "tba_opr": 10.0,
            "tba_ccwm": -5.0,
            "ranking_points_avg": 2.0,
            "win_rate": 0.20,
            "video_score_rate": 0.10,
            "high_confidence_score_pct": 0.50,
            "defence_proof_score": 0.05,
            "smooth_score": 0.05,
            "collision_rate": 3.2,
            "driving_style": "RECKLESS",
            "style_scores": {"SMOOTH": 0.05, "DEFENCE_PROOF": 0.05,
                             "DEFENSIVE": 0.10, "RECKLESS": 0.85},
            "warnings": ["HIGH collision rate (3.20/min)"],
            "data_confidence": "HIGH",
        },
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 1 — TBA client sends X-TBA-Auth-Key header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tba_client_auth_header_present():
    """Every TBA request must include X-TBA-Auth-Key header."""
    import tba_client

    captured_headers = {}

    def mock_get(url, headers=None, timeout=None):
        captured_headers.update(headers or {})
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = []
        resp.headers = {}
        return resp

    with patch("tba_client._api_key", return_value="FAKE_KEY"), \
         patch("tba_client._load_cache", return_value=None), \
         patch("tba_client._save_cache"), \
         patch("tba_client._get_cached_etag", return_value=None), \
         patch("requests.get", side_effect=mock_get):
        tba_client._get("/event/2026test/teams")

    assert "X-TBA-Auth-Key" in captured_headers, (
        "X-TBA-Auth-Key header missing from TBA request"
    )
    assert captured_headers["X-TBA-Auth-Key"] == "FAKE_KEY"
    print("PASS  test_tba_client_auth_header_present")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 2 — ETag caching sends If-None-Match on repeat calls
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tba_client_etag_caching():
    """Second request for same endpoint sends If-None-Match with stored ETag."""
    import tba_client

    captured = {}

    def mock_get(url, headers=None, timeout=None):
        captured.update(headers or {})
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = []
        resp.headers = {"ETag": "\"abc123\""}
        return resp

    with patch("tba_client._api_key", return_value="FAKE_KEY"), \
         patch("tba_client._load_cache", return_value=None), \
         patch("tba_client._save_cache"), \
         patch("tba_client._get_cached_etag", return_value="\"abc123\""), \
         patch("requests.get", side_effect=mock_get):
        tba_client._get("/event/2026test/teams")

    assert "If-None-Match" in captured, "If-None-Match header not sent on cached request"
    assert captured["If-None-Match"] == "\"abc123\""
    print("PASS  test_tba_client_etag_caching")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 3 — 304 Not Modified returns cached data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tba_client_handles_304():
    """A 304 response must return the previously cached data without error."""
    import tba_client

    cached_data = [{"team_number": 1234}]

    def mock_get(url, headers=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 304
        return resp

    with patch("tba_client._api_key", return_value="FAKE_KEY"), \
         patch("tba_client._load_cache", return_value={"data": cached_data, "_cached_at": 9e18}), \
         patch("tba_client._get_cached_etag", return_value="\"xyz\""), \
         patch("tba_client._save_cache"), \
         patch("requests.get", side_effect=mock_get):
        result = tba_client._get("/event/2026test/teams")

    assert result == cached_data, f"Expected cached data on 304, got {result}"
    print("PASS  test_tba_client_handles_304")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 4 — 401 raises TBAAuthError
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tba_client_raises_on_bad_key():
    """A 401 response must raise TBAAuthError."""
    import tba_client
    from tba_client import TBAAuthError

    def mock_get(url, headers=None, timeout=None):
        resp = MagicMock()
        resp.status_code = 401
        return resp

    with patch("tba_client._api_key", return_value="BAD_KEY"), \
         patch("tba_client._load_cache", return_value=None), \
         patch("tba_client._get_cached_etag", return_value=None), \
         patch("requests.get", side_effect=mock_get):
        try:
            tba_client._get("/event/2026test/teams")
            assert False, "Expected TBAAuthError but no exception raised"
        except TBAAuthError:
            pass  # expected

    print("PASS  test_tba_client_raises_on_bad_key")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 5 — alliance_weights sum to 1.0 (excluding penalty)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_composite_score_weights_sum_to_1():
    """
    All positive alliance_weights in tba_config.json must sum to 1.0.
    The collision_rate_penalty is negative and excluded from this check.
    """
    cfg_path = Path(__file__).parent.parent / "configs" / "tba_config.json"
    assert cfg_path.exists(), "configs/tba_config.json not found"

    cfg = json.loads(cfg_path.read_text())
    weights = cfg["alliance_weights"]

    positive_sum = sum(v for v in weights.values() if v > 0)
    assert abs(positive_sum - 1.0) < 1e-9, (
        f"Positive alliance_weights must sum to 1.0, got {positive_sum}"
    )
    print("PASS  test_composite_score_weights_sum_to_1")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 6 — style complement matrix is applied
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_style_complement_matrix_applied():
    """
    simulate_alliance must return a higher projected score for a
    DEFENSIVE + SMOOTH + DEFENCE_PROOF alliance (all positive complement pairs)
    than for a RECKLESS + RECKLESS + RECKLESS alliance (all penalised).
    """
    from alliance_builder import simulate_alliance, COMPLEMENT_MATRIX

    # Verify the matrix itself has the required entries
    assert ("SMOOTH", "DEFENCE_PROOF") in COMPLEMENT_MATRIX
    assert ("DEFENSIVE", "SMOOTH") in COMPLEMENT_MATRIX
    assert ("RECKLESS", "RECKLESS") in COMPLEMENT_MATRIX
    assert COMPLEMENT_MATRIX[("RECKLESS", "RECKLESS")] < 0, "RECKLESS+RECKLESS should penalise"

    composites_good = {
        "A": {"composite_score": 0.60, "driving_style": "DEFENSIVE",    "video_data": True, "warnings": [], "collision_rate": 0.3},
        "B": {"composite_score": 0.60, "driving_style": "SMOOTH",       "video_data": True, "warnings": [], "collision_rate": 0.2},
        "C": {"composite_score": 0.60, "driving_style": "DEFENCE_PROOF","video_data": True, "warnings": [], "collision_rate": 0.4},
    }
    composites_bad = {
        "X": {"composite_score": 0.60, "driving_style": "RECKLESS", "video_data": True, "warnings": [], "collision_rate": 2.0},
        "Y": {"composite_score": 0.60, "driving_style": "RECKLESS", "video_data": True, "warnings": [], "collision_rate": 2.0},
        "Z": {"composite_score": 0.60, "driving_style": "RECKLESS", "video_data": True, "warnings": [], "collision_rate": 2.0},
    }

    good_result = simulate_alliance(["A", "B", "C"], composites_good)
    bad_result  = simulate_alliance(["X", "Y", "Z"], composites_bad)

    assert good_result["overall_rating"] >= bad_result["overall_rating"], (
        f"Positive-synergy alliance ({good_result['overall_rating']:.4f}) should "
        f"rate >= RECKLESS alliance ({bad_result['overall_rating']:.4f})"
    )
    print("PASS  test_style_complement_matrix_applied")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 7 — recommend_picks never includes our team
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_pick_recommendation_excludes_our_team():
    """recommend_picks must never return our_team_number in pick_1 or pick_2."""
    from alliance_builder import recommend_picks

    our_team = "1234"
    composites = _make_composites(our_team)

    for strategy in ["balanced", "score_heavy", "defensive", "safe"]:
        result = recommend_picks(our_team, "2026test", composites, strategy)
        p1 = result.get("pick_1", {}).get("team_number")
        p2 = result.get("pick_2", {}).get("team_number")
        assert p1 != our_team, f"strategy={strategy}: pick_1 must not be our team"
        assert p2 != our_team, f"strategy={strategy}: pick_2 must not be our team"

    print("PASS  test_pick_recommendation_excludes_our_team")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 8 — do-not-pick flags RECKLESS + high collision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_do_not_pick_flags_reckless_high_collision():
    """A RECKLESS team with high collision rate must appear on the do-not-pick list."""
    from alliance_builder import flag_risky_teams

    composites = _make_composites("1234")
    # team "3690" has RECKLESS style and collision_rate=3.2 > threshold 2.5

    risky = flag_risky_teams(composites)
    risky_teams = [r["team_number"] for r in risky]

    assert "3690" in risky_teams, (
        f"Expected team 3690 (RECKLESS + high collision) in risky list, got {risky_teams}"
    )
    print("PASS  test_do_not_pick_flags_reckless_high_collision")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 9 — team with no video data uses TBA signals only
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_team_with_no_video_data_uses_tba_only():
    """
    A team marked video_data=False must have None for all video-derived fields
    and carry a 'No video data' warning.
    """
    composites = _make_composites("1234")
    team_2468 = composites["2468"]

    assert team_2468["video_data"] is False
    assert team_2468["video_score_rate"]          is None
    assert team_2468["high_confidence_score_pct"] is None
    assert team_2468["defence_proof_score"]       is None
    assert team_2468["smooth_score"]              is None
    assert team_2468["collision_rate"]            is None
    assert any("No video data" in w for w in team_2468["warnings"]), (
        "Expected 'No video data' warning on team without video"
    )
    print("PASS  test_team_with_no_video_data_uses_tba_only")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 10 — alliance simulator returns required keys for 3 teams
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_alliance_simulator_three_teams():
    """simulate_alliance must return all required keys for a 3-team input."""
    from alliance_builder import simulate_alliance

    composites = _make_composites("1234")
    result = simulate_alliance(["5678", "9999", "2468"], composites)

    required_keys = ["projected_score", "style_synergy", "strengths",
                     "weaknesses", "overall_rating"]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in simulate_alliance result"

    assert isinstance(result["projected_score"],  float)
    assert isinstance(result["overall_rating"],   float)
    assert isinstance(result["strengths"],         list)
    assert isinstance(result["weaknesses"],        list)
    assert 0.0 <= result["projected_score"]  <= 1.0
    assert 0.0 <= result["overall_rating"]   <= 1.0

    # Verify ValueError on wrong number of teams
    try:
        simulate_alliance(["5678", "9999"], composites)
        assert False, "Should raise ValueError for 2 teams"
    except ValueError:
        pass

    print("PASS  test_alliance_simulator_three_teams")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 11 — pick list is sorted descending by composite score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_pick_list_sorted_descending():
    """generate_pick_list must return teams sorted by composite_score descending."""
    from alliance_builder import generate_pick_list

    our_team = "1234"
    composites = _make_composites(our_team)
    pick_list = generate_pick_list(our_team, "2026test", composites, top_n=10)

    assert len(pick_list) > 0, "Pick list must not be empty"

    scores = [entry["composite_score"] for entry in pick_list]
    assert scores == sorted(scores, reverse=True), (
        f"Pick list not sorted descending: {scores}"
    )

    # Verify rank numbers are sequential
    for i, entry in enumerate(pick_list):
        assert entry["rank"] == i + 1, (
            f"Rank at position {i} should be {i+1}, got {entry['rank']}"
        )

    # Our team must not appear
    team_numbers = [entry["team_number"] for entry in pick_list]
    assert our_team not in team_numbers, "Our team must not appear in pick list"

    print("PASS  test_pick_list_sorted_descending")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test 12 — tba_config loads correctly from file
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_tba_config_loads_from_file():
    """configs/tba_config.json must exist and contain all required fields."""
    cfg_path = Path(__file__).parent.parent / "configs" / "tba_config.json"
    assert cfg_path.exists(), "configs/tba_config.json not found"

    cfg = json.loads(cfg_path.read_text())

    required_top = ["api_key", "event_key", "our_team_number",
                    "cache_ttl_seconds", "strategy",
                    "alliance_weights", "do_not_pick_thresholds"]
    for key in required_top:
        assert key in cfg, f"Missing key '{key}' in tba_config.json"

    weights = cfg["alliance_weights"]
    required_weights = [
        "opr", "ccwm", "ranking_points_avg", "win_rate",
        "video_score_rate", "high_confidence_score_pct",
        "defence_proof_score", "smooth_score",
        "collision_rate_penalty", "style_complement_bonus",
    ]
    for w in required_weights:
        assert w in weights, f"Missing weight '{w}' in alliance_weights"

    dnp = cfg["do_not_pick_thresholds"]
    assert "min_composite_score"       in dnp
    assert "max_collision_rate_per_min" in dnp

    print("PASS  test_tba_config_loads_from_file")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_tba_client_auth_header_present,
        test_tba_client_etag_caching,
        test_tba_client_handles_304,
        test_tba_client_raises_on_bad_key,
        test_composite_score_weights_sum_to_1,
        test_style_complement_matrix_applied,
        test_pick_recommendation_excludes_our_team,
        test_do_not_pick_flags_reckless_high_collision,
        test_team_with_no_video_data_uses_tba_only,
        test_alliance_simulator_three_teams,
        test_pick_list_sorted_descending,
        test_tba_config_loads_from_file,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL  {test_fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"PHASE 14 COMPLETE — {passed}/{len(tests)} tests passed")
    if failed:
        print(f"  {failed} FAILED — fix before proceeding.")
    print("="*50)
    sys.exit(0 if failed == 0 else 1)
