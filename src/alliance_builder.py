"""
src/alliance_builder.py — Phase 14: Alliance Recommendation Engine

Merges TBA API data with our per-robot video analysis (Phases 9 + 13) to
produce a ranked pick list and alliance recommendations.

All weights loaded from configs/tba_config.json — never hardcoded.

Depends on:
  - src/tba_client.py            (Phase 14 TBA API)
  - Phase 9 scoring attribution  (final_scores, score_timeline)
  - Phase 13 driving analysis    (driving_report)
  - configs/tba_config.json      (ALLIANCE_WEIGHTS, do_not_pick_thresholds)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── Config ────────────────────────────────────────────────────────────────────

def _load_tba_config() -> dict:
    cfg_path = Path(__file__).parent.parent / "configs" / "tba_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            "configs/tba_config.json not found. Complete INPUT CHECKPOINT #8."
        )
    with open(cfg_path) as f:
        return json.load(f)


# ── Style complement matrix (exactly as specified in CLAUDE.md Phase 14) ──────

COMPLEMENT_MATRIX: dict[tuple[str, str], float] = {
    ("SMOOTH",        "DEFENCE_PROOF"): +0.08,
    ("DEFENCE_PROOF", "SMOOTH"):        +0.08,  # symmetric
    ("DEFENSIVE",     "SMOOTH"):        +0.10,
    ("SMOOTH",        "DEFENSIVE"):     +0.10,  # symmetric
    ("DEFENSIVE",     "DEFENCE_PROOF"): +0.06,
    ("DEFENCE_PROOF", "DEFENSIVE"):     +0.06,  # symmetric
    ("SMOOTH",        "SMOOTH"):        +0.04,
    ("RECKLESS",      "DEFENSIVE"):     -0.05,
    ("DEFENSIVE",     "RECKLESS"):      -0.05,  # symmetric
    ("RECKLESS",      "RECKLESS"):      -0.10,
}


def _style_complement_bonus(styles: list[str]) -> float:
    """
    Compute the total style complement bonus for a list of robot styles.

    Applies COMPLEMENT_MATRIX pairwise across all robot pairs.

    Args:
        styles: List of driving style strings for the alliance robots.

    Returns:
        Total bonus float (can be negative).
    """
    total = 0.0
    for i in range(len(styles)):
        for j in range(i + 1, len(styles)):
            key = (styles[i], styles[j])
            total += COMPLEMENT_MATRIX.get(key, 0.0)
    return total


# ── Main scoring function ─────────────────────────────────────────────────────

def build_team_composite_scores(
    event_key: str,
    our_team_number: str,
    video_analysis_results: dict,
    driving_results: dict,
) -> dict[str, dict]:
    """
    Compute a composite score for every team at the event, merging TBA + video data.

    For teams without video analysis, uses TBA data only and flags
    "video_data": False.

    Args:
        event_key:             TBA event key.
        our_team_number:       Our team (excluded from picks but included in scores).
        video_analysis_results: {"final_scores": ..., "score_timeline": ...}
        driving_results:       Output of generate_driving_report() from Phase 13.

    Returns:
        {team_number: composite_score_dict}
        Each dict contains: composite_score, video_data, tba_opr, tba_ccwm,
        ranking_points_avg, win_rate, video_score_rate, high_confidence_score_pct,
        defence_proof_score, smooth_score, collision_rate, driving_style, warnings.

    Depends on: src/tba_client.py, Phase 9, Phase 13.
    """
    from tba_client import (
        get_event_teams, get_event_rankings, get_event_oprs, get_event_matches
    )

    cfg = _load_tba_config()
    weights: dict[str, float] = cfg["alliance_weights"]

    # ── Fetch TBA data ────────────────────────────────────────────────────────
    try:
        teams        = get_event_teams(event_key)
        rankings     = get_event_rankings(event_key)
        oprs_data    = get_event_oprs(event_key)
        matches      = get_event_matches(event_key)
    except Exception as e:
        print(f"  [Alliance Builder] TBA fetch warning: {e}")
        teams = []; rankings = []; oprs_data = {}; matches = []

    oprs  = oprs_data.get("oprs",  {})
    dprs  = oprs_data.get("dprs",  {})
    ccwms = oprs_data.get("ccwms", {})

    # Build ranking lookup
    rank_lookup: dict[str, dict] = {}
    total_matches = 1
    for r in rankings:
        tn = str(r["team_number"])
        rank_lookup[tn] = r
        wins   = r.get("wins", 0)
        losses = r.get("losses", 0)
        ties   = r.get("ties", 0)
        total_matches = max(total_matches, wins + losses + ties)

    # Video analysis lookup
    final_scores   = video_analysis_results.get("final_scores", {})
    score_timeline = video_analysis_results.get("score_timeline", [])

    # Total frames (for score-per-match approximation)
    total_events = max(len(score_timeline), 1)

    composites: dict[str, dict] = {}

    team_numbers = [str(t["team_number"]) for t in teams]
    if not team_numbers:
        # Fall back to teams with video data if TBA unavailable
        team_numbers = list(final_scores.keys()) + list(driving_results.keys())
        team_numbers = list(set(team_numbers))

    for team in team_numbers:
        team_key = f"frc{team}"
        has_video = team in final_scores or team in driving_results

        # ── TBA signals ───────────────────────────────────────────────────────
        tba_opr  = oprs.get(team_key, 0.0)
        tba_ccwm = ccwms.get(team_key, 0.0)

        rank_info = rank_lookup.get(team, {})
        wins   = rank_info.get("wins",   0)
        losses = rank_info.get("losses", 0)
        ties   = rank_info.get("ties",   0)
        total  = wins + losses + ties or 1
        win_rate = wins / total
        rp_avg   = rank_info.get("ranking_points", 0) / total

        # Normalise OPR/CCWM to 0–1 (rough scale: 60 pts = max)
        opr_norm  = min(1.0, max(0.0, tba_opr  / 60.0))
        ccwm_norm = min(1.0, max(0.0, (tba_ccwm + 30) / 60.0))
        rp_norm   = min(1.0, rp_avg / 20.0)

        # ── Video signals (only if video_data=True) ───────────────────────────
        video_score_rate          = 0.0
        high_conf_pct             = 0.0
        defence_proof_score       = 0.0
        smooth_score              = 0.0
        collision_penalty         = 0.0
        driving_style             = None
        style_scores: dict        = {}
        collision_rate            = 0.0

        if has_video:
            score_data = final_scores.get(team, {})
            total_scores = score_data.get("score", 0)
            high_conf    = score_data.get("high_conf", 0)
            high_conf_pct = high_conf / max(total_scores, 1)
            video_score_rate = min(1.0, total_scores / 20.0)

            driving_data = driving_results.get(team, {})
            driving_style = driving_data.get("style")
            style_scores  = driving_data.get("style_scores", {})
            metrics       = driving_data.get("metrics", {})

            defence_proof_score = style_scores.get("DEFENCE_PROOF", 0.0)
            smooth_score        = style_scores.get("SMOOTH", 0.0)
            collision_rate      = metrics.get("collision_rate_per_minute", 0.0)
            max_col_rate        = cfg["do_not_pick_thresholds"].get(
                "max_collision_rate_per_min", 2.5
            )
            collision_penalty   = min(1.0, collision_rate / max_col_rate)

        # ── Composite score ───────────────────────────────────────────────────
        video_multiplier = 1.0 if has_video else 0.0

        composite = (
            weights["opr"]                       * opr_norm
          + weights["ccwm"]                      * ccwm_norm
          + weights["ranking_points_avg"]        * rp_norm
          + weights["win_rate"]                  * win_rate
          + weights["video_score_rate"]          * video_score_rate          * video_multiplier
          + weights["high_confidence_score_pct"] * high_conf_pct             * video_multiplier
          + weights["defence_proof_score"]       * defence_proof_score       * video_multiplier
          + weights["smooth_score"]              * smooth_score              * video_multiplier
          + weights["collision_rate_penalty"]    * collision_penalty         * video_multiplier
        )
        # Clamp to [0, 1]
        composite = min(1.0, max(0.0, composite))

        warnings: list[str] = []
        if collision_rate > cfg["do_not_pick_thresholds"].get("max_collision_rate_per_min", 2.5):
            warnings.append(f"HIGH collision rate ({collision_rate:.2f}/min)")
        if not has_video:
            warnings.append("No video data — TBA signals only")

        composites[team] = {
            "composite_score":          round(composite, 4),
            "video_data":               has_video,
            "tba_opr":                  tba_opr,
            "tba_ccwm":                 tba_ccwm,
            "ranking_points_avg":       rp_avg,
            "win_rate":                 round(win_rate, 3),
            "video_score_rate":         round(video_score_rate, 3) if has_video else None,
            "high_confidence_score_pct":round(high_conf_pct, 3) if has_video else None,
            "defence_proof_score":      round(defence_proof_score, 3) if has_video else None,
            "smooth_score":             round(smooth_score, 3) if has_video else None,
            "collision_rate":           round(collision_rate, 3) if has_video else None,
            "driving_style":            driving_style,
            "style_scores":             style_scores,
            "warnings":                 warnings,
            "data_confidence":          "HIGH" if has_video else ("MEDIUM" if tba_opr > 0 else "LOW"),
        }

    return composites


def recommend_picks(
    our_team_number: str,
    event_key: str,
    team_composite_scores: dict[str, dict],
    strategy: str = "balanced",
) -> dict:
    """
    Recommend Pick 1 and Pick 2 for our alliance.

    Strategy options:
      "balanced"     — maximise overall composite score
      "score_heavy"  — prioritise highest composite, ignore style
      "defensive"    — require at least one DEFENSIVE robot as pick 2
      "safe"         — prioritise low collision rate + HIGH data confidence

    Our team is NEVER included in picks.

    Args:
        our_team_number:       Our team number (excluded).
        event_key:             TBA event key (informational).
        team_composite_scores: Output of build_team_composite_scores().
        strategy:              One of the four strategy strings above.

    Returns:
        {pick_1: {...}, pick_2: {...}, projected_alliance_score: float,
         style_synergy: str, warnings: [str]}

    Depends on: build_team_composite_scores().
    """
    cfg = _load_tba_config()

    eligible = {
        tn: data for tn, data in team_composite_scores.items()
        if str(tn) != str(our_team_number)
    }

    our_data     = team_composite_scores.get(str(our_team_number), {})
    our_style    = our_data.get("driving_style") or "SMOOTH"
    our_composite = our_data.get("composite_score", 0.5)

    def _score_for_strategy(team: str, data: dict, pick_num: int) -> float:
        base = data["composite_score"]
        if strategy == "score_heavy":
            return base
        if strategy == "safe":
            conf_bonus = 0.1 if data.get("data_confidence") == "HIGH" else 0.0
            col_penalty = (data.get("collision_rate") or 0) * 0.05
            return base + conf_bonus - col_penalty
        if strategy == "defensive" and pick_num == 2:
            if data.get("driving_style") == "DEFENSIVE":
                return base + 0.15
        # balanced: add style complement bonus
        candidate_style = data.get("driving_style") or ""
        complement = COMPLEMENT_MATRIX.get((our_style, candidate_style), 0.0)
        return base + complement

    sorted_eligible = sorted(
        eligible.items(),
        key=lambda kv: _score_for_strategy(kv[0], kv[1], 1),
        reverse=True,
    )

    pick1_team, pick1_data = sorted_eligible[0] if sorted_eligible else (None, {})
    pick1_style = (pick1_data.get("driving_style") or "") if pick1_data else ""

    # Pick 2: exclude pick 1, re-score considering pick 1 complement
    remaining = [(tn, d) for tn, d in sorted_eligible[1:]]
    pick2_team, pick2_data = (None, {})
    if remaining:
        if strategy == "defensive":
            defenders = [(tn, d) for tn, d in remaining
                         if d.get("driving_style") == "DEFENSIVE"]
            pool = defenders if defenders else remaining
        else:
            pool = remaining
        pool_scored = sorted(
            pool,
            key=lambda kv: (
                kv[1]["composite_score"]
                + COMPLEMENT_MATRIX.get((pick1_style, kv[1].get("driving_style") or ""), 0.0)
                + COMPLEMENT_MATRIX.get((our_style, kv[1].get("driving_style") or ""), 0.0)
            ),
            reverse=True,
        )
        pick2_team, pick2_data = pool_scored[0]

    # Projected alliance score
    projected = (
        our_composite
        + (pick1_data.get("composite_score", 0) if pick1_data else 0)
        + (pick2_data.get("composite_score", 0) if pick2_data else 0)
    ) / 3.0

    # Style synergy string
    styles_in_alliance = [
        s for s in [our_style, pick1_style,
                    (pick2_data.get("driving_style") or "") if pick2_data else ""]
        if s
    ]
    synergy = " + ".join(styles_in_alliance) if styles_in_alliance else "N/A"

    # Collect warnings
    all_warnings: list[str] = []
    for pick_team, pick_data in [(pick1_team, pick1_data), (pick2_team, pick2_data)]:
        if pick_data:
            all_warnings.extend(pick_data.get("warnings", []))

    def _reasoning(team: str, data: dict) -> list[str]:
        if not data:
            return []
        reasons = []
        reasons.append(f"Composite score: {data.get('composite_score', 0):.3f}")
        if data.get("tba_opr"):
            reasons.append(f"OPR: {data['tba_opr']:.1f}")
        if data.get("driving_style"):
            reasons.append(f"Driving style: {data['driving_style']}")
        if data.get("video_data"):
            reasons.append(f"Video score rate: {data.get('video_score_rate', 0):.2f}")
        if data.get("data_confidence"):
            reasons.append(f"Data confidence: {data['data_confidence']}")
        return reasons[:5]

    return {
        "pick_1": {
            "team_number":     pick1_team,
            "composite_score": pick1_data.get("composite_score") if pick1_data else None,
            "reasoning":       _reasoning(pick1_team, pick1_data),
            "data_confidence": pick1_data.get("data_confidence") if pick1_data else "LOW",
            "tba_opr":         pick1_data.get("tba_opr") if pick1_data else None,
            "video_score_rate":pick1_data.get("video_score_rate") if pick1_data else None,
            "driving_style":   pick1_data.get("driving_style") if pick1_data else None,
        } if pick1_team else {},
        "pick_2": {
            "team_number":     pick2_team,
            "composite_score": pick2_data.get("composite_score") if pick2_data else None,
            "reasoning":       _reasoning(pick2_team, pick2_data),
            "data_confidence": pick2_data.get("data_confidence") if pick2_data else "LOW",
            "tba_opr":         pick2_data.get("tba_opr") if pick2_data else None,
            "video_score_rate":pick2_data.get("video_score_rate") if pick2_data else None,
            "driving_style":   pick2_data.get("driving_style") if pick2_data else None,
        } if pick2_team else {},
        "projected_alliance_score": round(projected, 4),
        "style_synergy":            synergy,
        "warnings":                 all_warnings,
    }


def recommend_do_not_pick(
    team_composite_scores: dict[str, dict],
    reason_threshold: float = 0.40,
) -> list[dict]:
    """
    Return a list of teams to avoid and why.

    A team makes the do-not-pick list if its composite_score < reason_threshold.

    Args:
        team_composite_scores: Output of build_team_composite_scores().
        reason_threshold:      Composite score below which a team is flagged.

    Returns:
        [{team_number, reason, composite_score}]

    Depends on: build_team_composite_scores().
    """
    cfg = _load_tba_config()
    threshold = cfg["do_not_pick_thresholds"].get("min_composite_score", reason_threshold)

    result = []
    for team, data in team_composite_scores.items():
        score = data.get("composite_score", 0)
        reasons = []
        if score < threshold:
            reasons.append(f"Composite score {score:.3f} below threshold {threshold:.2f}")
        col = data.get("collision_rate") or 0
        max_col = cfg["do_not_pick_thresholds"].get("max_collision_rate_per_min", 2.5)
        if col > max_col:
            reasons.append(f"Collision rate {col:.2f}/min exceeds threshold {max_col}")
        if data.get("driving_style") == "RECKLESS" and (data.get("tba_ccwm") or 0) < 0:
            reasons.append("RECKLESS style + negative CCWM (aggressive but not winning)")
        if not reasons:
            continue
        result.append({
            "team_number":     team,
            "reason":          "; ".join(reasons),
            "composite_score": score,
        })

    return sorted(result, key=lambda x: x["composite_score"])


def generate_pick_list(
    our_team_number: str,
    event_key: str,
    team_composite_scores: dict[str, dict],
    top_n: int = 10,
) -> list[dict]:
    """
    Generate a full ranked pick list (not just top 2).

    Our team is excluded.

    Args:
        our_team_number:       Our team number.
        event_key:             TBA event key (informational).
        team_composite_scores: Output of build_team_composite_scores().
        top_n:                 How many teams to include.

    Returns:
        [{rank, team_number, composite_score, style, reasoning, warnings,
          tba_opr, video_score_rate, data_confidence, video_data}]

    Depends on: build_team_composite_scores().
    """
    eligible = [
        (tn, data) for tn, data in team_composite_scores.items()
        if str(tn) != str(our_team_number)
    ]
    sorted_teams = sorted(
        eligible, key=lambda kv: kv[1]["composite_score"], reverse=True
    )

    result = []
    for rank, (team, data) in enumerate(sorted_teams[:top_n], start=1):
        result.append({
            "rank":             rank,
            "team_number":      team,
            "composite_score":  data["composite_score"],
            "style":            data.get("driving_style") or "",
            "reasoning":        [
                f"OPR: {data.get('tba_opr', 0):.1f}",
                f"Win rate: {data.get('win_rate', 0):.1%}",
                f"Data confidence: {data.get('data_confidence', 'LOW')}",
            ],
            "warnings":         data.get("warnings", []),
            "tba_opr":          data.get("tba_opr"),
            "video_score_rate": data.get("video_score_rate"),
            "data_confidence":  data.get("data_confidence", "LOW"),
            "video_data":       data.get("video_data", False),
        })

    return result


def compare_teams(
    team_a: str,
    team_b: str,
    team_composite_scores: dict[str, dict],
) -> dict[str, dict]:
    """
    Side-by-side comparison of two teams across all metrics.

    Args:
        team_a:                First team number.
        team_b:                Second team number.
        team_composite_scores: Output of build_team_composite_scores().

    Returns:
        {metric: {team_a: value, team_b: value}}

    Depends on: build_team_composite_scores().
    """
    a = team_composite_scores.get(str(team_a), {})
    b = team_composite_scores.get(str(team_b), {})

    metrics = [
        "composite_score", "tba_opr", "tba_ccwm", "win_rate",
        "ranking_points_avg", "video_score_rate", "high_confidence_score_pct",
        "defence_proof_score", "smooth_score", "collision_rate",
        "driving_style", "data_confidence", "video_data",
    ]

    return {
        m: {
            "team_a": a.get(m, "N/A"),
            "team_b": b.get(m, "N/A"),
        }
        for m in metrics
    }


def simulate_alliance(
    team_list: list[str],
    team_composite_scores: dict[str, dict],
) -> dict:
    """
    Simulate the projected performance of a 3-robot alliance.

    Args:
        team_list:             List of exactly 3 team number strings.
        team_composite_scores: Output of build_team_composite_scores().

    Returns:
        {projected_score, style_synergy, strengths, weaknesses, overall_rating}

    Depends on: build_team_composite_scores().
    """
    if len(team_list) != 3:
        raise ValueError("simulate_alliance requires exactly 3 team numbers.")

    team_data = [team_composite_scores.get(str(t), {}) for t in team_list]
    scores    = [d.get("composite_score", 0) for d in team_data]
    styles    = [d.get("driving_style") or "" for d in team_data]

    projected = sum(scores) / len(scores)
    complement_bonus = _style_complement_bonus([s for s in styles if s])
    projected = min(1.0, projected + complement_bonus * 0.1)

    style_synergy = " + ".join(s for s in styles if s) or "Unknown"

    strengths: list[str] = []
    weaknesses: list[str] = []

    if all(d.get("video_data") for d in team_data):
        strengths.append("All three robots have video analysis data")
    if "DEFENSIVE" in styles and any(s in ("SMOOTH", "DEFENCE_PROOF") for s in styles):
        strengths.append("Classic defender + scorer synergy")
    if "SMOOTH" in styles:
        strengths.append("Consistent scoring robot in alliance")
    if "DEFENCE_PROOF" in styles:
        strengths.append("Hard-to-stop scorer included")

    if styles.count("RECKLESS") >= 2:
        weaknesses.append("Two RECKLESS robots — high collision risk")
    for i, d in enumerate(team_data):
        col = d.get("collision_rate") or 0
        if col > 2.0:
            weaknesses.append(
                f"Team {team_list[i]} has high collision rate ({col:.2f}/min)"
            )
    no_video = [t for t, d in zip(team_list, team_data) if not d.get("video_data")]
    if no_video:
        weaknesses.append(f"No video data for: {', '.join(no_video)}")

    overall_rating = projected + complement_bonus * 0.05

    return {
        "projected_score": round(projected, 4),
        "style_synergy":   style_synergy,
        "strengths":       strengths,
        "weaknesses":      weaknesses,
        "overall_rating":  round(min(1.0, overall_rating), 4),
    }


def flag_risky_teams(
    team_composite_scores: dict[str, dict],
    thresholds: dict | None = None,
) -> list[dict]:
    """
    Flag teams with risk indicators for the do-not-pick analysis.

    Flags teams with:
      - collision_rate > threshold (reckless, may damage alliance partners)
      - low data confidence + mediocre TBA data (unknown risk)
      - RECKLESS style + low CCWM (aggressive but not winning)
      - Significant gap between video score and TBA OPR (inconsistent)

    Args:
        team_composite_scores: Output of build_team_composite_scores().
        thresholds:            Optional override. If None, loads from tba_config.json.

    Returns:
        [{team_number, flags: [str], risk_level: "HIGH" | "MEDIUM"}]

    Depends on: build_team_composite_scores().
    """
    cfg = _load_tba_config()
    if thresholds is None:
        thresholds = cfg["do_not_pick_thresholds"]

    max_col = thresholds.get("max_collision_rate_per_min", 2.5)
    min_comp = thresholds.get("min_composite_score", 0.40)

    risky: list[dict] = []

    for team, data in team_composite_scores.items():
        flags: list[str] = []
        risk_level = "MEDIUM"

        col_rate = data.get("collision_rate") or 0
        if col_rate > max_col:
            flags.append(f"Collision rate {col_rate:.2f}/min > threshold {max_col}")
            risk_level = "HIGH"

        if data.get("data_confidence") == "LOW" and (data.get("tba_opr") or 0) < 15:
            flags.append("Low data confidence + low TBA OPR (unknown risk)")

        if (data.get("driving_style") == "RECKLESS"
                and (data.get("tba_ccwm") or 0) < 0):
            flags.append("RECKLESS style + negative CCWM (aggressive but not winning)")
            risk_level = "HIGH"

        # OPR vs video score gap
        opr = data.get("tba_opr") or 0
        vsr = data.get("video_score_rate")
        if vsr is not None and opr > 0:
            opr_normalised = opr / 60.0
            gap = abs(opr_normalised - vsr)
            if gap > 0.3:
                flags.append(
                    f"Large OPR vs video score gap ({gap:.2f}) — inconsistent performance"
                )

        if data.get("composite_score", 1) < min_comp:
            flags.append(f"Composite score below minimum ({data['composite_score']:.3f})")

        if flags:
            risky.append({
                "team_number": team,
                "flags":       flags,
                "risk_level":  risk_level,
            })

    return sorted(risky, key=lambda x: x["risk_level"] == "HIGH", reverse=True)
