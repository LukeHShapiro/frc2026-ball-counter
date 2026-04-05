"""
src/count.py - Phase 10: Score Aggregation

Functions:
  aggregate_scores()          - group timeline by team, split by confidence
  generate_accuracy_report()  - attribution rate, unattributed points, mismatch flag

Depends on: Phase 9 (score_timeline), Phase 8 (scoreboard_validation).
"""

from __future__ import annotations

import json
from pathlib import Path


def aggregate_scores(
    score_timeline:    list[dict],
    robot_identity_map: dict[int, dict] | None = None,
) -> dict[str, dict]:
    """
    Group score timeline by team number and split attributions by confidence tier.

    Args:
        score_timeline:      From inference_engine.build_score_timeline().
        robot_identity_map:  {track_id: {team_number, ...}} (optional,
                              used to resolve UNKNOWN_ entries if possible).

    Returns:
        {
          team_number: {
              score:        int,
              high_conf:    int,    # conf >= 0.85
              med_conf:     int,    # conf 0.50-0.84
              low_conf:     int,    # conf < 0.50
              unattributed: int,
              events:       [dict], # full event list for this team
          }
        }
    """
    # Build reverse lookup: team_number -> resolved name (in case of UNKNOWN_n)
    resolved: dict[str, str] = {}
    if robot_identity_map:
        for tid, info in robot_identity_map.items():
            unk_key = f"UNKNOWN_{tid}"
            tnum    = info.get("team_number", unk_key)
            if not tnum.startswith("UNKNOWN"):
                resolved[unk_key] = tnum

    totals: dict[str, dict] = {}

    for event in score_timeline:
        raw_team = event.get("team_number", "UNATTRIBUTED")
        team     = resolved.get(raw_team, raw_team)
        conf     = event.get("confidence", 0.0)
        flag     = event.get("flag", "")

        if team == "UNATTRIBUTED" or flag == "UNATTRIBUTED":
            totals.setdefault("UNATTRIBUTED", _blank())
            totals["UNATTRIBUTED"]["unattributed"] += 1
            totals["UNATTRIBUTED"]["events"].append(event)
            continue

        # Ambiguous events (team = "A / B") counted for both
        teams = [t.strip() for t in team.split("/")]
        for t in teams:
            totals.setdefault(t, _blank())
            totals[t]["score"] += 1
            if conf >= 0.85:
                totals[t]["high_conf"] += 1
            elif conf >= 0.50:
                totals[t]["med_conf"] += 1
            else:
                totals[t]["low_conf"] += 1
            totals[t]["events"].append(event)

    return totals


def _blank() -> dict:
    return {
        "score": 0, "high_conf": 0, "med_conf": 0,
        "low_conf": 0, "unattributed": 0, "events": [],
    }


def generate_accuracy_report(
    final_scores:         dict[str, dict],
    scoreboard_validation: dict | None = None,
) -> dict:
    """
    Compute attribution accuracy metrics and flag any scoreboard mismatches.

    Args:
        final_scores:          From aggregate_scores().
        scoreboard_validation: From scoreboard.validate_attribution() (optional).

    Returns:
        {
            total_attributed:    int,
            total_unattributed:  int,
            attribution_rate_pct: float,
            high_conf_pct:       float,
            med_conf_pct:        float,
            low_conf_pct:        float,
            discrepancy:         bool,
            discrepancy_details: str,
            per_team:            {team: {score, high_conf, med_conf, low_conf}},
        }
    """
    total_attr   = sum(v["score"]        for k, v in final_scores.items() if k != "UNATTRIBUTED")
    total_unattr = final_scores.get("UNATTRIBUTED", {}).get("unattributed", 0)
    total_all    = total_attr + total_unattr

    high_total = sum(v["high_conf"] for k, v in final_scores.items() if k != "UNATTRIBUTED")
    med_total  = sum(v["med_conf"]  for k, v in final_scores.items() if k != "UNATTRIBUTED")
    low_total  = sum(v["low_conf"]  for k, v in final_scores.items() if k != "UNATTRIBUTED")

    attr_rate  = (total_attr / total_all * 100)  if total_all > 0 else 0.0
    high_pct   = (high_total / total_attr * 100) if total_attr > 0 else 0.0
    med_pct    = (med_total  / total_attr * 100) if total_attr > 0 else 0.0
    low_pct    = (low_total  / total_attr * 100) if total_attr > 0 else 0.0

    discrepancy      = False
    discrepancy_text = ""

    if scoreboard_validation:
        if not scoreboard_validation.get("balanced", True):
            discrepancy = True
            red_gap     = scoreboard_validation.get("red_gap",  0)
            blue_gap    = scoreboard_validation.get("blue_gap", 0)
            parts = []
            if red_gap != 0:
                parts.append(f"Red: {red_gap} pts unaccounted")
            if blue_gap != 0:
                parts.append(f"Blue: {blue_gap} pts unaccounted")
            discrepancy_text = "; ".join(parts)

    per_team = {
        k: {"score": v["score"], "high_conf": v["high_conf"],
            "med_conf": v["med_conf"], "low_conf": v["low_conf"]}
        for k, v in final_scores.items()
        if k != "UNATTRIBUTED"
    }

    report = {
        "total_attributed":    total_attr,
        "total_unattributed":  total_unattr,
        "attribution_rate_pct": round(attr_rate, 1),
        "high_conf_pct":       round(high_pct, 1),
        "med_conf_pct":        round(med_pct,  1),
        "low_conf_pct":        round(low_pct,  1),
        "discrepancy":         discrepancy,
        "discrepancy_details": discrepancy_text,
        "per_team":            per_team,
    }

    # Print report
    print("\n" + "=" * 50)
    print("  SCORE ATTRIBUTION REPORT")
    print("=" * 50)
    print(f"  Total scored    : {total_all}")
    print(f"  Attributed      : {total_attr}  ({attr_rate:.1f}%)")
    print(f"  Unattributed    : {total_unattr}")
    print(f"  Confidence mix  : high={high_pct:.0f}%  "
          f"med={med_pct:.0f}%  low={low_pct:.0f}%")

    if discrepancy:
        print(f"\n  (!) DISCREPANCY: {discrepancy_text}")
    else:
        print("\n  OK Attribution totals match scoreboard.")

    print("\n  Per-robot scores:")
    for team, s in sorted(per_team.items()):
        flags = final_scores[team].get("events", [])
        flag_list = [e.get("flag","") for e in flags if e.get("flag","")]
        flag_str  = f"  [{', '.join(set(flag_list))}]" if flag_list else ""
        print(f"    {team}: {s['score']} pts  "
              f"(H={s['high_conf']} M={s['med_conf']} L={s['low_conf']}){flag_str}")
    print("=" * 50)

    return report


def save_accuracy_report(
    report:   dict,
    out_path: str | Path = "data/accuracy_report.json",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove non-serialisable event lists before saving
    safe = {k: v for k, v in report.items() if k != "per_team_events"}
    with open(out_path, "w") as f:
        json.dump(safe, f, indent=2)
    print(f"  [Count] Accuracy report saved -> {out_path}")
    return out_path
