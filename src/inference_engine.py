"""
src/inference_engine.py - Phase 9: Attribution Engine

Core intelligence: decides which robot gets credit for each scored ball.

Functions:
  attribute_score()        - walk ATTRIBUTION_PRIORITY, return best result
  build_score_timeline()   - list of all scoring events with attribution
  compute_final_scores()   - per-robot totals with confidence breakdown

Handles 5 explicit cases:
  Case 1: Robot visible at goal, trajectory + possession agree  (0.95+)
  Case 2: Robot shot then moved away before entry               (0.85+)
  Case 3: Ball from off-screen, last_possessor only             (0.60-0.75)
  Case 4: Scoreboard changed, no ball detected                  (0.40-0.60)
  Case 5: Two robots near zone simultaneously                   (AMBIGUOUS)

Depends on: Phase 6 (possession_log), Phase 7 (scoring_events),
            Phase 5 (robot_tracks, ball_tracks).
"""

from __future__ import annotations

import json
from pathlib import Path


# ---- Attribution priority ---------------------------------------------------

ATTRIBUTION_PRIORITY = [
    "trajectory_origin",   # Ball path traced to robot — HIGHEST
    "last_possessor",      # Last confirmed holder — HIGH
    "proximity_to_zone",   # Nearest robot at score time — MEDIUM
    "alliance_only",       # Alliance known, robot unknown — UNATTRIBUTED
]


# ---- Geometry helpers -------------------------------------------------------

def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _zone_centre(zone_bbox):
    return _bbox_centre(zone_bbox)


# ---- attribute_score --------------------------------------------------------

def attribute_score(
    scoring_event:  dict,
    possession_log: dict[int, list[dict]],
    robot_tracks:   dict[int, list[dict]],
    ball_tracks:    dict[int, list[dict]],
    scoring_zones:  dict[str, list[float]],
    identity_map:   dict[int, dict] | None = None,
    _robot_frame_idx: dict[int, dict[int, dict]] | None = None,
) -> dict:
    """
    Attribute a single scoring event to a robot.

    Walks ATTRIBUTION_PRIORITY and returns on the first confident result.

    Args:
        scoring_event:  Dict from trajectory.detect_scoring_event().
        possession_log: From possession.build_possession_log().
        robot_tracks:   {track_id: [{frame_id, bbox, ...}]}
        ball_tracks:    {track_id: [{frame_id, bbox, ...}]}
        scoring_zones:  {"zone_name": [x1,y1,x2,y2]}
        identity_map:   {track_id: {team_number, ...}} (optional)

    Returns:
        {
            team_number: str,
            method:      str,     # which priority level succeeded
            confidence:  float,
            notes:       str,
            case:        int,     # 1-5
            flag:        str,     # "" | "INFERRED-LOW-CONF" | "AMBIGUOUS-MANUAL-REVIEW" | "TEAM-NUMBER-UNCONFIRMED"
        }
    """
    event_frame  = scoring_event["event_frame"]
    ball_tid     = scoring_event["ball_track_id"]
    zone_name    = scoring_event["zone"]
    zone_bbox    = scoring_zones.get(zone_name)
    traj_robot   = scoring_event.get("trajectory_origin_robot")
    last_poss    = scoring_event.get("last_possessor")
    base_conf    = scoring_event.get("confidence", 0.5)

    def _team(track_id):
        """Resolve track_id to team number via identity_map or track entry."""
        if identity_map and track_id in identity_map:
            return identity_map[track_id].get("team_number", f"UNKNOWN_{track_id}")
        # Try to find team_number stored in track entries
        for entry in robot_tracks.get(track_id, []):
            if "team_number" in entry:
                return entry["team_number"]
        return f"UNKNOWN_{track_id}"

    # ---- Determine which alliance owns this zone (G407: robots must be in  ----
    # ---- their own alliance zone to score; red scores red_goal, etc.)      ----
    zone_alliance: str | None = None
    if "red" in zone_name.lower():
        zone_alliance = "red"
    elif "blue" in zone_name.lower():
        zone_alliance = "blue"

    def _robot_alliance(track_id: int) -> str | None:
        """Return 'red', 'blue', or None for a robot track."""
        if identity_map and track_id in identity_map:
            return identity_map[track_id].get("alliance")
        for entry in robot_tracks.get(track_id, []):
            al = entry.get("alliance")
            if al in ("red", "blue"):
                return al
        return None

    # ---- Robots visible near the zone at event_frame -------------------------
    robots_near_zone: list[tuple[float, int]] = []   # (dist, track_id)
    if zone_bbox:
        zc = _zone_centre(zone_bbox)
        # Use pre-built frame index for O(1) lookup per robot instead of O(track_len)
        if _robot_frame_idx is not None:
            frame_robots = _robot_frame_idx.get(event_frame, {})
            for rtid, e in frame_robots.items():
                # Alliance filter: only include robots from the correct alliance
                if zone_alliance and _robot_alliance(rtid) not in (zone_alliance, None):
                    continue
                rc = _bbox_centre(e["bbox"])
                robots_near_zone.append((_dist(rc, zc), rtid))
        else:
            for rtid, entries in robot_tracks.items():
                if zone_alliance and _robot_alliance(rtid) not in (zone_alliance, None):
                    continue
                for e in entries:
                    if e["frame_id"] == event_frame:
                        rc = _bbox_centre(e["bbox"])
                        robots_near_zone.append((_dist(rc, zc), rtid))
                        break
        robots_near_zone.sort()

    # ---- CASE 1: Robot visible AT zone, trajectory + possession agree --------
    if robots_near_zone and traj_robot and last_poss:
        nearest_dist, nearest_rtid = robots_near_zone[0]
        nearest_team = _team(nearest_rtid)
        if nearest_team == traj_robot == last_poss and nearest_dist < 150:
            return {
                "team_number": traj_robot,
                "method":      "trajectory_origin",
                "confidence":  min(0.98, base_conf + 0.35),
                "notes":       "Robot visible at goal, trajectory and possession agree.",
                "case":        1,
                "flag":        _unconfirmed_flag(traj_robot),
            }

    # ---- CASE 2: Robot shot then moved away before entry --------------------
    if traj_robot and last_poss and traj_robot == last_poss:
        return {
            "team_number": traj_robot,
            "method":      "trajectory_origin",
            "confidence":  min(0.92, base_conf + 0.25),
            "notes":       "Trajectory origin and last possessor agree; robot may have moved.",
            "case":        2,
            "flag":        _unconfirmed_flag(traj_robot),
        }

    # ---- CASE 3: Ball from off-screen, last_possessor only ------------------
    if last_poss:
        return {
            "team_number": last_poss,
            "method":      "last_possessor",
            "confidence":  min(0.72, base_conf + 0.10),
            "notes":       "Robot not visible at goal; attributed to last confirmed possessor.",
            "case":        3,
            "flag":        _unconfirmed_flag(last_poss),
        }

    # ---- CASE 4: Scoreboard changed but no ball detected --------------------
    if robots_near_zone:
        # Check for tie (two or more robots equally close)
        if len(robots_near_zone) >= 2:
            d0, t0 = robots_near_zone[0]
            d1, t1 = robots_near_zone[1]
            if abs(d1 - d0) < 30:        # within 30px — ambiguous
                return _ambiguous(t0, t1, event_frame, _team)

        _, nearest_rtid = robots_near_zone[0]
        team = _team(nearest_rtid)
        return {
            "team_number": team,
            "method":      "proximity_to_zone",
            "confidence":  min(0.55, base_conf),
            "notes":       "No ball detection at goal; attributed by nearest robot proximity.",
            "case":        4,
            "flag":        "INFERRED-LOW-CONF",
        }

    # ---- CASE 5 / Alliance only --------------------------------------------
    return {
        "team_number": "UNATTRIBUTED",
        "method":      "alliance_only",
        "confidence":  0.0,
        "notes":       "No robot track near zone at event time; point unattributed.",
        "case":        5,
        "flag":        "UNATTRIBUTED",
    }


def _unconfirmed_flag(team_number: str) -> str:
    if team_number.startswith("UNKNOWN"):
        return "TEAM-NUMBER-UNCONFIRMED"
    return ""


def _ambiguous(tid_a: int, tid_b: int, frame_id: int, team_fn) -> dict:
    return {
        "team_number": f"{team_fn(tid_a)} / {team_fn(tid_b)}",
        "method":      "proximity_to_zone",
        "confidence":  0.35,
        "notes":       (f"Two robots equally near zone at frame {frame_id}. "
                        "Manual review required."),
        "case":        5,
        "flag":        "AMBIGUOUS-MANUAL-REVIEW",
    }


# ---- reconcile_score_history ------------------------------------------------

def reconcile_score_history(
    tracked_timeline:   list[dict],
    score_history:      list[dict],
    ball_exits:         list[dict],
    robot_identity_map: dict[int, dict],
    opr_map:            dict[str, float],
    source_fps:         float = 30.0,
    lookback_frames:    int   = 90,
) -> list[dict]:
    """
    Cross-reference scoreboard score changes against ball exit events to
    validate and correct the tracked_timeline.

    For each scoreboard delta (score increase of N points between two readings):
      - Find ball exits by the correct alliance in the window
        [reading_start_frame - lookback_frames, reading_end_frame]
      - Simple case (one robot, exits ≈ delta):
          Boost confidence of matching timeline entries.
      - Dense case (multiple robots or exits ≠ delta):
          Distribute delta using weight_i = exits_i × OPR_i,
          creating new OPR-weighted timeline events.
          Low-confidence proximity events in the same window are marked
          SUPERSEDED so they are excluded from final totals.

    Args:
        tracked_timeline:   Existing timeline from trajectory attribution.
        score_history:      [{frame_id, red_score, blue_score}] from scoreboard OCR.
        ball_exits:         From possession.detect_ball_exits().
        robot_identity_map: {track_id: {team_number, alliance?, ...}}
        opr_map:            {team_number: epa_float} from statbotics_client.
        source_fps:         Frames per second.
        lookback_frames:    How far before a score reading to search for exits.

    Returns:
        Updated timeline sorted by frame_id.
    """
    if not score_history or not ball_exits:
        return tracked_timeline

    # Build team → alliance lookup from identity map
    _alliance_cache: dict[str, str] = {}

    def _alliance(team_number: str) -> str:
        if team_number in _alliance_cache:
            return _alliance_cache[team_number]
        for tid, info in robot_identity_map.items():
            if str(info.get("team_number", "")) == team_number:
                al = info.get("alliance", "unknown")
                _alliance_cache[team_number] = al
                return al
        _alliance_cache[team_number] = "unknown"
        return "unknown"

    # Enrich ball_exits with alliance (avoid repeated lookups)
    for ex in ball_exits:
        ex.setdefault("alliance", _alliance(ex["team_number"]))

    # Compute score deltas between consecutive scoreboard readings
    history = sorted(score_history, key=lambda s: s["frame_id"])
    deltas: list[dict] = []
    for i in range(1, len(history)):
        prev, curr = history[i - 1], history[i]
        for alliance in ("red", "blue"):
            ps = prev.get(f"{alliance}_score", -1)
            cs = curr.get(f"{alliance}_score", -1)
            if ps < 0 or cs < 0:
                continue
            delta = cs - ps
            if delta > 0:
                deltas.append({
                    "alliance":    alliance,
                    "delta":       delta,
                    "start_frame": prev["frame_id"],
                    "end_frame":   curr["frame_id"],
                })

    if not deltas:
        return tracked_timeline

    # Work on a copy so we can mutate confidence/notes in place
    timeline = [dict(t) for t in tracked_timeline]
    new_events: list[dict] = []
    n_dense = 0

    for d in deltas:
        alliance    = d["alliance"]
        delta       = d["delta"]
        start_frame = d["start_frame"]
        end_frame   = d["end_frame"]
        search_lo   = max(0, start_frame - lookback_frames)

        # Exits by this alliance in the search window
        window_exits = [
            ex for ex in ball_exits
            if search_lo <= ex["exit_frame"] <= end_frame
            and ex["alliance"] == alliance
        ]

        if not window_exits:
            continue

        exits_per_team: dict[str, int] = {}
        for ex in window_exits:
            t = ex["team_number"]
            exits_per_team[t] = exits_per_team.get(t, 0) + 1

        total_exits = sum(exits_per_team.values())
        single_team = len(exits_per_team) == 1
        counts_match = total_exits == delta

        # Existing timeline events in this scoring window for this alliance
        existing = [
            t for t in timeline
            if start_frame <= t["frame_id"] <= end_frame
            and alliance in t.get("zone", "").lower()
            and t.get("team_number") not in ("UNATTRIBUTED", "REPLACED")
        ]

        if single_team and counts_match:
            # Simple case: one robot, exact exit count match
            team = next(iter(exits_per_team))
            for t in existing:
                if t["team_number"] == team:
                    t["confidence"] = min(0.97, t["confidence"] + 0.15)
                    t["notes"] += " [exit-count confirmed]"
                    t["method"] = "exit_count_confirmed"
            continue

        # Dense case: OPR-weighted distribution
        n_dense += 1
        weights: dict[str, float] = {
            team: max(count * max(opr_map.get(team, 1.0), 0.01), 0.01)
            for team, count in exits_per_team.items()
        }
        total_weight = sum(weights.values())

        sorted_teams = sorted(weights.items(), key=lambda x: -x[1])
        remaining    = delta

        for i, (team, w) in enumerate(sorted_teams):
            pts = remaining if i == len(sorted_teams) - 1 else max(0, round(delta * w / total_weight))
            pts = min(pts, remaining)
            remaining -= pts
            if pts <= 0:
                continue

            team_exits  = [ex for ex in window_exits if ex["team_number"] == team]
            rep_frame   = team_exits[0]["exit_frame"] if team_exits else end_frame
            rep_ball_id = team_exits[0]["ball_track_id"] if team_exits else -1
            opr_val     = opr_map.get(team, 1.0)
            pct         = w / total_weight

            note = (
                f"Dense-shot OPR-weighted: {exits_per_team[team]} exits "
                f"× EPA {opr_val:.1f} = {pts}/{delta} pts "
                f"({', '.join(f'{t}:{c}ex' for t, c in exits_per_team.items())} in window)"
            )

            for _ in range(pts):
                new_events.append({
                    "frame_id":      rep_frame,
                    "timestamp_s":   round(rep_frame / source_fps, 2),
                    "team_number":   team,
                    "method":        "exit_count_opr_weighted",
                    "confidence":    round(min(0.88, 0.60 + 0.28 * pct), 3),
                    "zone":          f"{alliance}_goal",
                    "notes":         note,
                    "case":          6,
                    "flag":          "OPR-WEIGHTED",
                    "ball_track_id": rep_ball_id,
                })

        # Mark superseded low-confidence proximity events so they don't double-count
        for t in existing:
            if (t["confidence"] < 0.60
                    and t["method"] in ("proximity_to_zone", "alliance_only")):
                t["flag"]        = "SUPERSEDED-BY-OPR-WEIGHTED"
                t["team_number"] = "REPLACED"

    combined = timeline + new_events
    combined.sort(key=lambda e: e["frame_id"])

    if new_events:
        print(f"  [Attribution] Reconcile: {n_dense} dense-shot windows -> "
              f"{len(new_events)} OPR-weighted events added")

    return combined


# ---- build_score_timeline ---------------------------------------------------

def build_score_timeline(
    all_scoring_events: list[dict],
    possession_log:     dict[int, list[dict]],
    robot_tracks:       dict[int, list[dict]],
    ball_tracks:        dict[int, list[dict]],
    scoring_zones:      dict[str, list[float]],
    source_fps:         float = 30.0,
    identity_map:       dict[int, dict] | None = None,
    score_history:      list[dict] | None = None,
    ball_exits:         list[dict] | None = None,
    opr_map:            dict[str, float] | None = None,
    lookback_frames:    int = 90,
) -> list[dict]:
    """
    Attribute every scoring event and return a full match timeline.

    Runs trajectory-based attribution first, then reconciles against
    scoreboard score changes using ball exit counts and OPR weights
    to correct dense-shot scenarios.

    Args:
        all_scoring_events: From trajectory.detect_all_scoring_events().
        possession_log:     From possession.build_possession_log().
        robot_tracks:       {track_id: [{frame_id, bbox, ...}]}
        ball_tracks:        {track_id: [{frame_id, bbox, ...}]}
        scoring_zones:      {"zone_name": [x1,y1,x2,y2]}
        source_fps:         Video FPS for timestamp conversion.
        identity_map:       {track_id: {team_number, alliance?, ...}}
        score_history:      Scoreboard readings [{frame_id, red_score, blue_score}].
                            If provided, enables exit-count reconciliation.
        ball_exits:         From possession.detect_ball_exits(). Required for
                            reconciliation; computed here if not provided.
        opr_map:            {team_number: epa_float}. Required for dense-shot
                            OPR weighting; neutral weights used if not provided.
        lookback_frames:    Window size (frames) for exit search before score change.

    Returns:
        [{frame_id, timestamp_s, team_number, method, confidence,
          zone, notes, case, flag, ball_track_id}]
        Sorted by frame_id.
    """
    # Pre-build robot frame index once — O(total_robot_entries) upfront,
    # then O(1) per event instead of O(track_length × n_robots)
    _robot_frame_idx: dict[int, dict[int, dict]] = {}
    for rtid, entries in robot_tracks.items():
        for e in entries:
            fid = e["frame_id"]
            if fid not in _robot_frame_idx:
                _robot_frame_idx[fid] = {}
            _robot_frame_idx[fid][rtid] = e

    # Phase 1: trajectory-based attribution
    timeline = []
    for event in sorted(all_scoring_events, key=lambda e: e["event_frame"]):
        attribution = attribute_score(
            event, possession_log, robot_tracks,
            ball_tracks, scoring_zones, identity_map,
            _robot_frame_idx=_robot_frame_idx,
        )
        timeline.append({
            "frame_id":      event["event_frame"],
            "timestamp_s":   round(event["event_frame"] / source_fps, 2),
            "team_number":   attribution["team_number"],
            "method":        attribution["method"],
            "confidence":    attribution["confidence"],
            "zone":          event["zone"],
            "notes":         attribution["notes"],
            "case":          attribution["case"],
            "flag":          attribution["flag"],
            "ball_track_id": event["ball_track_id"],
        })

    flags = [e for e in timeline if e["flag"]]
    print(f"  [Attribution] Trajectory timeline: {len(timeline)} events  |  "
          f"{len(flags)} flagged")

    # Phase 2: exit-count + OPR reconciliation (when scoreboard data available)
    if score_history:
        # Build ball exits if not pre-computed
        _exits = ball_exits
        if _exits is None:
            # Import guard: possession imports trajectory which imports inference_engine
            # Use a direct import that skips the circular chain
            import importlib as _il
            _pos = _il.import_module("possession")
            _exits = _pos.detect_ball_exits(ball_tracks, possession_log)

        # Build neutral OPR map if not provided
        _opr = opr_map or {}
        if identity_map and not _opr:
            all_teams = [
                str(info.get("team_number", ""))
                for info in identity_map.values()
                if info.get("team_number")
            ]
            _opr = {t: 1.0 for t in all_teams}

        timeline = reconcile_score_history(
            timeline, score_history, _exits,
            identity_map or {}, _opr,
            source_fps=source_fps,
            lookback_frames=lookback_frames,
        )

    total_flags = sum(1 for e in timeline if e.get("flag")
                      and e["flag"] not in ("", "OPR-WEIGHTED", "SUPERSEDED-BY-OPR-WEIGHTED"))
    print(f"  [Attribution] Final timeline: {len(timeline)} events  |  "
          f"{total_flags} flagged for review")
    return timeline


# ---- compute_final_scores ---------------------------------------------------

def compute_final_scores(score_timeline: list[dict]) -> dict[str, dict]:
    """
    Aggregate per-robot scores from the timeline.

    Buckets attributions by confidence:
      high:   >= 0.85
      medium: 0.50 – 0.84
      low:    < 0.50  (and any INFERRED-LOW-CONF)

    Returns:
        {
          "team_number": {
              score:        int,
              high_conf:    int,
              med_conf:     int,
              low_conf:     int,
              unattributed: int,
              flags:        [str],
          }
        }
    """
    totals: dict[str, dict] = {}

    for event in score_timeline:
        team = event["team_number"]
        conf = event["confidence"]
        flag = event["flag"]

        # Skip entries superseded by OPR-weighted reconciliation
        if team == "REPLACED" or flag == "SUPERSEDED-BY-OPR-WEIGHTED":
            continue

        if team == "UNATTRIBUTED" or flag == "UNATTRIBUTED":
            totals.setdefault("UNATTRIBUTED", {
                "score": 0, "high_conf": 0, "med_conf": 0,
                "low_conf": 0, "unattributed": 0, "flags": [],
            })
            totals["UNATTRIBUTED"]["unattributed"] += 1
            continue

        # Handle ambiguous (two teams)
        teams = [t.strip() for t in team.split("/")]
        for t in teams:
            totals.setdefault(t, {
                "score": 0, "high_conf": 0, "med_conf": 0,
                "low_conf": 0, "unattributed": 0, "flags": [],
            })
            totals[t]["score"] += 1
            if conf >= 0.85:
                totals[t]["high_conf"] += 1
            elif conf >= 0.50:
                totals[t]["med_conf"] += 1
            else:
                totals[t]["low_conf"] += 1
            if flag:
                totals[t]["flags"].append(flag)

    # Ensure every known robot appears — even if it scored 0 points.
    # Load match_identity.json to get all robot team numbers.
    try:
        _identity_path = Path(__file__).parent.parent / "configs" / "match_identity.json"
        if _identity_path.exists():
            _id_data = json.loads(_identity_path.read_text())
            for _r in _id_data.get("robots", []):
                _tn = _r.get("team_number", "")
                if _tn and _tn not in ("UNATTRIBUTED", "REPLACED") and \
                   not _tn.startswith("UNKNOWN") and _tn not in totals:
                    totals[_tn] = {
                        "score": 0, "high_conf": 0, "med_conf": 0,
                        "low_conf": 0, "unattributed": 0, "flags": [],
                    }
    except Exception:
        pass

    print("  [Attribution] Final scores:")
    for team, s in sorted(totals.items()):
        print(f"    {team}: {s['score']} pts  "
              f"(high={s['high_conf']} med={s['med_conf']} "
              f"low={s['low_conf']} unattr={s['unattributed']})")

    return totals


# ---- Save / load ------------------------------------------------------------

def save_score_timeline(
    timeline: list[dict],
    out_path: str | Path = "data/score_timeline.json",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"  [Attribution] Timeline saved -> {out_path}")
    return out_path


def load_score_timeline(path: str | Path = "data/score_timeline.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)
