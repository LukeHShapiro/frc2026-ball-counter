"""
src/driving_analysis.py — Phase 13: Driving Style Analysis

Analyzes robot movement patterns from robot_tracks (Phase 5) and classifies
each robot's driving style as DEFENSIVE, RECKLESS, SMOOTH, or DEFENCE_PROOF.

All classification thresholds are loaded from configs/field_config.json.
No values are hardcoded.

Depends on:
  - robot_tracks from Phase 5 (track.py)
  - score_timeline from Phase 9 (inference_engine.py)
  - configs/field_config.json  (driving_classification thresholds, alliance_zones)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

# ── Config loading ─────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "field_config.json"


def _load_config() -> dict:
    """Load field_config.json. Raises FileNotFoundError if missing."""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"field_config.json not found at {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "r") as f:
        return json.load(f)


# ── Type aliases (lightweight dicts used throughout) ──────────────────────────
# robot_tracks: {track_id: [{frame_id, bbox: [x1,y1,x2,y2], team_number}]}
# all_robot_tracks: same dict, all robots in the match


# ── Velocity cache (avoids recomputing per-robot velocities N² times) ─────────
# Populated by classify_all_robots before the per-robot loop.
_vel_cache: dict[int, list[dict]] = {}


# ── Helper utilities ──────────────────────────────────────────────────────────

def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    """Return (cx, cy) for a [x1, y1, x2, y2] bounding box."""
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _bbox_distance(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Euclidean distance between the centres of two bounding boxes."""
    ax, ay = _bbox_center(bbox_a)
    bx, by = _bbox_center(bbox_b)
    return math.hypot(ax - bx, ay - by)


def _sorted_track(robot_track: list[dict]) -> list[dict]:
    """Return track entries sorted by frame_id ascending."""
    return sorted(robot_track, key=lambda r: r["frame_id"])


# ── Core metric functions ──────────────────────────────────────────────────────

def compute_robot_velocity(
    robot_track_id: int,
    robot_tracks: dict[int, list[dict]],
    frame_window: int = 3,
) -> list[dict]:
    """
    Compute per-frame velocity and heading for a single robot.

    Args:
        robot_track_id: Track ID of the robot to analyse.
        robot_tracks:   Dict mapping track_id → list of frame dicts
                        [{frame_id, bbox, team_number}].
        frame_window:   Number of frames to look back for smoothing.

    Returns:
        List of dicts: [{frame_id, vx, vy, speed, heading_deg}]

    Depends on: Phase 5 robot_tracks.
    """
    if robot_track_id in _vel_cache:
        return _vel_cache[robot_track_id]

    track = _sorted_track(robot_tracks.get(robot_track_id, []))
    velocities: list[dict] = []

    for i in range(len(track)):
        start_idx = max(0, i - frame_window)
        if start_idx == i:
            velocities.append({
                "frame_id": track[i]["frame_id"],
                "vx": 0.0, "vy": 0.0,
                "speed": 0.0, "heading_deg": 0.0,
            })
            continue

        cx_now, cy_now = _bbox_center(track[i]["bbox"])
        cx_prev, cy_prev = _bbox_center(track[start_idx]["bbox"])
        frames_elapsed = track[i]["frame_id"] - track[start_idx]["frame_id"]

        if frames_elapsed == 0:
            vx = vy = 0.0
        else:
            vx = (cx_now - cx_prev) / frames_elapsed
            vy = (cy_now - cy_prev) / frames_elapsed

        speed = math.hypot(vx, vy)
        heading_deg = math.degrees(math.atan2(vy, vx)) if speed > 0 else 0.0

        velocities.append({
            "frame_id": track[i]["frame_id"],
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "heading_deg": heading_deg,
        })

    _vel_cache[robot_track_id] = velocities
    return velocities


def detect_collisions(
    robot_track_id: int,
    robot_tracks: dict[int, list[dict]],
    all_robot_tracks: dict[int, list[dict]],
    velocity_drop_threshold: float = 0.4,
    proximity_threshold_px: float = 100.0,
) -> list[dict]:
    """
    Detect collision events for a robot.

    A collision = sudden speed drop >= velocity_drop_threshold (40% by default)
    while within proximity_threshold_px of another robot.

    Args:
        robot_track_id:          Track ID to analyse.
        robot_tracks:            Full track dict (same robot).
        all_robot_tracks:        Track dict for all robots (used for proximity).
        velocity_drop_threshold: Fractional speed drop that signals a collision.
        proximity_threshold_px:  Max distance to opponent to count as collision.

    Returns:
        List of dicts: [{frame_id, opponent_track_id, pre_velocity, post_velocity}]

    Depends on: Phase 5 robot_tracks.
    """
    velocities = compute_robot_velocity(robot_track_id, robot_tracks)
    track = _sorted_track(robot_tracks.get(robot_track_id, []))

    # Build frame_id → bbox lookup for this robot
    bbox_by_frame: dict[int, list[float]] = {
        entry["frame_id"]: entry["bbox"] for entry in track
    }

    # Build frame_id → bbox lookup per opponent track
    opp_bboxes: dict[int, dict[int, list[float]]] = {}
    for tid, otrack in all_robot_tracks.items():
        if tid == robot_track_id:
            continue
        opp_bboxes[tid] = {e["frame_id"]: e["bbox"] for e in otrack}

    collisions: list[dict] = []

    for i in range(1, len(velocities)):
        pre_speed = velocities[i - 1]["speed"]
        post_speed = velocities[i]["speed"]

        if pre_speed < 0.5:
            continue  # already nearly stopped — not meaningful

        if pre_speed == 0:
            continue

        drop_ratio = (pre_speed - post_speed) / pre_speed
        if drop_ratio < velocity_drop_threshold:
            continue

        frame_id = velocities[i]["frame_id"]
        my_bbox = bbox_by_frame.get(frame_id)
        if my_bbox is None:
            continue

        # Check proximity to any opponent at this frame
        for opp_tid, opp_frame_bboxes in opp_bboxes.items():
            opp_bbox = opp_frame_bboxes.get(frame_id)
            if opp_bbox is None:
                continue
            dist = _bbox_distance(my_bbox, opp_bbox)
            if dist <= proximity_threshold_px:
                collisions.append({
                    "frame_id": frame_id,
                    "opponent_track_id": opp_tid,
                    "pre_velocity": pre_speed,
                    "post_velocity": post_speed,
                })
                break  # one collision per frame

    return collisions


def compute_path_repetition(
    robot_track_id: int,
    robot_tracks: dict[int, list[dict]],
    grid_resolution: int = 50,
) -> float:
    """
    Measure how repetitively a robot travels the same route.

    Divides the field into grid cells of grid_resolution x grid_resolution pixels
    and records the sequence of cells visited. Repetition score = fraction of
    consecutive cell-pair transitions that appeared in a previous pass.

    Args:
        robot_track_id:   Track ID to analyse.
        robot_tracks:     Full track dict.
        grid_resolution:  Cell size in pixels.

    Returns:
        Float 0.0 (totally varied paths) to 1.0 (always same path).

    Depends on: Phase 5 robot_tracks.
    """
    track = _sorted_track(robot_tracks.get(robot_track_id, []))
    if len(track) < 2:
        return 0.0

    def to_cell(bbox: list[float]) -> tuple[int, int]:
        cx, cy = _bbox_center(bbox)
        return int(cx // grid_resolution), int(cy // grid_resolution)

    cells = [to_cell(e["bbox"]) for e in track]

    transitions = list(zip(cells[:-1], cells[1:]))
    if not transitions:
        return 0.0

    seen: set[tuple] = set()
    repeated = 0
    for t in transitions:
        if t in seen:
            repeated += 1
        seen.add(t)

    return repeated / len(transitions)


def detect_shadowing_events(
    robot_track_id: int,
    robot_tracks: dict[int, list[dict]],
    all_robot_tracks: dict[int, list[dict]],
    follow_duration_frames: int = 20,
    proximity_threshold_px: float = 120.0,
) -> list[dict]:
    """
    Detect when this robot mirrors an opponent's movement for N+ consecutive frames.

    Used both for classifying DEFENSIVE robots and for measuring pressure on others.

    Args:
        robot_track_id:         Track ID of the potential defender.
        robot_tracks:           Full track dict.
        all_robot_tracks:       Track dict for all robots.
        follow_duration_frames: Min consecutive frames required to count as shadowing.
        proximity_threshold_px: Max distance to target to count as shadowing.

    Returns:
        List of dicts: [{start_frame, end_frame, target_robot_track_id, duration_frames}]

    Depends on: Phase 5 robot_tracks.
    """
    my_vels = compute_robot_velocity(robot_track_id, robot_tracks)
    my_vel_by_frame: dict[int, dict] = {v["frame_id"]: v for v in my_vels}

    my_track = _sorted_track(robot_tracks.get(robot_track_id, []))
    my_bbox_by_frame: dict[int, list[float]] = {
        e["frame_id"]: e["bbox"] for e in my_track
    }

    shadowing_events: list[dict] = []

    for opp_tid, opp_track_list in all_robot_tracks.items():
        if opp_tid == robot_track_id:
            continue

        opp_vels = compute_robot_velocity(opp_tid, all_robot_tracks)
        opp_vel_by_frame: dict[int, dict] = {v["frame_id"]: v for v in opp_vels}
        opp_bbox_by_frame: dict[int, list[float]] = {
            e["frame_id"]: e["bbox"] for e in opp_track_list
        }

        # Find all frames where both robots are present and close
        shared_frames = sorted(
            set(my_bbox_by_frame.keys()) & set(opp_bbox_by_frame.keys())
        )

        streak_start: int | None = None
        streak_last: int | None = None

        def _flush(start: int, end: int) -> None:
            dur = end - start + 1
            if dur >= follow_duration_frames:
                shadowing_events.append({
                    "start_frame": start,
                    "end_frame": end,
                    "target_robot_track_id": opp_tid,
                    "duration_frames": dur,
                })

        for frame_id in shared_frames:
            my_bbox = my_bbox_by_frame[frame_id]
            opp_bbox = opp_bbox_by_frame[frame_id]
            dist = _bbox_distance(my_bbox, opp_bbox)

            if dist > proximity_threshold_px:
                if streak_start is not None:
                    _flush(streak_start, streak_last)
                    streak_start = streak_last = None
                continue

            # Check heading alignment (mirroring = similar heading direction)
            my_v = my_vel_by_frame.get(frame_id)
            opp_v = opp_vel_by_frame.get(frame_id)
            mirroring = True
            if my_v and opp_v and my_v["speed"] > 0.3 and opp_v["speed"] > 0.3:
                angle_diff = abs(my_v["heading_deg"] - opp_v["heading_deg"]) % 360
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                # Shadowing = moving in same or close direction while nearby
                mirroring = angle_diff < 60

            if not mirroring:
                if streak_start is not None:
                    _flush(streak_start, streak_last)
                    streak_start = streak_last = None
                continue

            if streak_start is None:
                streak_start = frame_id
            streak_last = frame_id

        if streak_start is not None:
            _flush(streak_start, streak_last)

    return shadowing_events


def compute_scoring_under_pressure(
    robot_track_id: int,
    score_timeline: list[dict],
    shadowing_events: list[dict],
    proximity_threshold_px: float = 120.0,
) -> dict:
    """
    Count scores this robot made while being shadowed by a defender.

    Cross-references score_timeline (Phase 9) with shadowing events where
    this robot was the TARGET (i.e., being defended).

    Args:
        robot_track_id:         Track ID of the robot being evaluated.
        score_timeline:         From inference_engine.build_score_timeline().
        shadowing_events:       From detect_shadowing_events() called on opponent robots,
                                where target_robot_track_id == robot_track_id.
        proximity_threshold_px: Proximity threshold used (informational only here).

    Returns:
        {scores_under_pressure: int, scores_total: int, pressure_rate: float}

    Depends on: Phase 9 score_timeline, Phase 5 robot_tracks.
    """
    # Build set of frames where this robot is under pressure
    pressure_frames: set[int] = set()
    for event in shadowing_events:
        pressure_frames.update(
            range(event["start_frame"], event["end_frame"] + 1)
        )

    scores_total = 0
    scores_under_pressure = 0

    for event in score_timeline:
        # Match by track_id if available, else skip
        if event.get("robot_track_id") != robot_track_id:
            # Also try team_number match if track_id not present
            continue
        scores_total += 1
        if event.get("frame_id") in pressure_frames:
            scores_under_pressure += 1

    pressure_rate = (
        scores_under_pressure / scores_total if scores_total > 0 else 0.0
    )

    return {
        "scores_under_pressure": scores_under_pressure,
        "scores_total": scores_total,
        "pressure_rate": pressure_rate,
    }


def compute_all_metrics(
    robot_track_id: int,
    robot_tracks: dict[int, list[dict]],
    all_robot_tracks: dict[int, list[dict]],
    score_timeline: list[dict],
    alliance_zones: dict,
    _precomputed_shadowing: dict[int, list[dict]] | None = None,
) -> dict:
    """
    Compute the full DRIVING_METRICS dict for a single robot.

    Calls all individual metric functions and assembles results.

    Args:
        robot_track_id:    Track ID to analyse.
        robot_tracks:      Full track dict (single robot focus, but same dict).
        all_robot_tracks:  All robots' tracks for cross-robot metrics.
        score_timeline:    From Phase 9.
        alliance_zones:    From field_config.json["alliance_zones"].

    Returns:
        DRIVING_METRICS dict as specified in Phase 13.

    Depends on: Phase 5 robot_tracks, Phase 9 score_timeline.
    """
    config = _load_config()
    field_w = config.get("field", {}).get("width_px", 1280)

    velocities = compute_robot_velocity(robot_track_id, robot_tracks)
    speeds = [v["speed"] for v in velocities]
    headings = [v["heading_deg"] for v in velocities]

    avg_velocity = sum(speeds) / len(speeds) if speeds else 0.0
    max_velocity = max(speeds) if speeds else 0.0

    # Velocity variance
    if len(speeds) > 1:
        mean_s = avg_velocity
        vel_variance = sum((s - mean_s) ** 2 for s in speeds) / len(speeds)
    else:
        vel_variance = 0.0

    # Angular velocity variance (differences between consecutive headings)
    heading_diffs = []
    for i in range(1, len(headings)):
        diff = abs(headings[i] - headings[i - 1]) % 360
        if diff > 180:
            diff = 360 - diff
        heading_diffs.append(diff)
    angular_vel_variance = (
        sum((d - (sum(heading_diffs) / len(heading_diffs))) ** 2
            for d in heading_diffs) / len(heading_diffs)
        if heading_diffs else 0.0
    )

    # Arc smoothness: inverse of mean heading change
    arc_smoothness = (
        1.0 / (1.0 + sum(heading_diffs) / len(heading_diffs))
        if heading_diffs else 1.0
    )

    # Collisions
    collisions = detect_collisions(robot_track_id, robot_tracks, all_robot_tracks)
    collision_count = len(collisions)
    track = _sorted_track(robot_tracks.get(robot_track_id, []))
    match_duration_frames = (
        (track[-1]["frame_id"] - track[0]["frame_id"]) if len(track) > 1 else 1
    )
    fps_assumed = 10.0
    match_duration_min = (match_duration_frames / fps_assumed) / 60.0
    collision_rate = (
        collision_count / match_duration_min if match_duration_min > 0 else 0.0
    )

    # Velocity recovery rate after collisions
    if collisions and velocities:
        vel_by_frame = {v["frame_id"]: v["speed"] for v in velocities}
        recoveries = []
        for col in collisions:
            post_frame = col["frame_id"]
            frames_after = [
                v for v in velocities if v["frame_id"] > post_frame
            ][:10]
            if frames_after:
                peak_recovery = max(v["speed"] for v in frames_after)
                frames_to_recover = len(frames_after)
                if frames_to_recover > 0:
                    recoveries.append(
                        (peak_recovery - col["post_velocity"]) / frames_to_recover
                    )
        vel_recovery_rate = sum(recoveries) / len(recoveries) if recoveries else 0.0
    else:
        vel_recovery_rate = 0.0

    # Path repetition
    path_rep = compute_path_repetition(robot_track_id, robot_tracks)
    path_variety = 1.0 - path_rep

    # Time in opponent half — determine which alliance this robot is on
    # by checking where it spends most time vs field center
    blue_zone = alliance_zones.get("blue", [640, 0, 1280, 720])
    blue_x1 = blue_zone[0]

    frames_in_opp_half = 0
    for entry in track:
        cx, _ = _bbox_center(entry["bbox"])
        # We assume red alliance robots going into blue zone = opponent half
        if cx >= blue_x1:
            frames_in_opp_half += 1
    time_opp_half_pct = (
        frames_in_opp_half / len(track) if track else 0.0
    )

    # Average distance to nearest opponent
    opp_tracks_list = {
        tid: t for tid, t in all_robot_tracks.items() if tid != robot_track_id
    }
    bbox_by_frame = {e["frame_id"]: e["bbox"] for e in track}
    opp_bboxes_by_frame: dict[int, list[list[float]]] = {}
    for opp_t in opp_tracks_list.values():
        for e in opp_t:
            opp_bboxes_by_frame.setdefault(e["frame_id"], []).append(e["bbox"])

    distances: list[float] = []
    for frame_id, my_bbox in bbox_by_frame.items():
        opp_list = opp_bboxes_by_frame.get(frame_id, [])
        if opp_list:
            min_dist = min(_bbox_distance(my_bbox, ob) for ob in opp_list)
            distances.append(min_dist)
    avg_dist_opponent = sum(distances) / len(distances) if distances else float("inf")

    # Shadowing events — use pre-computed data if available (avoids O(N³) recompute)
    if _precomputed_shadowing is not None:
        shadowing_as_defender = _precomputed_shadowing.get(robot_track_id, [])
        shadowing_on_me = [
            evt
            for opp_events in _precomputed_shadowing.values()
            for evt in opp_events
            if evt["target_robot_track_id"] == robot_track_id
        ]
    else:
        # Fallback: compute on demand (slow for large matches)
        shadowing_as_defender = detect_shadowing_events(
            robot_track_id, robot_tracks, all_robot_tracks
        )
        shadowing_on_me = []
        for opp_tid in opp_tracks_list:
            for evt in detect_shadowing_events(opp_tid, all_robot_tracks, all_robot_tracks):
                if evt["target_robot_track_id"] == robot_track_id:
                    shadowing_on_me.append(evt)

    # Scoring under pressure
    pressure_result = compute_scoring_under_pressure(
        robot_track_id, score_timeline, shadowing_on_me
    )

    # Escape success rate: fraction of shadowing-on-me events that ended within 3s (30 frames)
    escape_successes = 0
    for evt in shadowing_on_me:
        duration = evt["duration_frames"]
        if duration <= 30:  # escaped within ~3 seconds at 10fps
            escape_successes += 1
    escape_rate = (
        escape_successes / len(shadowing_on_me) if shadowing_on_me else 0.0
    )

    return {
        "avg_velocity_px_per_frame":        avg_velocity,
        "velocity_variance":                vel_variance,
        "max_velocity":                     max_velocity,
        "velocity_recovery_rate":           vel_recovery_rate,
        "collision_count":                  collision_count,
        "collision_rate_per_minute":        collision_rate,
        "angular_velocity_variance":        angular_vel_variance,
        "path_repetition_score":            path_rep,
        "arc_smoothness":                   arc_smoothness,
        "time_in_opponent_half_pct":        time_opp_half_pct,
        "shadowing_events":                 len(shadowing_as_defender),
        "avg_distance_to_nearest_opponent": avg_dist_opponent,
        "scoring_under_pressure_rate":      pressure_result["pressure_rate"],
        "escape_success_rate":              escape_rate,
        "path_variety_score":               path_variety,
    }


def classify_driving_style(metrics: dict) -> dict:
    """
    Classify a robot's driving style from its DRIVING_METRICS dict.

    All thresholds are loaded from configs/field_config.json.

    Args:
        metrics: Full DRIVING_METRICS dict from compute_all_metrics().

    Returns:
        {
            primary_style:  str,   # DEFENSIVE | RECKLESS | SMOOTH | DEFENCE_PROOF
            secondary_style: str | None,
            confidence:     float,
            style_scores:   {DEFENSIVE: float, RECKLESS: float, SMOOTH: float,
                             DEFENCE_PROOF: float},
            key_evidence:   [str]
        }

    Depends on: configs/field_config.json driving_classification thresholds.
    """
    config = _load_config()
    thresholds = config["driving_classification"]

    # ── Score each style ──────────────────────────────────────────────────────

    style_scores: dict[str, float] = {
        "DEFENSIVE": 0.0,
        "RECKLESS": 0.0,
        "SMOOTH": 0.0,
        "DEFENCE_PROOF": 0.0,
    }
    evidence: list[str] = []
    # Pre-initialise all conditional evidence strings to None
    evidence_d     = None
    evidence_s     = None
    evidence_r_col = None

    # DEFENSIVE
    d_thresh = thresholds["defensive"]
    defensive_score = 0.0
    defensive_hits = 0
    if metrics["time_in_opponent_half_pct"] >= d_thresh["min_time_opponent_half_pct"]:
        defensive_score += 0.40
        defensive_hits += 1
        evidence_d = (
            f"Spent {metrics['time_in_opponent_half_pct']:.0%} of match in opponent half"
        )
    if metrics["shadowing_events"] >= d_thresh["min_shadowing_events"]:
        defensive_score += 0.40
        defensive_hits += 1
        evidence_s = f"{metrics['shadowing_events']} shadowing events detected"
    # Low personal scoring is checked via score rate in score_timeline;
    # approximate via scoring_under_pressure_rate being low
    if metrics["avg_velocity_px_per_frame"] < 3.0:
        defensive_score += 0.20
    style_scores["DEFENSIVE"] = min(1.0, defensive_score)

    # RECKLESS
    r_thresh = thresholds["reckless"]
    reckless_score = 0.0
    if metrics["collision_rate_per_minute"] >= r_thresh["min_collision_rate_per_min"]:
        reckless_score += 0.45
        evidence_r_col = (
            f"High collision rate: {metrics['collision_rate_per_minute']:.2f}/min"
        )
    else:
        evidence_r_col = None
    if metrics["angular_velocity_variance"] > 200:
        reckless_score += 0.30
    if metrics["max_velocity"] > 15:
        reckless_score += 0.25
    style_scores["RECKLESS"] = min(1.0, reckless_score)

    # SMOOTH
    s_thresh = thresholds["smooth"]
    smooth_score = 0.0
    if metrics["velocity_variance"] < 5.0:
        smooth_score += 0.35
    if metrics["collision_rate_per_minute"] <= s_thresh["max_collision_rate_per_min"]:
        smooth_score += 0.35
    if metrics["arc_smoothness"] > 0.70:
        smooth_score += 0.30
    style_scores["SMOOTH"] = min(1.0, smooth_score)

    # DEFENCE_PROOF
    dp_thresh = thresholds["defence_proof"]
    dp_score = 0.0
    if metrics["scoring_under_pressure_rate"] >= dp_thresh["min_scoring_under_pressure_rate"]:
        dp_score += 0.40
    if metrics["escape_success_rate"] >= dp_thresh["min_escape_success_rate"]:
        dp_score += 0.35
    if metrics["path_repetition_score"] <= dp_thresh["max_path_repetition_score"]:
        dp_score += 0.25
    style_scores["DEFENCE_PROOF"] = min(1.0, dp_score)

    # ── Assign primary + secondary ────────────────────────────────────────────
    sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
    primary_style = sorted_styles[0][0]
    primary_score = sorted_styles[0][1]
    secondary_style = sorted_styles[1][0] if sorted_styles[1][1] >= 0.35 else None

    # Confidence = how far ahead primary is from the second best
    if len(sorted_styles) >= 2:
        gap = primary_score - sorted_styles[1][1]
        confidence = min(1.0, 0.50 + gap)
    else:
        confidence = primary_score

    # ── Build key evidence bullets ────────────────────────────────────────────
    key_evidence: list[str] = []

    if primary_style == "DEFENSIVE":
        if evidence_d:
            key_evidence.append(evidence_d)
        if evidence_s:
            key_evidence.append(evidence_s)
        _avg_dist = metrics['avg_distance_to_nearest_opponent']
        _avg_dist_str = "N/A" if _avg_dist == float("inf") else f"{_avg_dist:.1f}px"
        key_evidence.append(f"Avg distance to opponent: {_avg_dist_str}")

    elif primary_style == "RECKLESS":
        if evidence_r_col:
            key_evidence.append(evidence_r_col)
        key_evidence.append(
            f"Max velocity: {metrics['max_velocity']:.1f}px/frame"
        )
        key_evidence.append(
            f"Angular variance: {metrics['angular_velocity_variance']:.1f}"
        )

    elif primary_style == "SMOOTH":
        key_evidence.append(
            f"Velocity variance: {metrics['velocity_variance']:.2f} (low = smooth)"
        )
        key_evidence.append(
            f"Arc smoothness: {metrics['arc_smoothness']:.2f}"
        )
        key_evidence.append(
            f"Collision rate: {metrics['collision_rate_per_minute']:.2f}/min"
        )

    elif primary_style == "DEFENCE_PROOF":
        key_evidence.append(
            f"Scored under pressure: {metrics['scoring_under_pressure_rate']:.0%}"
        )
        key_evidence.append(
            f"Escape success rate: {metrics['escape_success_rate']:.0%}"
        )
        key_evidence.append(
            f"Path variety score: {metrics['path_variety_score']:.2f}"
        )

    if not key_evidence:
        key_evidence.append(
            f"Primary style score: {primary_score:.2f}"
        )

    return {
        "primary_style":  primary_style,
        "secondary_style": secondary_style,
        "confidence":     round(confidence, 3),
        "style_scores":   {k: round(v, 3) for k, v in style_scores.items()},
        "key_evidence":   key_evidence[:4],
    }


def classify_all_robots(
    robot_identity_map: dict[int, dict],
    robot_tracks: dict[int, list[dict]],
    all_robot_tracks: dict[int, list[dict]],
    score_timeline: list[dict],
    alliance_zones: dict,
) -> dict[str, dict]:
    """
    Classify driving style for every robot in the match.

    Args:
        robot_identity_map: {track_id: {team_number, confidence, ...}} from Phase 5.
        robot_tracks:       Full track dict (all robots).
        all_robot_tracks:   Same as robot_tracks (passed separately for clarity).
        score_timeline:     From Phase 9.
        alliance_zones:     From field_config.json.

    Returns:
        {team_number: driving_style_result_dict}

    Depends on: Phase 5 robot_tracks, Phase 9 score_timeline.
    """
    results: dict[str, dict] = {}

    # ── Pre-compute velocities for ALL robots once ───────────────────────────
    # Populates _vel_cache so detect_shadowing_events() never recomputes them.
    global _vel_cache
    _vel_cache.clear()
    print("  [Phase 13] Pre-computing robot velocities...")
    for track_id in robot_identity_map:
        compute_robot_velocity(track_id, all_robot_tracks)

    # ── Pre-compute all-pairs shadowing events once ──────────────────────────
    # Without this, classify_all_robots is O(N³) — it hangs for 6 robots.
    # {shadower_track_id: [shadowing_event_dicts]}
    print("  [Phase 13] Pre-computing shadowing events (all pairs)...")
    precomputed_shadowing: dict[int, list[dict]] = {}
    for track_id in robot_identity_map:
        precomputed_shadowing[track_id] = detect_shadowing_events(
            track_id, all_robot_tracks, all_robot_tracks
        )

    for track_id, identity in robot_identity_map.items():
        team_number = identity.get("team_number", f"UNKNOWN_{track_id}")
        print(f"  [Phase 13] Classifying driving style for Team {team_number} (track {track_id})...")

        metrics = compute_all_metrics(
            track_id, robot_tracks, all_robot_tracks,
            score_timeline, alliance_zones,
            _precomputed_shadowing=precomputed_shadowing,
        )
        style = classify_driving_style(metrics)

        results[team_number] = {
            **style,
            "metrics": metrics,
        }

        print(
            f"    -> {style['primary_style']}"
            + (f" / {style['secondary_style']}" if style["secondary_style"] else "")
            + f"  (confidence: {style['confidence']:.2f})"
        )

    return results


def generate_driving_report(
    all_classifications: dict[str, dict],
    all_metrics: dict[str, dict],
) -> dict[str, dict]:
    """
    Assemble the final structured driving report.

    Args:
        all_classifications: Output of classify_all_robots().
        all_metrics:         {team_number: DRIVING_METRICS dict}.

    Returns:
        {
            team_number: {
                style:        str,
                secondary:    str | None,
                confidence:   float,
                style_scores: dict,
                key_evidence: [str],
                metrics:      DRIVING_METRICS dict
            }
        }

    Depends on: Phase 5 robot_tracks, Phase 9 score_timeline.
    """
    report: dict[str, dict] = {}

    for team_number, classification in all_classifications.items():
        report[team_number] = {
            "style":        classification["primary_style"],
            "secondary":    classification.get("secondary_style"),
            "confidence":   classification["confidence"],
            "style_scores": classification["style_scores"],
            "key_evidence": classification["key_evidence"],
            "metrics":      all_metrics.get(
                team_number,
                classification.get("metrics", {})
            ),
        }

    return report
