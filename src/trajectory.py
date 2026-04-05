"""
src/trajectory.py - Phase 7: Trajectory Engine

Predicts ball flight paths and detects scoring zone entries.

Functions:
  compute_ball_velocity()    - dx/dy per frame from recent track history
  predict_trajectory()       - linear + parabolic position prediction
  will_enter_zone()          - check if predicted path crosses a scoring zone
  detect_scoring_event()     - detect when a ball enters a scoring zone

Depends on: Phase 5 (ball_tracks), Phase 6 (possession_log),
            configs/field_config.json (scoring_zones).
"""

from __future__ import annotations

import json
import math
from pathlib import Path


# ---- Config -----------------------------------------------------------------

def _load_field_config(config_path: str | Path = "configs/field_config.json") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ---- Geometry ---------------------------------------------------------------

def _bbox_centre(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in_bbox(px: float, py: float, bbox: list[float]) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


# ---- compute_ball_velocity --------------------------------------------------

def compute_ball_velocity(
    ball_track_id: int,
    ball_tracks:   dict[int, list[dict]],
    frame_window:  int = 5,
    at_frame:      int | None = None,
) -> dict:
    """
    Estimate ball velocity at a given frame using recent track history.

    Uses linear regression over the last frame_window positions to smooth
    noisy detection positions.

    Args:
        ball_track_id: Ball track to analyse.
        ball_tracks:   {track_id: [{frame_id, bbox, ...}]}
        frame_window:  Number of recent frames to use.
        at_frame:      Frame to evaluate at (default: latest).

    Returns:
        {vx: float, vy: float, speed: float, heading_deg: float}
        All values in pixels per frame.
        Returns zero-velocity dict if insufficient history.
    """
    entries = ball_tracks.get(ball_track_id, [])
    if not entries:
        return {"vx": 0.0, "vy": 0.0, "speed": 0.0, "heading_deg": 0.0}

    # Avoid re-sorting if already sorted (detect_all_scoring_events pre-sorts)
    if len(entries) > 1 and entries[0]["frame_id"] <= entries[-1]["frame_id"]:
        sorted_entries = entries
    else:
        sorted_entries = sorted(entries, key=lambda e: e["frame_id"])

    if at_frame is not None:
        # Binary search instead of linear filter
        lo, hi = 0, len(sorted_entries)
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_entries[mid]["frame_id"] <= at_frame:
                lo = mid + 1
            else:
                hi = mid
        sorted_entries = sorted_entries[:lo]

    recent = sorted_entries[-frame_window:]
    if len(recent) < 2:
        return {"vx": 0.0, "vy": 0.0, "speed": 0.0, "heading_deg": 0.0}

    # Weighted linear regression (more weight to recent frames)
    n = len(recent)
    weights = [float(i + 1) for i in range(n)]
    frames  = [e["frame_id"] for e in recent]
    xs      = [_bbox_centre(e["bbox"])[0] for e in recent]
    ys      = [_bbox_centre(e["bbox"])[1] for e in recent]

    # Weighted means
    w_sum = sum(weights)
    f_mean = sum(w * f for w, f in zip(weights, frames)) / w_sum
    x_mean = sum(w * x for w, x in zip(weights, xs))     / w_sum
    y_mean = sum(w * y for w, y in zip(weights, ys))     / w_sum

    # Weighted slopes
    denom = sum(w * (f - f_mean) ** 2 for w, f in zip(weights, frames))
    if abs(denom) < 1e-9:
        return {"vx": 0.0, "vy": 0.0, "speed": 0.0, "heading_deg": 0.0}

    vx = sum(w * (f - f_mean) * (x - x_mean)
             for w, f, x in zip(weights, frames, xs)) / denom
    vy = sum(w * (f - f_mean) * (y - y_mean)
             for w, f, y in zip(weights, frames, ys)) / denom

    speed       = math.sqrt(vx * vx + vy * vy)
    heading_deg = math.degrees(math.atan2(-vy, vx)) % 360   # screen y inverted

    return {
        "vx":          round(vx, 4),
        "vy":          round(vy, 4),
        "speed":       round(speed, 4),
        "heading_deg": round(heading_deg, 2),
    }


# ---- predict_trajectory -----------------------------------------------------

def predict_trajectory(
    ball_position: tuple[float, float],
    ball_velocity: dict,
    frames_ahead:  int = 30,
) -> list[tuple[float, float]]:
    """
    Predict future ball positions using linear extrapolation.

    For FRC field footage (top-down/angled view), linear prediction is
    sufficient for short-range scoring trajectories.  A simple gravity
    term is added on the y-axis for lobbed shots.

    Args:
        ball_position: (cx, cy) in pixels at the current frame.
        ball_velocity: {vx, vy, speed, heading_deg} from compute_ball_velocity.
        frames_ahead:  Number of future frames to predict.

    Returns:
        List of (cx, cy) predicted positions for frames 1..frames_ahead.
    """
    cx, cy = ball_position
    vx     = ball_velocity.get("vx", 0.0)
    vy     = ball_velocity.get("vy", 0.0)
    speed  = ball_velocity.get("speed", 0.0)

    # Light gravity effect for lobbed / bounced balls (y increases downward in
    # screen coords).  Only applied when the ball has significant vertical
    # component and is moving upward (vy < 0 in screen coords).
    gravity_px_per_frame2 = 0.3 if vy < -1.0 and speed > 5.0 else 0.0

    positions = []
    for t in range(1, frames_ahead + 1):
        pred_x = cx + vx * t
        pred_y = cy + vy * t + 0.5 * gravity_px_per_frame2 * t * t
        positions.append((round(pred_x, 2), round(pred_y, 2)))

    return positions


# ---- will_enter_zone --------------------------------------------------------

def will_enter_zone(
    predicted_positions: list[tuple[float, float]],
    zone_bbox:           list[float],
) -> dict:
    """
    Check whether any predicted ball position falls inside a scoring zone.

    Args:
        predicted_positions: List of (cx, cy) from predict_trajectory().
        zone_bbox:           [x1, y1, x2, y2] of the scoring zone in pixels.

    Returns:
        {
            will_score:       bool,
            predicted_frame:  int,   # frames from now (1-based), or -1
            confidence:       float, # based on how central the entry point is
        }
    """
    x1, y1, x2, y2 = zone_bbox
    zw = max(x2 - x1, 1)
    zh = max(y2 - y1, 1)

    for i, (px, py) in enumerate(predicted_positions):
        if _point_in_bbox(px, py, zone_bbox):
            # Confidence: how close to zone centre vs zone edge
            zone_cx = (x1 + x2) / 2
            zone_cy = (y1 + y2) / 2
            dist_from_centre = math.sqrt((px - zone_cx) ** 2 + (py - zone_cy) ** 2)
            max_dist = math.sqrt((zw / 2) ** 2 + (zh / 2) ** 2)
            conf = max(0.1, 1.0 - dist_from_centre / max_dist)
            return {
                "will_score":      True,
                "predicted_frame": i + 1,
                "confidence":      round(conf, 3),
            }

    return {"will_score": False, "predicted_frame": -1, "confidence": 0.0}


# ---- detect_scoring_event ---------------------------------------------------

def detect_scoring_event(
    ball_track_id:  int,
    ball_tracks:    dict[int, list[dict]],
    scoring_zones:  dict[str, list[float]],
    possession_log: dict[int, list[dict]],
    config_path:    str | Path = "configs/field_config.json",
) -> list[dict]:
    """
    Detect all frames where a ball enters a scoring zone.

    For each zone entry:
      - Records the event frame and zone name
      - Looks up the last possessor from the possession log
      - Estimates trajectory confidence

    Args:
        ball_track_id:  Ball track to analyse.
        ball_tracks:    {track_id: [{frame_id, bbox, ...}]}
        scoring_zones:  {"zone_name": [x1, y1, x2, y2], ...}
        possession_log: Output of possession.build_possession_log().
        config_path:    Path to field_config.json (for frames_ahead default).

    Returns:
        List of scoring event dicts:
        [{
            event_frame:                int,
            zone:                       str,
            ball_track_id:              int,
            last_possessor:             str (team_number),
            last_possessor_frame:       int,
            trajectory_origin_robot:    str (team_number) or None,
            confidence:                 float,
        }]
    """
    from possession import get_last_possessor

    cfg         = _load_field_config(config_path)
    frames_ahead = cfg.get("trajectory_frames_ahead", 30)

    entries = sorted(ball_tracks.get(ball_track_id, []), key=lambda e: e["frame_id"])
    if not entries:
        return []

    events: list[dict] = []
    in_zone_last_frame: set[str] = set()   # zones the ball was already inside

    for entry in entries:
        fid = entry["frame_id"]
        cx, cy = _bbox_centre(entry["bbox"])

        for zone_name, zone_bbox in scoring_zones.items():
            currently_inside = _point_in_bbox(cx, cy, zone_bbox)

            if currently_inside and zone_name not in in_zone_last_frame:
                # New zone entry — check ball is actually flying, not drifting
                vel  = compute_ball_velocity(ball_track_id, ball_tracks,
                                              at_frame=fid)
                entry_speed = math.sqrt(vel.get("vx", 0)**2 + vel.get("vy", 0)**2)
                if entry_speed < _MIN_ENTRY_SPEED_PX_PER_FRAME:
                    in_zone_last_frame.add(zone_name)
                    continue   # ball drifted in — not a score
                pred = predict_trajectory((cx, cy), vel, frames_ahead)
                zone_check = will_enter_zone(pred, zone_bbox)

                # Trajectory origin: look back for the robot that last held ball
                last_poss = get_last_possessor(ball_track_id, fid, possession_log)
                last_team = last_poss["team_number"] if last_poss else None
                last_frame = last_poss["frame_id"]   if last_poss else None

                # Confidence combines zone-entry certainty + possession recency
                recency_conf = 1.0
                if last_frame is not None:
                    frames_since = fid - last_frame
                    recency_conf = max(0.3, 1.0 - frames_since / 90.0)

                confidence = round(
                    0.6 * zone_check.get("confidence", 0.5) + 0.4 * recency_conf, 3
                )

                events.append({
                    "event_frame":             fid,
                    "zone":                    zone_name,
                    "ball_track_id":           ball_track_id,
                    "last_possessor":          last_team,
                    "last_possessor_frame":    last_frame,
                    "trajectory_origin_robot": last_team,
                    "confidence":              confidence,
                })
                in_zone_last_frame.add(zone_name)
            elif not currently_inside:
                in_zone_last_frame.discard(zone_name)

    return events


# ---- Batch over all balls ---------------------------------------------------

# Minimum peak speed (px/frame) a ball must reach to be considered in-flight.
# Lowered from 6.0 -> 2.0: FRC Reefscape algae is often hand-placed into the
# processor at slow speeds. We still need SOME movement to avoid counting
# static pre-loaded game pieces that happen to sit inside the zone.
_MIN_FLIGHT_SPEED_PX_PER_FRAME = 2.0

# Minimum speed when entering the scoring zone (px/frame).
# Lowered from 4.0 -> 1.5: catch slow hand-placement into processor zone.
_MIN_ENTRY_SPEED_PX_PER_FRAME = 1.5


def _track_peak_speed(entries: list[dict]) -> float:
    """Return the maximum frame-to-frame speed seen in a ball track."""
    peak = 0.0
    for i in range(1, len(entries)):
        cx0, cy0 = _bbox_centre(entries[i-1]["bbox"])
        cx1, cy1 = _bbox_centre(entries[i]["bbox"])
        df = max(1, entries[i]["frame_id"] - entries[i-1]["frame_id"])
        speed = math.sqrt((cx1-cx0)**2 + (cy1-cy0)**2) / df
        if speed > peak:
            peak = speed
    return peak


def detect_all_scoring_events(
    ball_tracks:    dict[int, list[dict]],
    scoring_zones:  dict[str, list[float]],
    possession_log: dict[int, list[dict]],
) -> list[dict]:
    """
    Run detect_scoring_event for every ball track that shows in-flight motion.

    Ground balls (slow-moving, stationary) are skipped entirely.
    Only tracks that reach _MIN_FLIGHT_SPEED_PX_PER_FRAME at any point are
    considered as potential scored balls.
    """
    # Pre-sort all tracks once
    sorted_ball_tracks: dict[int, list[dict]] = {
        tid: sorted(entries, key=lambda e: e["frame_id"])
        for tid, entries in ball_tracks.items()
    }

    # Filter to in-flight tracks only
    flight_tracks = {
        tid: entries
        for tid, entries in sorted_ball_tracks.items()
        if _track_peak_speed(entries) >= _MIN_FLIGHT_SPEED_PX_PER_FRAME
    }

    skipped = len(sorted_ball_tracks) - len(flight_tracks)
    print(f"  [Trajectory] {len(flight_tracks)} in-flight ball tracks "
          f"({skipped} ground/stationary balls skipped)")

    all_events: list[dict] = []
    for ball_tid in flight_tracks:
        all_events.extend(
            detect_scoring_event(ball_tid, flight_tracks, scoring_zones, possession_log)
        )
    all_events.sort(key=lambda e: e["event_frame"])

    # ── Temporal deduplication ─────────────────────────────────────────────────
    # With interpolated tracks, multiple track fragments of the same physical ball
    # can each trigger a scoring event at the same zone. Apply a per-zone cooldown:
    # after a score is counted, suppress any further events in the same zone for
    # COOLDOWN_FRAMES (default 45 = 0.75s at 60fps).
    # This also merges events where the same ball track's interpolated segments
    # overlap the zone independently.
    #
    # Cooldown is loaded from field_config.json if present, otherwise 45 frames.
    try:
        import json as _json
        _cfg = _json.loads(Path("configs/field_config.json").read_text())
        _cooldown = int(_cfg.get("scoring_cooldown_frames", 45))
    except Exception:
        _cooldown = 45

    last_event_frame: dict[str, int] = {}   # zone_name -> last accepted frame
    deduped: list[dict] = []
    for ev in all_events:
        zone = ev["zone"]
        last = last_event_frame.get(zone, -9999)
        if ev["event_frame"] - last >= _cooldown:
            deduped.append(ev)
            last_event_frame[zone] = ev["event_frame"]

    n_removed = len(all_events) - len(deduped)
    if n_removed:
        print(f"  [Trajectory] Temporal dedup removed {n_removed} duplicate events "
              f"(cooldown={_cooldown} frames). {len(deduped)} events remain.")

    print(f"  [Trajectory] Detected {len(deduped)} scoring events")
    return deduped
