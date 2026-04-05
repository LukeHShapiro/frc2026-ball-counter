"""
src/possession.py - Phase 6: Possession Engine

CRITICAL module. Determines which robot controls each ball at every frame.

Functions:
  assign_possession()      - per-frame ball->robot assignment
  build_possession_log()   - full match possession timeline
  get_last_possessor()     - last robot holding a ball before a given frame
  detect_handoff()         - detect ball passed between robots

Depends on: Phase 5 track.py output (ball_tracks, robot_tracks).

(!) STOP after tests pass --
   possession_log feeds into src/trajectory.py and src/inference_engine.py.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


# ---- Constants (loaded from field_config.json) ------------------------------

def _load_possession_rules(config_path: str | Path = "configs/field_config.json") -> dict:
    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            cfg = json.load(f)
        return cfg.get("possession_rules", _DEFAULT_RULES)
    return _DEFAULT_RULES


_DEFAULT_RULES = {
    "proximity_threshold_px":      80,
    "possession_min_frames":        3,
    "release_threshold_px":        150,
    "velocity_handoff_threshold":    5,
}


# ---- Geometry helpers -------------------------------------------------------

def _bbox_centre(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _dist_point_to_bbox(px: float, py: float, bbox: list[float]) -> float:
    """
    Shortest distance from point (px, py) to the edge of bbox.
    Returns 0 if the point is inside the bbox.
    """
    x1, y1, x2, y2 = bbox
    dx = max(x1 - px, 0.0, px - x2)
    dy = max(y1 - py, 0.0, py - y2)
    return (dx * dx + dy * dy) ** 0.5


def _bbox_to_bbox_dist(bbox_a: list[float], bbox_b: list[float]) -> float:
    """Shortest edge-to-edge distance between two bboxes."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    dx = max(ax1 - bx2, 0.0, bx1 - ax2)
    dy = max(ay1 - by2, 0.0, by1 - ay2)
    return (dx * dx + dy * dy) ** 0.5


def _ball_speed(ball_track_id: int,
                ball_tracks: dict[int, list[dict]],
                frame_id: int,
                window: int = 3) -> float:
    """
    Estimate ball speed (px/frame) at frame_id using recent track history.
    Returns 0 if insufficient history.
    """
    entries = ball_tracks.get(ball_track_id, [])
    entries_before = [e for e in entries if e["frame_id"] <= frame_id]
    if len(entries_before) < 2:
        return 0.0
    recent = sorted(entries_before, key=lambda e: e["frame_id"])[-window:]
    if len(recent) < 2:
        return 0.0
    cx0, cy0 = _bbox_centre(recent[0]["bbox"])
    cx1, cy1 = _bbox_centre(recent[-1]["bbox"])
    n_frames = max(1, recent[-1]["frame_id"] - recent[0]["frame_id"])
    return ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5 / n_frames


# ---- assign_possession ------------------------------------------------------

def _build_frame_index(tracks: dict[int, list[dict]]) -> dict[int, dict[int, dict]]:
    """
    Pre-index track entries by frame_id for O(1) lookup.

    Returns:
        {frame_id: {track_id: entry}}
    """
    index: dict[int, dict[int, dict]] = {}
    for tid, entries in tracks.items():
        for e in entries:
            fid = e["frame_id"]
            if fid not in index:
                index[fid] = {}
            index[fid][tid] = e
    return index


def _build_speed_index(ball_tracks: dict[int, list[dict]],
                        window: int = 3) -> dict[tuple, float]:
    """
    Pre-compute ball speed at every (track_id, frame_id) that appears in ball_tracks.
    Returns a dict keyed by (track_id, frame_id) -> speed in px/frame.
    """
    speed_index: dict[tuple, float] = {}
    for tid, entries in ball_tracks.items():
        sorted_entries = sorted(entries, key=lambda e: e["frame_id"])
        for i, e in enumerate(sorted_entries):
            fid = e["frame_id"]
            recent = sorted_entries[max(0, i - window + 1): i + 1]
            if len(recent) < 2:
                speed_index[(tid, fid)] = 0.0
            else:
                cx0, cy0 = _bbox_centre(recent[0]["bbox"])
                cx1, cy1 = _bbox_centre(recent[-1]["bbox"])
                n = max(1, recent[-1]["frame_id"] - recent[0]["frame_id"])
                speed_index[(tid, fid)] = ((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2) ** 0.5 / n
    return speed_index


def assign_possession(
    ball_tracks:   dict[int, list[dict]],
    robot_tracks:  dict[int, list[dict]],
    frame_id:      int,
    rules:         dict | None = None,
    # Pre-built indices (pass these from build_possession_log to avoid O(n²))
    _ball_idx:     dict | None = None,
    _robot_idx:    dict | None = None,
    _speed_idx:    dict | None = None,
) -> dict[int, dict]:
    """
    For each ball visible in frame_id, determine which robot (if any) possesses it.

    Possession conditions (both must hold for possession_min_frames):
      1. Ball centre is within proximity_threshold_px of a robot bbox edge.
      2. Ball speed < velocity_handoff_threshold px/frame (held, not in flight).

    Args:
        ball_tracks:   {track_id: [{frame_id, bbox, ...}]}
        robot_tracks:  {track_id: [{frame_id, bbox, team_number?, ...}]}
        frame_id:      Current frame to evaluate.
        rules:         Possession rules dict (loaded from config if None).

    Returns:
        {ball_track_id: {robot_track_id, team_number, confidence, frame_id}}
        Empty dict if no ball is possessed this frame.
    """
    if rules is None:
        rules = _load_possession_rules()

    prox_thresh = rules["proximity_threshold_px"]
    vel_thresh  = rules["velocity_handoff_threshold"]

    # Use pre-built indices if provided, else build on the fly (slow path)
    ball_frame  = (_ball_idx  or _build_frame_index(ball_tracks )).get(frame_id, {})
    robot_frame = (_robot_idx or _build_frame_index(robot_tracks)).get(frame_id, {})

    result: dict[int, dict] = {}

    # Build numpy array of robot bboxes for vectorised distance computation
    robot_tids    = list(robot_frame.keys())
    robot_entries = list(robot_frame.values())
    if robot_tids:
        # robot_bboxes: (N_robots, 4)  columns: x1, y1, x2, y2
        robot_bboxes = np.array([e["bbox"] for e in robot_entries], dtype=np.float32)

    for ball_tid, ball_entry in ball_frame.items():
        bcx, bcy = _bbox_centre(ball_entry["bbox"])

        # Speed lookup: use pre-computed index if available
        if _speed_idx is not None:
            speed = _speed_idx.get((ball_tid, frame_id), 0.0)
        else:
            speed = _ball_speed(ball_tid, ball_tracks, frame_id)

        if speed > vel_thresh or not robot_tids:
            continue  # ball in flight or no robots visible

        # Vectorised edge-to-edge distance: ball centre → each robot bbox
        dx = np.maximum(robot_bboxes[:, 0] - bcx, 0.0)
        dx = np.maximum(dx, bcx - robot_bboxes[:, 2])
        dy = np.maximum(robot_bboxes[:, 1] - bcy, 0.0)
        dy = np.maximum(dy, bcy - robot_bboxes[:, 3])
        dists = np.sqrt(dx * dx + dy * dy)   # (N_robots,)

        best_idx  = int(np.argmin(dists))
        best_dist = float(dists[best_idx])

        if best_dist > prox_thresh:
            continue

        best_robot   = robot_tids[best_idx]
        best_r_entry = robot_entries[best_idx]

        confidence = max(0.0, 1.0 - best_dist / prox_thresh) * (
            1.0 - min(speed, vel_thresh) / vel_thresh
        )

        result[ball_tid] = {
            "robot_track_id": best_robot,
            "team_number":    best_r_entry.get("team_number", f"UNKNOWN_{best_robot}"),
            "confidence":     round(confidence, 3),
            "frame_id":       frame_id,
        }

    return result


# ---- build_possession_log ---------------------------------------------------

def _possession_cache_sig(
    ball_tracks:  dict[int, list[dict]],
    robot_tracks: dict[int, list[dict]],
) -> str:
    """
    Short hash of ball+robot track data so we can detect when recompute is needed.
    Includes team numbers so identity changes (OCR corrections, TBA assignments)
    correctly invalidate the cache.
    """
    n_ball_entries  = sum(len(v) for v in ball_tracks.values())
    n_robot_entries = sum(len(v) for v in robot_tracks.values())
    n_ball_tracks   = len(ball_tracks)
    n_robot_tracks  = len(robot_tracks)
    # Include sorted team number list so identity changes bust the cache
    team_names = sorted(
        ents[0].get("team_number", "") if ents else ""
        for ents in robot_tracks.values()
    )
    sig_str = (f"{n_ball_tracks}:{n_ball_entries}:"
               f"{n_robot_tracks}:{n_robot_entries}:"
               f"{'|'.join(team_names)}")
    return hashlib.md5(sig_str.encode()).hexdigest()[:12]


_POSSESSION_CACHE_PATH = Path("data/possession_log.json")
_POSSESSION_SIG_PATH   = Path("data/possession_log.sig")


def build_possession_log(
    ball_tracks:   dict[int, list[dict]],
    robot_tracks:  dict[int, list[dict]],
    total_frames:  int,
    rules:         dict | None = None,
    use_cache:     bool = True,
) -> dict[int, list[dict]]:
    """
    Build a full match possession timeline by running assign_possession on
    every frame that has at least one ball or robot detection.

    Applies possession_min_frames confirmation window: a robot is only
    credited with possession if it holds the ball for N consecutive frames.

    Args:
        ball_tracks:   {track_id: [{frame_id, bbox, ...}]}
        robot_tracks:  {track_id: [{frame_id, bbox, ...}]}
        total_frames:  Total frame count of the match video.
        rules:         Possession rules dict.
        use_cache:     Load from data/possession_log.json if tracks haven't changed.

    Returns:
        {ball_track_id: [{frame_id, robot_track_id, team_number, confidence}]}
        Only confirmed possession events (>= possession_min_frames).
    """
    if rules is None:
        rules = _load_possession_rules()

    # ── Cache check ────────────────────────────────────────────────────────────
    if use_cache and _POSSESSION_CACHE_PATH.exists() and _POSSESSION_SIG_PATH.exists():
        sig = _possession_cache_sig(ball_tracks, robot_tracks)
        try:
            if _POSSESSION_SIG_PATH.read_text().strip() == sig:
                raw = json.loads(_POSSESSION_CACHE_PATH.read_text())
                log = {int(k): v for k, v in raw.items()}
                total_events = sum(len(v) for v in log.values())
                print(f"  [Possession] Loaded from cache: {len(log)} ball tracks, "
                      f"{total_events} events (detections unchanged).")
                return log
        except Exception:
            pass  # cache corrupt or unreadable — recompute

    min_frames = rules["possession_min_frames"]

    # Collect all frame IDs that appear in any track
    all_frame_ids = sorted({
        e["frame_id"]
        for tracks in (ball_tracks, robot_tracks)
        for entries in tracks.values()
        for e in entries
    })

    # Pre-build indices once — avoids O(frames² × tracks) linear scans
    ball_idx  = _build_frame_index(ball_tracks)
    robot_idx = _build_frame_index(robot_tracks)
    speed_idx = _build_speed_index(ball_tracks)

    total = len(all_frame_ids)
    print(f"  [Possession] Processing {total} frames...")

    # Build a set for fast frame membership checks (used in backfill below)
    all_frame_set = set(all_frame_ids)

    # Raw per-frame assignments
    raw: dict[int, dict[int, dict]] = {}   # frame_id -> {ball_tid: assignment}
    for i, fid in enumerate(all_frame_ids):
        raw[fid] = assign_possession(
            ball_tracks, robot_tracks, fid, rules,
            _ball_idx=ball_idx, _robot_idx=robot_idx, _speed_idx=speed_idx,
        )
        if i > 0 and i % 500 == 0:
            print(f"  [Possession] {i}/{total} frames done...")

    # Confirm possession using sliding window
    log: dict[int, list[dict]] = {}

    for ball_tid in ball_tracks:
        frames_with_ball = [fid for fid in all_frame_ids if ball_tid in raw.get(fid, {})]
        if not frames_with_ball:
            continue

        confirmed: list[dict] = []
        run_start_idx = None   # index into frames_with_ball (not a frame ID)
        run_robot     = None
        run_length    = 0

        for fi, fid in enumerate(frames_with_ball):
            assignment = raw[fid].get(ball_tid)
            if assignment is None:
                run_start_idx = None
                run_robot     = None
                run_length    = 0
                continue

            robot_tid = assignment["robot_track_id"]
            if robot_tid == run_robot:
                run_length += 1
                if run_length == min_frames:
                    # Confirmed — back-fill using frames_with_ball slice (not range())
                    back_start = run_start_idx if run_start_idx is not None else fi - run_length + 1
                    for back_fi in range(back_start, fi + 1):
                        back_fid = frames_with_ball[back_fi]
                        if ball_tid in raw.get(back_fid, {}):
                            confirmed.append({
                                "frame_id":       back_fid,
                                "robot_track_id": robot_tid,
                                "team_number":    raw[back_fid][ball_tid]["team_number"],
                                "confidence":     raw[back_fid][ball_tid]["confidence"],
                            })
                elif run_length > min_frames:
                    confirmed.append({
                        "frame_id":       fid,
                        "robot_track_id": robot_tid,
                        "team_number":    assignment["team_number"],
                        "confidence":     assignment["confidence"],
                    })
            else:
                run_robot     = robot_tid
                run_start_idx = fi
                run_length    = 1

        if confirmed:
            log[ball_tid] = confirmed

    total_events = sum(len(v) for v in log.values())
    print(f"  [Possession] Built log: {len(log)} ball tracks, "
          f"{total_events} confirmed possession events")

    # ── Persist to cache ───────────────────────────────────────────────────────
    try:
        _POSSESSION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {str(k): v for k, v in log.items()}
        _POSSESSION_CACHE_PATH.write_text(json.dumps(serialisable))
        _POSSESSION_SIG_PATH.write_text(_possession_cache_sig(ball_tracks, robot_tracks))
    except Exception:
        pass  # cache write failure is non-fatal

    return log


# ---- get_last_possessor -----------------------------------------------------

def get_last_possessor(
    ball_track_id: int,
    before_frame:  int,
    possession_log: dict[int, list[dict]],
) -> dict | None:
    """
    Return the last robot confirmed holding a ball before a given frame.

    CRITICAL: used by the attribution engine when the robot is not visible
    at the moment of scoring.

    Args:
        ball_track_id:  Ball track ID to query.
        before_frame:   Look for possession entries with frame_id < this value.
        possession_log: Output of build_possession_log().

    Returns:
        {robot_track_id, team_number, confidence, frame_id} of last possessor,
        or None if no possession was recorded.
    """
    entries = possession_log.get(ball_track_id, [])
    candidates = [e for e in entries if e["frame_id"] < before_frame]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e["frame_id"])


# ---- detect_handoff ---------------------------------------------------------

def detect_handoff(
    ball_track_id:  int,
    possession_log: dict[int, list[dict]],
    frame_window:   int = 15,
) -> dict | None:
    """
    Detect if a ball was passed/handed from one robot to another.

    A handoff is detected when:
      - Possessor A holds ball, then ball is uncontrolled for <= frame_window frames,
        then Possessor B holds it.

    Args:
        ball_track_id:  Ball track ID to check.
        possession_log: Output of build_possession_log().
        frame_window:   Max frames of uncontrolled flight to still count as a handoff.

    Returns:
        {from_robot: int, to_robot: int, handoff_frame: int,
         from_team: str, to_team: str}
        or None if no handoff detected.
    """
    entries = sorted(possession_log.get(ball_track_id, []), key=lambda e: e["frame_id"])
    if len(entries) < 2:
        return None

    # Find transitions between different robots
    prev = entries[0]
    for curr in entries[1:]:
        if curr["robot_track_id"] == prev["robot_track_id"]:
            prev = curr
            continue
        gap = curr["frame_id"] - prev["frame_id"]
        if gap <= frame_window:
            return {
                "from_robot":   prev["robot_track_id"],
                "to_robot":     curr["robot_track_id"],
                "handoff_frame": curr["frame_id"],
                "from_team":    prev["team_number"],
                "to_team":      curr["team_number"],
                "gap_frames":   gap,
            }
        prev = curr

    return None


# ---- detect_ball_exits ------------------------------------------------------

def detect_ball_exits(
    ball_tracks:        dict[int, list[dict]],
    possession_log:     dict[int, list[dict]],
    release_gap_frames: int = 10,
) -> list[dict]:
    """
    Detect all ball exit events — moments when a ball leaves a robot's control.

    A ball "exits" when the possession_log shows a run of possession ending:
    either the ball is never re-possessed (shot/scored) or it becomes
    uncontrolled for > release_gap_frames before being re-possessed or
    switching to a different robot.

    Exit velocity and position are computed from ball_tracks at the exit frame.

    Args:
        ball_tracks:        {track_id: [{frame_id, bbox, ...}]}
        possession_log:     From build_possession_log().
        release_gap_frames: Gap (frames) that constitutes a release vs. missed detection.

    Returns:
        List of exit events sorted by exit_frame:
        [{
            ball_track_id:  int,
            robot_track_id: int,
            team_number:    str,
            exit_frame:     int,    # frame after last confirmed possession
            exit_velocity:  dict,   # {vx, vy, speed, heading_deg}
            exit_position:  tuple,  # (cx, cy) in pixels
        }]
    """
    # Inline helpers to avoid circular import (trajectory imports possession)
    def _bbox_centre(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    def compute_ball_velocity(ball_tid, b_tracks, at_frame, window=5):
        entries = sorted(b_tracks.get(ball_tid, []), key=lambda e: e["frame_id"])
        before  = [e for e in entries if e["frame_id"] <= at_frame]
        if len(before) < 2:
            return {"vx": 0.0, "vy": 0.0, "speed": 0.0, "heading_deg": 0.0}
        recent = before[-window:]
        cx0, cy0 = _bbox_centre(recent[0]["bbox"])
        cx1, cy1 = _bbox_centre(recent[-1]["bbox"])
        n  = max(1, recent[-1]["frame_id"] - recent[0]["frame_id"])
        vx, vy = (cx1 - cx0) / n, (cy1 - cy0) / n
        import math
        return {"vx": vx, "vy": vy,
                "speed": math.hypot(vx, vy),
                "heading_deg": math.degrees(math.atan2(-vy, vx))}

    exits: list[dict] = []

    for ball_tid, entries in possession_log.items():
        if not entries:
            continue

        sorted_entries = sorted(entries, key=lambda e: e["frame_id"])

        # Group consecutive frames with the same robot into "possession runs"
        runs: list[list[dict]] = []
        current_run: list[dict] = [sorted_entries[0]]

        for j in range(1, len(sorted_entries)):
            prev = sorted_entries[j - 1]
            curr = sorted_entries[j]
            gap          = curr["frame_id"] - prev["frame_id"]
            same_robot   = curr["robot_track_id"] == prev["robot_track_id"]

            if gap <= release_gap_frames and same_robot:
                current_run.append(curr)
            else:
                runs.append(current_run)
                current_run = [curr]
        runs.append(current_run)

        # Each run ending = one ball exit
        for run in runs:
            last = run[-1]
            exit_frame = last["frame_id"] + 1

            # Find ball position at / just after exit frame
            ball_entries_sorted = sorted(
                ball_tracks.get(ball_tid, []), key=lambda e: e["frame_id"]
            )
            # Pick the first ball entry at or after exit_frame, else last known
            ball_at_exit = next(
                (be for be in ball_entries_sorted if be["frame_id"] >= exit_frame),
                ball_entries_sorted[-1] if ball_entries_sorted else None,
            )

            if ball_at_exit:
                pos = _bbox_centre(ball_at_exit["bbox"])
                vel = compute_ball_velocity(ball_tid, ball_tracks,
                                            at_frame=exit_frame)
            else:
                pos = (0.0, 0.0)
                vel = {"vx": 0.0, "vy": 0.0, "speed": 0.0, "heading_deg": 0.0}

            exits.append({
                "ball_track_id":  ball_tid,
                "robot_track_id": last["robot_track_id"],
                "team_number":    last["team_number"],
                "exit_frame":     exit_frame,
                "exit_velocity":  vel,
                "exit_position":  pos,
            })

    exits.sort(key=lambda e: e["exit_frame"])
    print(f"  [Possession] Detected {len(exits)} ball exit events "
          f"across {len(possession_log)} ball tracks")
    return exits


# ---- Save / load ------------------------------------------------------------

def save_possession_log(
    log:      dict[int, list[dict]],
    out_path: str | Path = "data/possession_log.json",
) -> Path:
    """Persist the possession log to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert int keys to strings for JSON
    serialisable = {str(k): v for k, v in log.items()}
    with open(out_path, "w") as f:
        json.dump(serialisable, f)
    print(f"  [Possession] Log saved -> {out_path}")
    return out_path


def load_possession_log(path: str | Path = "data/possession_log.json") -> dict[int, list[dict]]:
    """Load a previously saved possession log."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}
