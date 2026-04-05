"""
src/export.py — Phase 11 + Phase 13 + Phase 14: Export functions

Exports analysis results to CSV, JSON, and annotated video.

Phase 11 functions:
  export_csv()               — Per-scoring-event CSV
  export_json()              — Full results JSON
  export_annotated_video()   — Video with overlays + driving style badges (Phase 13)

Phase 13 additions:
  export_driving_report_csv()
  export_driving_report_json()

Phase 14 additions:
  export_pick_list_csv()
  export_pick_list_json()
  export_do_not_pick_csv()
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# ── Driving style badge colours (BGR for OpenCV) ──────────────────────────────
STYLE_COLOURS = {
    "DEFENSIVE":    (0,   165, 255),   # Orange
    "RECKLESS":     (0,   0,   255),   # Red
    "SMOOTH":       (0,   200, 0),     # Green
    "DEFENCE_PROOF":(255, 100, 0),     # Blue
}

# ── Phase 11 ──────────────────────────────────────────────────────────────────

def export_csv(
    final_scores: dict,
    score_timeline: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Export per-scoring-event data to CSV.

    One row per scoring event:
      timestamp, team_number, method, confidence, zone, notes

    Args:
        final_scores:   Per-robot aggregated scores from Phase 10.
        score_timeline: List of scoring events from Phase 9.
        output_path:    Destination .csv file path.

    Returns:
        Path to the written CSV file.

    Depends on: Phase 9 score_timeline, Phase 10 final_scores.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "team_number", "method", "confidence", "zone", "notes"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event in score_timeline:
            writer.writerow({
                "timestamp":   event.get("timestamp", ""),
                "team_number": event.get("team_number", ""),
                "method":      event.get("method", ""),
                "confidence":  event.get("confidence", ""),
                "zone":        event.get("zone", ""),
                "notes":       event.get("notes", ""),
            })

    print(f"  [Export] Scoring CSV written -> {output_path}")
    return output_path


def export_json(
    full_results: dict,
    output_path: str | Path,
) -> Path:
    """
    Export the full analysis results as JSON.

    Args:
        full_results: Complete results dict (scores, timeline, metadata).
        output_path:  Destination .json file path.

    Returns:
        Path to the written JSON file.

    Depends on: Phase 9, Phase 10.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2, default=str)

    print(f"  [Export] Full results JSON written -> {output_path}")
    return output_path


def append_match_history(
    full_results: dict,
    match_label: str,
    history_path: str | Path | None = None,
) -> Path:
    """
    Append this match's per-team scores to the cumulative match_history.json.

    Each entry: {match_label, analyzed_at, video_file, teams: {team: score_dict}}

    Args:
        full_results:  The same dict passed to export_json().
        match_label:   Human-readable match name, e.g. "Qual 3".
        history_path:  Destination file. Defaults to data/exports/match_history.json.

    Returns:
        Path to the history file.
    """
    import datetime

    if history_path is None:
        history_path = Path(__file__).parent.parent / "data" / "exports" / "match_history.json"
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []

    final_scores = full_results.get("final_scores", {})
    teams_snapshot = {
        team: {
            "score":      data.get("score", 0),
            "high_conf":  data.get("high_conf", 0),
            "med_conf":   data.get("med_conf", 0),
            "low_conf":   data.get("low_conf", 0),
            "unattributed": data.get("unattributed", 0),
        }
        for team, data in final_scores.items()
        if team not in ("UNATTRIBUTED", "REPLACED")
    }

    entry = {
        "match_label":  match_label,
        "analyzed_at":  datetime.datetime.now().isoformat(timespec="seconds"),
        "video_file":   full_results.get("video_file", ""),
        "teams":        teams_snapshot,
    }

    # Replace existing entry with same match_label, or append
    idx = next((i for i, e in enumerate(history) if e.get("match_label") == match_label), None)
    if idx is not None:
        history[idx] = entry
    else:
        history.append(entry)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    print(f"  [Export] Match history updated ({len(history)} matches) -> {history_path}")
    return history_path


def export_robot_positions(
    robot_tracks: dict,
    robot_identity_map: dict,
    output_path: str | Path | None = None,
) -> Path:
    """
    Save per-team robot centroid positions for heatmap visualisation.

    Output format:
        {team_number: [[cx, cy, frame_id], ...]}

    Args:
        robot_tracks:       {track_id: [{frame_id, bbox, ...}]}
        robot_identity_map: {track_id: {team_number, ...}}
        output_path:        Defaults to data/exports/robot_positions.json.

    Returns:
        Path to the written file.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "exports" / "robot_positions.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positions: dict[str, list] = {}

    for track_id, entries in robot_tracks.items():
        team = robot_identity_map.get(track_id, {}).get("team_number", f"UNKNOWN_{track_id}")
        if str(team).startswith("UNKNOWN"):
            continue
        pts: list = []
        for e in entries:
            b = e["bbox"]
            cx = round((b[0] + b[2]) / 2, 1)
            cy = round((b[1] + b[3]) / 2, 1)
            pts.append([cx, cy, e["frame_id"]])
        if pts:
            if team in positions:
                positions[team].extend(pts)
            else:
                positions[team] = pts

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(positions, f)

    total = sum(len(v) for v in positions.values())
    print(f"  [Export] Robot positions: {len(positions)} teams, {total} points -> {output_path}")
    return output_path


def export_annotated_video(
    video_path: str | Path,
    ball_tracks: dict,
    robot_tracks: dict,
    possession_log: dict,
    score_timeline: list[dict],
    output_path: str | Path,
    driving_report: dict | None = None,
) -> Path:
    """
    Render an annotated video with all overlays.

    Per-frame draws:
      - Ball bboxes + track IDs
      - Robot bboxes + team numbers
      - Possession line: ball → owning robot (colour-coded by team)
      - Running per-robot score overlay (top-left corner, all 6 robots)
      - Confidence badge on each score event (+1 HIGH / +1 LOW / +1 ?)
      - Driving style badge under each robot bbox (Phase 13, if driving_report provided)
        Colours: DEFENSIVE=orange, RECKLESS=red, SMOOTH=green, DEFENCE_PROOF=blue

    Args:
        video_path:     Source video file.
        ball_tracks:    Ball tracking data from Phase 5.
        robot_tracks:   Robot tracking data from Phase 5.
        possession_log: Ball-to-robot possession from Phase 6.
        score_timeline: Scoring events from Phase 9.
        output_path:    Destination video file.
        driving_report: Optional Phase 13 driving report for style badges.

    Returns:
        Path to the annotated video file.

    Depends on: Phase 5, Phase 6, Phase 9, Phase 13 (optional).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("opencv-python required for annotated video export.")

    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Build per-frame lookup structures
    ball_by_frame: dict[int, list[dict]] = {}
    for tid, entries in ball_tracks.items():
        for e in entries:
            ball_by_frame.setdefault(e["frame_id"], []).append({**e, "track_id": tid})

    robot_by_frame: dict[int, list[dict]] = {}
    for tid, entries in robot_tracks.items():
        for e in entries:
            robot_by_frame.setdefault(e["frame_id"], []).append({**e, "track_id": tid})

    # Score event lookup by frame
    score_events_by_frame: dict[int, list[dict]] = {}
    for evt in score_timeline:
        score_events_by_frame.setdefault(evt.get("frame_id", -1), []).append(evt)

    # Running per-robot scores
    running_scores: dict[str, int] = {}

    # Team number → track_id reverse map (for driving report lookup)
    team_to_track: dict[str, int] = {}
    for tid, entries in robot_tracks.items():
        for e in entries:
            tn = e.get("team_number")
            if tn:
                team_to_track[tn] = tid

    # Palette for team colours
    TEAM_COLOURS = [
        (255, 80,  80),
        (80,  80,  255),
        (80,  200, 80),
        (200, 200, 80),
        (200, 80,  200),
        (80,  200, 200),
    ]
    team_colour_map: dict[str, tuple] = {}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── Score events this frame ───────────────────────────────────────────
        for evt in score_events_by_frame.get(frame_idx, []):
            team = evt.get("team_number", "?")
            running_scores[team] = running_scores.get(team, 0) + 1
            conf = evt.get("confidence", 0)
            badge = "+1 HIGH" if conf >= 0.85 else ("+1 LOW" if conf < 0.50 else "+1")
            cv2.putText(frame, badge, (width // 2 - 40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        # ── Running score overlay (top-left) ──────────────────────────────────
        y_off = 30
        for i, (team, score) in enumerate(running_scores.items()):
            colour = TEAM_COLOURS[i % len(TEAM_COLOURS)]
            cv2.putText(frame, f"Team {team}: {score}",
                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
            y_off += 22

        # ── Ball bboxes ───────────────────────────────────────────────────────
        for ball in ball_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in ball["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)
            cv2.putText(frame, f"B{ball['track_id']}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

        # ── Robot bboxes + possession lines + style badges ────────────────────
        for robot in robot_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in robot["bbox"]]
            team = robot.get("team_number", f"T{robot['track_id']}")

            if team not in team_colour_map:
                idx = len(team_colour_map) % len(TEAM_COLOURS)
                team_colour_map[team] = TEAM_COLOURS[idx]
            colour = team_colour_map[team]

            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, team, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

            # Driving style badge (Phase 13)
            if driving_report and team in driving_report:
                style = driving_report[team].get("style", "")
                badge_colour = STYLE_COLOURS.get(style, (200, 200, 200))
                cv2.putText(frame, f"[{style}]",
                            (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, badge_colour, 1)

        # ── Possession lines ──────────────────────────────────────────────────
        for ball_tid, poss_entries in possession_log.items():
            for poss in poss_entries:
                if poss.get("frame_id") != frame_idx:
                    continue
                robot_tid = poss.get("robot_track_id")
                robot_entries = [
                    r for r in robot_by_frame.get(frame_idx, [])
                    if r.get("track_id") == robot_tid
                ]
                ball_entries = [
                    b for b in ball_by_frame.get(frame_idx, [])
                    if b.get("track_id") == ball_tid
                ]
                if robot_entries and ball_entries:
                    rx = int((robot_entries[0]["bbox"][0] + robot_entries[0]["bbox"][2]) / 2)
                    ry = int((robot_entries[0]["bbox"][1] + robot_entries[0]["bbox"][3]) / 2)
                    bx = int((ball_entries[0]["bbox"][0] + ball_entries[0]["bbox"][2]) / 2)
                    by = int((ball_entries[0]["bbox"][1] + ball_entries[0]["bbox"][3]) / 2)
                    cv2.line(frame, (bx, by), (rx, ry), (255, 255, 0), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"  [Export] Annotated video written -> {output_path}")
    return output_path


# ── Phase 13 ──────────────────────────────────────────────────────────────────

def export_driving_report_csv(
    driving_report: dict,
    output_path: str | Path,
) -> Path:
    """
    Export driving analysis report as CSV — one row per robot.

    Columns:
      team_number, primary_style, secondary_style, confidence,
      avg_velocity, collision_count, shadowing_events,
      scoring_under_pressure_rate, escape_success_rate, path_repetition_score

    Args:
        driving_report: Output of generate_driving_report() from Phase 13.
        output_path:    Destination .csv file path.

    Returns:
        Path to the written CSV file.

    Depends on: Phase 13 driving_analysis.generate_driving_report().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "team_number", "primary_style", "secondary_style", "confidence",
        "avg_velocity", "collision_count", "shadowing_events",
        "scoring_under_pressure_rate", "escape_success_rate",
        "path_repetition_score",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for team, data in driving_report.items():
            m = data.get("metrics", {})
            writer.writerow({
                "team_number":                 team,
                "primary_style":               data.get("style", ""),
                "secondary_style":             data.get("secondary") or "",
                "confidence":                  data.get("confidence", ""),
                "avg_velocity":                m.get("avg_velocity_px_per_frame", ""),
                "collision_count":             m.get("collision_count", ""),
                "shadowing_events":            m.get("shadowing_events", ""),
                "scoring_under_pressure_rate": m.get("scoring_under_pressure_rate", ""),
                "escape_success_rate":         m.get("escape_success_rate", ""),
                "path_repetition_score":       m.get("path_repetition_score", ""),
            })

    print(f"  [Export] Driving report CSV written -> {output_path}")
    return output_path


def export_driving_report_json(
    driving_report: dict,
    output_path: str | Path,
) -> Path:
    """
    Export driving analysis report as JSON.

    Args:
        driving_report: Output of generate_driving_report() from Phase 13.
        output_path:    Destination .json file path.

    Returns:
        Path to the written JSON file.

    Depends on: Phase 13 driving_analysis.generate_driving_report().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(driving_report, f, indent=2, default=str)

    print(f"  [Export] Driving report JSON written -> {output_path}")
    return output_path


# ── Phase 14 ──────────────────────────────────────────────────────────────────

def export_pick_list_csv(
    pick_list: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Export the alliance pick list as CSV.

    Columns:
      rank, team_number, composite_score, driving_style, opr,
      video_score_rate, data_confidence, key_reasoning, warnings

    Args:
        pick_list:   Output of generate_pick_list() from Phase 14.
        output_path: Destination .csv file path.

    Returns:
        Path to the written CSV file.

    Depends on: Phase 14 alliance_builder.generate_pick_list().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank", "team_number", "composite_score", "driving_style",
        "opr", "video_score_rate", "data_confidence",
        "key_reasoning", "warnings",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in pick_list:
            warnings = entry.get("warnings", [])
            reasoning = entry.get("reasoning", [])
            writer.writerow({
                "rank":             entry.get("rank", ""),
                "team_number":      entry.get("team_number", ""),
                "composite_score":  entry.get("composite_score", ""),
                "driving_style":    entry.get("style", ""),
                "opr":              entry.get("tba_opr", ""),
                "video_score_rate": entry.get("video_score_rate", ""),
                "data_confidence":  entry.get("data_confidence", ""),
                "key_reasoning":    "; ".join(reasoning) if reasoning else "",
                "warnings":         "; ".join(warnings) if warnings else "",
            })

    print(f"  [Export] Pick list CSV written -> {output_path}")
    return output_path


def export_pick_list_json(
    full_alliance_recommendation: dict,
    output_path: str | Path,
) -> Path:
    """
    Export the full alliance recommendation as JSON.

    Args:
        full_alliance_recommendation: Output of recommend_picks() from Phase 14.
        output_path:                  Destination .json file path.

    Returns:
        Path to the written JSON file.

    Depends on: Phase 14 alliance_builder.recommend_picks().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_alliance_recommendation, f, indent=2, default=str)

    print(f"  [Export] Pick list JSON written -> {output_path}")
    return output_path


def export_do_not_pick_csv(
    risky_teams: list[dict],
    output_path: str | Path,
) -> Path:
    """
    Export the do-not-pick list as CSV.

    Columns: team_number, reason, composite_score

    Args:
        risky_teams: Output of recommend_do_not_pick() or flag_risky_teams()
                     from Phase 14.
        output_path: Destination .csv file path.

    Returns:
        Path to the written CSV file.

    Depends on: Phase 14 alliance_builder.recommend_do_not_pick().
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["team_number", "reason", "composite_score"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for team in risky_teams:
            writer.writerow({
                "team_number":     team.get("team_number", ""),
                "reason":          team.get("reason", ""),
                "composite_score": team.get("composite_score", ""),
            })

    print(f"  [Export] Do-not-pick CSV written -> {output_path}")
    return output_path
