"""
main.py - FRC 2026 Ball Counter - Full pipeline orchestrator

Phase execution order:
  Phase 2   ingest.py        extract_frames, filter_duplicates
  Phase 5   detect.py        process_video (Roboflow API)
             track.py        run_ball_tracker, run_robot_tracker,
                             calibrate_robot_identities
  Phase 6   possession.py    build_possession_log
  Phase 7   trajectory.py    detect_all_scoring_events
  Phase 8   scoreboard.py    locate_scoreboard, read_score (auto-detected)
  Phase 9   inference_engine build_score_timeline, compute_final_scores
  Phase 13  driving_analysis classify_all_robots, generate_driving_report
  Phase 14  alliance_builder build_team_composite_scores, recommend_picks
  Phase 10  count.py         aggregate_scores, generate_accuracy_report
  Phase 11  export.py + ui.py

Run:
    python main.py --video path/to/match.mp4 [--skip-ingest] [--skip-detect]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _net_ok(host: str = "api.statbotics.io", port: int = 443,
            timeout: float = 4.0) -> bool:
    """Return True if host is reachable (HTTP HEAD preferred; TCP socket fallback)."""
    # HTTP HEAD is more reliable than raw TCP when Windows Defender or a proxy
    # intercepts raw sockets but allows HTTPS traffic.
    try:
        import requests as _req
        scheme = "https" if port == 443 else "http"
        _req.head(f"{scheme}://{host}", timeout=timeout, allow_redirects=True)
        return True
    except Exception:
        pass
    # TCP socket fallback
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_field_config() -> dict:
    path = Path("configs/field_config.json")
    if not path.exists():
        raise FileNotFoundError("configs/field_config.json not found. Run Phase 1 first.")
    with open(path) as f:
        return json.load(f)


def _load_roboflow_config() -> dict:
    path = Path("configs/roboflow_config.json")
    if not path.exists():
        raise FileNotFoundError(
            "configs/roboflow_config.json not found.\n"
            "Create it with api_key, workspace, workflow_id."
        )
    with open(path) as f:
        return json.load(f)


def run_pipeline(
    video_path:   str,
    skip_ingest:  bool = False,
    skip_detect:  bool = False,
    sample_every: int  = 3,
    no_ui:        bool = False,
) -> None:
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    cfg = _load_field_config()
    Path("data/exports").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PHASE 2 - Video Metadata (no frame extraction — process_video reads directly)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 - Video Metadata")
    print("=" * 60)

    import cv2 as _cv2_meta
    _cap = _cv2_meta.VideoCapture(str(video_path))
    source_fps   = _cap.get(_cv2_meta.CAP_PROP_FPS) or 30.0
    total_frames = int(_cap.get(_cv2_meta.CAP_PROP_FRAME_COUNT))
    _w = int(_cap.get(_cv2_meta.CAP_PROP_FRAME_WIDTH))
    _h = int(_cap.get(_cv2_meta.CAP_PROP_FRAME_HEIGHT))
    _cap.release()
    duration   = total_frames / source_fps if source_fps > 0 else 0.0
    resolution = (_w, _h)

    meta = {
        "video_file": str(video_path),
        "total_frames": total_frames,
        "duration_seconds": round(duration, 2),
        "fps": round(source_fps, 3),
        "width": _w,
        "height": _h,
    }
    Path("data/metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  [Phase 2] {video_path.name} - {total_frames} frames, "
          f"{duration:.1f}s, {_w}×{_h} @ {source_fps:.1f} fps")

    # -------------------------------------------------------------------------
    # PHASE 5 - Detection (Roboflow API) + Tracking
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5 - Detection + Tracking")
    print("=" * 60)

    detect_cache = Path("data/detections.json")

    if not skip_detect:
        rf_cfg = _load_roboflow_config()
        api_key = rf_cfg.get("api_key", "")
        if not api_key or api_key == "YOUR_ROBOFLOW_KEY_HERE":
            print("  [Phase 5] (!) Roboflow API key not set in configs/roboflow_config.json")
            print("             Detection skipped - using empty track data.")
            all_frame_detections = []
        else:
            from detect import process_video, save_detection_cache
            all_frame_detections = process_video(
                video_path, rf_cfg,
                sample_every_n = sample_every,
                max_workers    = rf_cfg.get("max_workers", 4),
                save_annotated = False,
                project_root   = Path(__file__).parent,
            )
            save_detection_cache(all_frame_detections, detect_cache)
    else:
        if detect_cache.exists():
            from detect import load_detection_cache
            all_frame_detections = load_detection_cache(detect_cache)
            print(f"  [Phase 5] Loaded cached detections: "
                  f"{len(all_frame_detections)} frames")
        else:
            print("  [Phase 5] No detection cache found; skipping detection.")
            all_frame_detections = []

    # Tracking
    # IMPORTANT: robot tracker runs on ORIGINAL sampled frames only.
    # The in-flight interpolation adds ball-only frames; feeding those empty
    # frames to DeepSORT causes robot tracks to age out and get new IDs,
    # breaking the identity match vs. match_identity.json.
    from track import run_ball_tracker, run_robot_tracker

    robot_tracks = run_robot_tracker(all_frame_detections)

    # In-flight interpolation: fills gaps between sampled frames using
    # linear interpolation (fast, <1s).  Runs AFTER robot tracking so that
    # the 8k added frames don't disrupt robot track IDs.
    # Skip if sample_every == 1 (no gaps to fill) or if detection was skipped.
    if sample_every > 1 and all_frame_detections:
        from inflight_detector import interpolate_inflight_balls
        interpolate_inflight_balls(
            video_path           = video_path,
            all_frame_detections = all_frame_detections,
            sample_every_n       = sample_every,
            use_optical_flow     = False,
            verbose              = True,
        )

    ball_tracks = run_ball_tracker(all_frame_detections)

    # ── Robot identity: spatial-match from saved file, OCR only as fallback ──────
    from track import build_robot_identity_map
    from detect import read_bumper_number

    def _start_pos(entries):
        e = min(entries, key=lambda x: x["frame_id"])
        b = e["bbox"]
        return (b[0]+b[2])/2, (b[1]+b[3])/2

    current_pos = {tid: _start_pos(ents) for tid, ents in robot_tracks.items()}

    # Start with UNKNOWN placeholders for all current tracks
    robot_identity_map: dict[int, dict] = {
        tid: {"team_number": f"UNKNOWN_{tid}", "confidence": 0.0, "frames_confirmed": 0}
        for tid in robot_tracks
    }

    # Apply saved identity via spatial matching first (fast — no OCR needed)
    identity_path = Path("configs/match_identity.json")
    used: set[int] = set()
    if identity_path.exists():
        with open(identity_path) as f:
            saved_data = json.load(f)

        candidates = [r for r in saved_data.get("robots", [])
                      if r.get("user_corrected")
                      or r.get("method") == "tba_match_data"
                      or r.get("confidence", 0) >= 0.55]
        for saved in candidates:
            sx = saved.get("start_x")
            sy = saved.get("start_y")
            if sx is None or sy is None:
                tid = saved.get("track_id")
                if tid in robot_identity_map and tid not in used:
                    robot_identity_map[tid]["team_number"] = saved["team_number"]
                    robot_identity_map[tid]["confidence"]  = saved["confidence"]
                    robot_identity_map[tid]["method"]      = saved.get("method", "ocr")
                    used.add(tid)
                continue
            best_tid, best_dist = None, float("inf")
            for tid, (cx, cy) in current_pos.items():
                if tid in used:
                    continue
                d = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is not None and best_dist < 250:
                robot_identity_map[best_tid]["team_number"] = saved["team_number"]
                robot_identity_map[best_tid]["confidence"]  = saved["confidence"]
                robot_identity_map[best_tid]["user_corrected"] = saved.get("user_corrected", False)
                robot_identity_map[best_tid]["method"]      = saved.get("method", "ocr")
                used.add(best_tid)

        if used:
            print(f"  [Phase 5] Applied {len(used)} team numbers from saved identity map.")

    # OCR only for tracks that are still UNKNOWN after spatial matching
    unknown_tracks = {tid: ents for tid, ents in robot_tracks.items()
                      if tid not in used}
    if unknown_tracks:
        print(f"  [Phase 5] Running bumper OCR on {len(unknown_tracks)} unidentified tracks...")
        ocr_results = build_robot_identity_map(
            unknown_tracks, read_bumper_number, video_path,
            frame_sample=200,   # cap at 200 frames to keep OCR fast
        )
        for tid, info in ocr_results.items():
            robot_identity_map[tid] = info
    else:
        print("  [Phase 5] All tracks identified from saved identity - skipping OCR.")

    print(f"  [Phase 5] Final identity map: {len(robot_identity_map)} tracks")
    for tid, info in sorted(robot_identity_map.items()):
        print(f"    Track {tid} -> {info['team_number']} "
              f"(conf {info['confidence']:.0%}, "
              f"{info.get('frames_confirmed', 0)} frames)")

    # Inject team_number into robot track entries
    for track_id, entries in robot_tracks.items():
        team = robot_identity_map.get(track_id, {}).get("team_number",
                                                         f"UNKNOWN_{track_id}")
        for e in entries:
            e["team_number"] = team

    # Save match_identity.json with current track_ids + positions (for next run)
    import time as _time
    from track import detect_alliances

    # Build alliance map: prefer saved alliances over visual detection
    saved_alliances: dict[int, str] = {}
    if identity_path.exists():
        try:
            _sid = json.loads(identity_path.read_text())
            for _r in _sid.get("robots", []):
                _al = _r.get("alliance", "unknown")
                if _al in ("red", "blue"):
                    saved_alliances[_r["track_id"]] = _al
        except Exception:
            pass

    # Also infer alliance from team number -> known alliance (TBA data)
    team_to_alliance: dict[str, str] = {}
    for tid, info in robot_identity_map.items():
        tn = info.get("team_number", "")
        if tid in saved_alliances:
            team_to_alliance[tn] = saved_alliances[tid]

    # Fill any remaining unknowns via visual bumper-color detection
    unknown_alliance_tids = [tid for tid in robot_identity_map
                             if tid not in saved_alliances]
    if unknown_alliance_tids:
        visual = detect_alliances(robot_tracks, video_path, frame_sample=300)
        # saved wins; visual fills the gaps
        alliance_map = {**visual, **saved_alliances}
    else:
        alliance_map = saved_alliances

    def _first_pos(ents):
        e = min(ents, key=lambda x: x["frame_id"])
        b = e["bbox"]
        return round((b[0]+b[2])/2, 1), round((b[1]+b[3])/2, 1)

    robots_json = []
    for tid, info in sorted(robot_identity_map.items()):
        sx, sy = _first_pos(robot_tracks[tid]) if tid in robot_tracks else (0.0, 0.0)
        robots_json.append({
            "track_id":       tid,
            "team_number":    info["team_number"],
            "alliance":       alliance_map.get(tid, "unknown"),
            "confidence":     info["confidence"],
            "frames_confirmed": info.get("frames_confirmed", 0),
            "user_corrected": info.get("user_corrected", False),
            "method":         info.get("method", "ocr"),
            "start_x":        sx,
            "start_y":        sy,
        })
    Path("configs/match_identity.json").write_text(
        json.dumps({
            "video_file":         Path(video_path).name,
            "calibration_frames": 600,
            "calibrated_at":      _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "robots":             robots_json,
            "user_confirmed":     False,
        }, indent=2)
    )
    print(f"  [Phase 5] Saved match_identity.json ({len(robots_json)} robots)")

    # -------------------------------------------------------------------------
    # PHASE 6 - Possession Engine
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 6 - Possession Engine")
    print("=" * 60)

    from possession import build_possession_log, save_possession_log

    possession_log = build_possession_log(
        ball_tracks, robot_tracks, total_frames or 99999
    )
    save_possession_log(possession_log)

    # -------------------------------------------------------------------------
    # FIELD AUTO-CALIBRATION (scoring zones + scoreboard)
    # -------------------------------------------------------------------------
    from field_calibration import calibrate_field
    calibrate_field(
        video_path      = video_path,
        config_path     = "configs/field_config.json",
        detections_path = "data/detections.json",
        n_samples       = 40,
    )
    # Reload config with freshly detected zones
    cfg = _load_field_config()

    # -------------------------------------------------------------------------
    # PHASE 7 - Trajectory Engine
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 7 - Trajectory Engine")
    print("=" * 60)

    from trajectory import detect_all_scoring_events

    scoring_zones = {
        k: v for k, v in cfg.get("scoring_zones", {}).items()
        if not k.startswith("_")
    }
    all_scoring_events = detect_all_scoring_events(
        ball_tracks, scoring_zones, possession_log
    )

    # -------------------------------------------------------------------------
    # PHASE 8 - Scoreboard OCR
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 8 - Scoreboard OCR")
    print("=" * 60)

    import cv2
    from scoreboard import locate_scoreboard, read_score, detect_score_change

    sb_cfg      = cfg.get("scoreboard", {})
    sb_coords   = sb_cfg.get("bbox")   # set by calibrate_field(); None = detect per-frame

    score_history: list[dict]  = []
    latest_scoreboard: dict    = {"red_score": -1, "blue_score": -1}

    # Seek directly to scoreboard sample frames — avoids reading all ~4500 frames.
    # At 5-second intervals only ~30 seeks are needed for a 2:30 match.
    interval_frames = max(1, int(source_fps * 5))
    sample_fids = list(range(0, total_frames, interval_frames))
    cap = cv2.VideoCapture(str(video_path))
    print(f"  [Phase 8] Seeking to {len(sample_fids)} scoreboard frames "
          f"(every {interval_frames} frames = ~5s)...")

    for frame_idx in sample_fids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = locate_scoreboard(frame, sb_coords)
        if bbox:
            score = read_score(frame, bbox)
            if score["red_score"] >= 0 or score["blue_score"] >= 0:
                change = detect_score_change(score_history, score, frame_idx)
                score_entry = {**score, "frame_id": frame_idx}
                score_history.append(score_entry)
                latest_scoreboard = score
                if change["changed"]:
                    print(f"  [Phase 8] Frame {frame_idx:5d}  "
                          f"red={score['red_score']}  "
                          f"blue={score['blue_score']}  "
                          f"(+{change['delta']} {change['alliance']})")

    cap.release()

    if score_history:
        # Use TBA actual score if available — more reliable than OCR
        _tba_red, _tba_blue = None, None
        try:
            from tba_client import get_event_matches as _get_matches
            import json as _jj2
            _tba_cfg_path = Path("configs/tba_config.json")
            _tba_event = _jj2.loads(_tba_cfg_path.read_text()).get("event_key", "") if _tba_cfg_path.exists() else ""
            if _tba_event:
                _tba_match_num = None
                _meta_path = Path("data/metadata.json")
                if _meta_path.exists():
                    _meta = _jj2.loads(_meta_path.read_text())
                    _vname = Path(_meta.get("video_file", "")).stem.lower()
                    # Try to parse match number from filename (e.g. "Qualification 1")
                    import re as _re
                    _mn = _re.search(r'qualif\w+\s+(\d+)', _vname, _re.IGNORECASE)
                    if _mn:
                        _tba_match_num = int(_mn.group(1))
                if _tba_match_num:
                    _matches = _get_matches(_tba_event)
                    for _m in _matches:
                        if (_m.get("comp_level") == "qm" and
                                _m.get("match_number") == _tba_match_num):
                            _al = _m.get("alliances", {})
                            _tba_red  = _al.get("red",  {}).get("score")
                            _tba_blue = _al.get("blue", {}).get("score")
                            break
        except Exception:
            pass

        if _tba_red is not None and _tba_blue is not None and _tba_red >= 0:
            latest_scoreboard = {
                "red_score":  _tba_red,
                "blue_score": _tba_blue,
                "confidence": 1.0,
                "source":     "tba",
            }
            print(f"  [Phase 8] Final score (from TBA): "
                  f"RED={_tba_red}  BLUE={_tba_blue}")
        else:
            # Fall back to OCR: latest high-confidence reading from last 20%
            # Use latest (highest frame_id) rather than highest-confidence to
            # avoid selecting a mid-match reading when a later reading exists.
            cutoff    = int(len(score_history) * 0.80)
            tail      = score_history[cutoff:] if cutoff < len(score_history) else score_history
            high_conf = [s for s in tail if s.get("confidence", 0) >= 0.70]
            latest_scoreboard = (
                max(high_conf, key=lambda s: s.get("frame_id", 0))
                if high_conf
                else max(score_history, key=lambda s: s.get("frame_id", 0))
            )
            print(f"  [Phase 8] Final score (OCR): "
                  f"RED={latest_scoreboard['red_score']}  "
                  f"BLUE={latest_scoreboard['blue_score']}"
                  f"  (conf={latest_scoreboard.get('confidence', 0):.2f})")
        print(f"  [Phase 8] Final score: "
              f"RED={latest_scoreboard['red_score']}  "
              f"BLUE={latest_scoreboard['blue_score']}"
              f"  (conf={latest_scoreboard.get('confidence', 0):.2f})")
    else:
        print("  [Phase 8] (!) No scoreboard readings obtained.")

    with open("data/score_history.json", "w") as f:
        json.dump(score_history, f, indent=2)

    # -------------------------------------------------------------------------
    # PHASE 9 - Attribution Engine
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 9 - Attribution Engine")
    print("=" * 60)

    from inference_engine import (
        build_score_timeline, compute_final_scores, save_score_timeline
    )
    from possession import detect_ball_exits

    # Detect when each ball left a robot (used for exit-count reconciliation)
    ball_exits = detect_ball_exits(ball_tracks, possession_log)

    # Fetch OPR weights from TBA + Statbotics for all robots in this match
    opr_map: dict[str, float] = {}
    all_team_numbers = [
        str(info.get("team_number", ""))
        for info in robot_identity_map.values()
        if info.get("team_number")
    ]

    # Load event_key and TBA API key once
    tba_cfg_path = Path("configs/tba_config.json")
    event_key    = None
    tba_api_key  = None
    if tba_cfg_path.exists():
        _tba_cfg   = json.loads(tba_cfg_path.read_text())
        event_key  = _tba_cfg.get("event_key")
        tba_api_key = _tba_cfg.get("api_key")

    tba_oprs: dict[str, float] = {}
    _tba_online = _net_ok("www.thebluealliance.com")
    _sb_online  = _net_ok("api.statbotics.io")

    if not _tba_online:
        print("  [Phase 9] TBA unreachable - skipping OPR fetch.")
    elif event_key and tba_api_key and tba_api_key != "YOUR_TBA_KEY_HERE":
        try:
            from tba_client import get_event_oprs
            raw_oprs = get_event_oprs(event_key)
            tba_oprs = {
                k.replace("frc", ""): float(v)
                for k, v in raw_oprs.get("oprs", {}).items()
            }
            print(f"  [Phase 9] TBA OPRs loaded: {len(tba_oprs)} teams "
                  f"at event {event_key}")
        except Exception as exc:
            print(f"  [Phase 9] TBA OPR fetch failed ({exc}) - "
                  f"continuing without TBA OPR")

    if not _sb_online:
        print("  [Phase 9] Statbotics unreachable - using TBA OPR / default weights.")
        opr_map = {t: tba_oprs.get(t, 1.0) for t in all_team_numbers}
    elif all_team_numbers:
        try:
            from statbotics_client import build_opr_map
            opr_map = build_opr_map(
                all_team_numbers,
                event_key = event_key,
                tba_oprs  = tba_oprs if tba_oprs else None,
            )
        except Exception as exc:
            print(f"  [Phase 9] Statbotics unavailable ({exc}) - "
                  f"falling back to TBA OPR only")
            opr_map = {t: tba_oprs.get(t, 1.0) for t in all_team_numbers}

    score_timeline = build_score_timeline(
        all_scoring_events, possession_log,
        robot_tracks, ball_tracks, scoring_zones,
        source_fps     = source_fps,
        identity_map   = robot_identity_map,
        score_history  = score_history,
        ball_exits     = ball_exits,
        opr_map        = opr_map,
    )
    final_scores = compute_final_scores(score_timeline)
    save_score_timeline(score_timeline)

    # -------------------------------------------------------------------------
    # PHASE 13 - Driving Style Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 13 - Driving Style Analysis")
    print("=" * 60)

    from driving_analysis import classify_all_robots, generate_driving_report

    alliance_zones = {
        k: v for k, v in cfg.get("alliance_zones", {}).items()
        if not k.startswith("_")
    }

    all_classifications = classify_all_robots(
        robot_identity_map = robot_identity_map,
        robot_tracks       = robot_tracks,
        all_robot_tracks   = robot_tracks,
        score_timeline     = score_timeline,
        alliance_zones     = alliance_zones,
    )

    # Metrics are already computed inside classify_all_robots — extract them
    # directly instead of recomputing (which would hit the slow O(N³) path again).
    all_metrics: dict[str, dict] = {
        team: data["metrics"]
        for team, data in all_classifications.items()
        if "metrics" in data
    }

    driving_report = generate_driving_report(all_classifications, all_metrics)
    print("  Driving analysis complete.")

    # -------------------------------------------------------------------------
    # PHASE 14 - TBA Alliance Builder
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 14 - TBA Alliance Builder")
    print("=" * 60)

    tba_cfg_path = Path("configs/tba_config.json")
    pick_list   = []
    picks       = {}
    do_not_pick = []

    if not tba_cfg_path.exists():
        print("  [Phase 14] configs/tba_config.json not found - skipping.")
    else:
        tba_cfg = json.loads(tba_cfg_path.read_text())
        api_key   = tba_cfg.get("api_key",          "")
        event_key = tba_cfg.get("event_key",         "")
        our_team  = tba_cfg.get("our_team_number",   "")

        if api_key == "YOUR_TBA_KEY_HERE" or not api_key:
            print("  INPUT CHECKPOINT #8: Set TBA API key in configs/tba_config.json")
        elif not _net_ok("www.thebluealliance.com"):
            print("  [Phase 14] TBA unreachable (offline?) - skipping alliance builder.")
        else:
            def _run_alliance_builder():
                from alliance_builder import (
                    build_team_composite_scores, generate_pick_list,
                    recommend_picks, recommend_do_not_pick,
                )
                strategy   = tba_cfg.get("strategy", "balanced")
                composites = build_team_composite_scores(
                    event_key, our_team,
                    {"final_scores": final_scores, "score_timeline": score_timeline},
                    driving_report,
                )
                pl  = generate_pick_list(our_team, event_key, composites)
                pk  = recommend_picks(our_team, event_key, composites, strategy)
                dnp = recommend_do_not_pick(composites)
                return composites, pl, pk, dnp

            try:
                with ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(_run_alliance_builder)
                    composites, pick_list, picks, do_not_pick = _fut.result(timeout=45)
                print(f"  Alliance builder complete. Pick list: {len(pick_list)} teams.")
            except FutureTimeout:
                print("  [Phase 14] Alliance builder timed out (>45s) - "
                      "check network or TBA event key and retry from Alliance tab.")
            except Exception as exc:
                print(f"  [Phase 14] Alliance builder failed: {exc}")

    # -------------------------------------------------------------------------
    # PHASE 10 - Score Aggregation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 10 - Score Aggregation")
    print("=" * 60)

    from count import aggregate_scores, generate_accuracy_report, save_accuracy_report
    from scoreboard import validate_attribution

    aggregated = aggregate_scores(score_timeline, robot_identity_map)

    robot_score_totals = {t: v["score"] for t, v in aggregated.items()
                          if t != "UNATTRIBUTED"}
    sb_validation = validate_attribution(robot_score_totals, latest_scoreboard)

    report = generate_accuracy_report(aggregated, sb_validation)
    save_accuracy_report(report)

    # -------------------------------------------------------------------------
    # PHASE 11 - Export
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 11 - Export")
    print("=" * 60)

    from export import (
        export_csv, export_json,
        export_driving_report_csv, export_driving_report_json,
    )

    export_csv(final_scores, score_timeline, "data/exports/scores.csv")
    export_json(
        {
            "final_scores":    final_scores,
            "score_timeline":  score_timeline,
            "accuracy_report": report,
        },
        "data/exports/results.json",
    )
    export_driving_report_csv(driving_report, "data/exports/driving_report.csv")
    export_driving_report_json(driving_report, "data/exports/driving_report.json")

    if pick_list:
        from export import export_pick_list_csv, export_pick_list_json
        export_pick_list_csv(pick_list, "data/exports/pick_list.csv")
        export_pick_list_json(
            {"picks": picks, "pick_list": pick_list, "do_not_pick": do_not_pick},
            "data/exports/pick_list.json",
        )

    print("\n  Exports written to data/exports/")

    # -------------------------------------------------------------------------
    # PHASE 11 - UI
    # -------------------------------------------------------------------------
    if not no_ui:
        from ui import set_state, launch
        set_state(
            final_scores    = final_scores,
            score_timeline  = score_timeline,
            driving_report  = driving_report,
            video_path      = str(video_path),
            pick_list       = pick_list,
            picks           = picks,
            do_not_pick     = do_not_pick,
        )
        print("\nLaunching Gradio UI at http://localhost:7860 ...")
        launch()
    else:
        print("\nPipeline complete. Run with --ui to launch the Gradio interface.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FRC 2026 Ball Counter - full analysis pipeline"
    )
    parser.add_argument("--video",             required=True,
                        help="Path to match video (.mp4/.mov/.avi)")
    parser.add_argument("--skip-ingest",       action="store_true",
                        help="Skip Phase 2 frame extraction (use cached frames)")
    parser.add_argument("--skip-detect",       action="store_true",
                        help="Skip Phase 5 detection (use cached detections.json)")
    parser.add_argument("--sample",            type=int, default=3,
                        help="Process every N-th frame via Roboflow (default: 3)")
    parser.add_argument("--no-ui",             action="store_true",
                        help="Skip launching Gradio UI after pipeline completes")
    # In-flight training data generation
    parser.add_argument("--extract-inflight",  action="store_true",
                        help="Extract in-flight ball frames for model training, then exit")
    parser.add_argument("--train-inflight",    action="store_true",
                        help="Train local in-flight ball model from extracted data, then exit")
    parser.add_argument("--inflight-epochs",   type=int, default=50,
                        help="Epochs for --train-inflight (default: 50)")
    args = parser.parse_args()

    # Standalone: extract in-flight training frames and exit
    if args.extract_inflight:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from extract_inflight_frames import extract_inflight_frames
        extract_inflight_frames(
            video_path  = args.video,
            scan_sample = 1,
            save_review = True,
        )
        print("INFLIGHT EXTRACTION COMPLETE. Review data/inflight_training/review/")
        print("Then run: py -3.13 main.py --video <path> --train-inflight")
        return

    # Standalone: train in-flight model from extracted data and exit
    if args.train_inflight:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from extract_inflight_frames import train_inflight_model
        train_inflight_model(epochs=args.inflight_epochs)
        print("INFLIGHT MODEL TRAINED. Re-run pipeline without --skip-detect.")
        return

    run_pipeline(
        video_path   = args.video,
        skip_ingest  = args.skip_ingest,
        skip_detect  = args.skip_detect,
        sample_every = args.sample,
        no_ui        = args.no_ui,
    )


if __name__ == "__main__":
    main()
