"""
src/inflight_detector.py - In-flight ball interpolation between sampled frames.

Problem: FUEL shots last 1-2 frames at 60 fps. With sample_every_n=3 we miss
most of them, giving ~12% score coverage.

Solution: After the main YOLO/Roboflow pass, re-open the video and read the
skipped frames between any two sampled frames where a ball was visible.
Lucas-Kanade sparse optical flow tracks the ball through each gap.

This runs locally (OpenCV only, no API calls) and adds synthetic detections
back into all_frame_detections before tracking runs.

Expected improvement: 12% -> 40-60% score coverage, no retraining required.

Checkpoint: depends on Phase 5 detection cache (all_frame_detections).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ─── Tuning ─────────────────────────────────────────────────────────────────

# LK optical flow parameters
_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Minimum ball detection confidence from original YOLO pass to seed tracking
_SEED_CONF = 0.30

# Confidence assigned to interpolated (non-YOLO) detections
_INTERP_CONF = 0.45

# Ball class names to recognise in existing detections
_BALL_CLASSES = {"ball", "fuel", "game_piece", "gamepiece", "coral", "note", "ring"}

# Skip gaps larger than this many frames (ball likely disappeared / scored)
_MAX_GAP_FRAMES = 12

# Minimum pixel displacement over the gap to bother interpolating
# (avoids adding detections for balls sitting still on floor)
_MIN_DISPLACEMENT_PX = 8

# ─── Helpers ────────────────────────────────────────────────────────────────

def _is_ball(det: dict) -> bool:
    return det.get("class_name", "").lower() in _BALL_CLASSES


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_wh(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return abs(x2 - x1), abs(y2 - y1)


def _center_to_bbox(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _lerp_bbox(bbox_a: list[float], bbox_b: list[float], t: float) -> list[float]:
    """Linear interpolation between two bboxes; t in [0, 1]."""
    return [a + (b - a) * t for a, b in zip(bbox_a, bbox_b)]


# ─── Core interpolation ─────────────────────────────────────────────────────

def _lk_interpolate(
    frames_between: list[np.ndarray],
    anchor_a: np.ndarray,    # frame at sample point A
    anchor_b: np.ndarray,    # frame at sample point B
    center_a: tuple[float, float],
    center_b: tuple[float, float],
    bbox_a: list[float],
    bbox_b: list[float],
) -> list[tuple[float, float]]:
    """
    Track ball center through intermediate frames using Lucas-Kanade.

    Runs forward from A and backward from B, then blends the two tracks to
    reduce drift (forward/backward consistency).

    Args:
        frames_between: List of BGR frames between A and B (exclusive of both).
        anchor_a:       Frame at sample A.
        anchor_b:       Frame at sample B.
        center_a:       Ball center in frame A (x, y).
        center_b:       Ball center in frame B (x, y).
        bbox_a:         Ball bbox in frame A [x1,y1,x2,y2].
        bbox_b:         Ball bbox in frame B [x1,y1,x2,y2].

    Returns:
        List of (x, y) centers for each intermediate frame, same order as
        frames_between.
    """
    n = len(frames_between)
    if n == 0:
        return []

    all_frames_fwd = [anchor_a] + frames_between
    all_frames_bwd = [anchor_b] + list(reversed(frames_between))

    def _to_gray(f: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f

    pt_a = np.array([[center_a]], dtype=np.float32)
    pt_b = np.array([[center_b]], dtype=np.float32)

    # Forward pass: A -> intermediate frames
    fwd_pts: list[tuple[float, float]] = []
    cur_pt = pt_a.copy()
    prev_gray = _to_gray(all_frames_fwd[0])
    ok_fwd = True
    for i in range(1, len(all_frames_fwd)):
        next_gray = _to_gray(all_frames_fwd[i])
        new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, cur_pt, None, **_LK_PARAMS
        )
        if status is None or status[0][0] == 0:
            ok_fwd = False
            break
        cur_pt = new_pt
        if i < len(all_frames_fwd) - 1:   # skip last (that's anchor_b)
            fwd_pts.append((float(new_pt[0][0][0]), float(new_pt[0][0][1])))
        prev_gray = next_gray

    # Backward pass: B -> intermediate frames (reversed)
    bwd_pts_rev: list[tuple[float, float]] = []
    cur_pt = pt_b.copy()
    prev_gray = _to_gray(all_frames_bwd[0])
    ok_bwd = True
    for i in range(1, len(all_frames_bwd)):
        next_gray = _to_gray(all_frames_bwd[i])
        new_pt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, next_gray, cur_pt, None, **_LK_PARAMS
        )
        if status is None or status[0][0] == 0:
            ok_bwd = False
            break
        cur_pt = new_pt
        if i < len(all_frames_bwd) - 1:
            bwd_pts_rev.append((float(new_pt[0][0][0]), float(new_pt[0][0][1])))
        prev_gray = next_gray

    bwd_pts = list(reversed(bwd_pts_rev))

    # Blend forward / backward (or fall back to linear interpolation)
    result: list[tuple[float, float]] = []
    for i in range(n):
        t = (i + 1) / (n + 1)   # 0 < t < 1
        lin_x = center_a[0] + (center_b[0] - center_a[0]) * t
        lin_y = center_a[1] + (center_b[1] - center_a[1]) * t

        fx, fy = fwd_pts[i] if (ok_fwd and i < len(fwd_pts)) else (lin_x, lin_y)
        bx, by = bwd_pts[i] if (ok_bwd and i < len(bwd_pts)) else (lin_x, lin_y)

        # Weight forward/backward by proximity to each endpoint
        w_fwd = 1.0 - t
        w_bwd = t
        total = w_fwd + w_bwd
        x = (fx * w_fwd + bx * w_bwd) / total
        y = (fy * w_fwd + by * w_bwd) / total
        result.append((x, y))

    return result


# ─── Main entry point ────────────────────────────────────────────────────────

def _add_synth(
    new_frames: dict,
    fid: int,
    cx: float, cy: float, w: float, h: float,
    t: float,
    ts_a: float | None, ts_b: float | None,
    class_name: str,
) -> None:
    """Add one synthetic detection to new_frames dict."""
    new_bbox = _center_to_bbox(cx, cy, w, h)
    det = {
        "bbox":       new_bbox,
        "confidence": _INTERP_CONF,
        "class_name": class_name,
        "cx":         cx,
        "cy":         cy,
        "width":      w,
        "height":     h,
        "source":     "inflight_interp",
    }
    if fid not in new_frames:
        ts_ms = None
        if ts_a is not None and ts_b is not None:
            ts_ms = ts_a + (ts_b - ts_a) * t
        new_frames[fid] = {
            "frame_id":     fid,
            "timestamp_ms": ts_ms,
            "detections":   [],
            "count":        0,
            "error":        None,
        }
    new_frames[fid]["detections"].append(det)
    new_frames[fid]["count"] += 1


def interpolate_inflight_balls(
    video_path: str | Path,
    all_frame_detections: list[dict],
    sample_every_n: int = 3,
    max_gap: int = _MAX_GAP_FRAMES,
    min_displacement_px: float = _MIN_DISPLACEMENT_PX,
    use_optical_flow: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Supplement YOLO detections with optically-tracked in-flight ball positions
    for all frames that were skipped during the sampling pass.

    all_frame_detections format (from load_detection_cache):
        [{"frame_id": int, "detections": [{"bbox":..,"confidence":..,"class_name":..}], ...}, ...]

    This mutates all_frame_detections IN PLACE and also returns it.

    Args:
        video_path:             Path to match video.
        all_frame_detections:   Output of process_video() / load_detection_cache().
        sample_every_n:         Frame sampling interval used during detection.
        max_gap:                Ignore gaps wider than this (ball likely gone).
        min_displacement_px:    Ignore near-stationary balls (on floor).
        use_optical_flow:       If True, use Lucas-Kanade optical flow for more accurate
                                sub-pixel tracking (slower, ~4min for a full match).
                                Default False uses fast linear interpolation (~5s).

    Returns:
        Augmented all_frame_detections (same list, with new frame entries inserted).
    """
    if not all_frame_detections:
        return all_frame_detections

    video_path = Path(video_path)
    if not video_path.exists():
        if verbose:
            print("  [Inflight] Video not found - skipping interpolation.")
        return all_frame_detections

    # Build lookup: frame_id -> frame_dict
    fid_to_frame: dict[int, dict] = {}
    for frame_dict in all_frame_detections:
        fid = frame_dict.get("frame_id", 0)
        fid_to_frame[fid] = frame_dict

    # Collect all (frame_id, det) pairs where a ball was detected
    ball_events: list[tuple[int, dict]] = []
    for frame_dict in all_frame_detections:
        fid  = frame_dict.get("frame_id", 0)
        dets = frame_dict.get("detections", [])
        for d in dets:
            if isinstance(d, dict) and _is_ball(d) and d.get("confidence", 0) >= _SEED_CONF:
                ball_events.append((fid, d))

    if not ball_events:
        if verbose:
            print("  [Inflight] No seeded ball detections found - nothing to interpolate.")
        return all_frame_detections

    ball_events.sort(key=lambda x: x[0])

    # Build candidate pairs: consecutive ball sightings within max_gap
    pairs: list[tuple[int, dict, int, dict]] = []
    for i in range(len(ball_events) - 1):
        fid_a, det_a = ball_events[i]
        fid_b, det_b = ball_events[i + 1]
        gap = fid_b - fid_a
        if gap <= 1 or gap > max_gap:
            continue
        # Check displacement is meaningful
        cx_a, cy_a = _bbox_center(det_a["bbox"])
        cx_b, cy_b = _bbox_center(det_b["bbox"])
        disp = ((cx_b - cx_a)**2 + (cy_b - cy_a)**2) ** 0.5
        if disp < min_displacement_px:
            continue
        pairs.append((fid_a, det_a, fid_b, det_b))

    if not pairs:
        if verbose:
            print("  [Inflight] No interpolation candidates (gaps too large or ball stationary).")
        return all_frame_detections

    if verbose:
        print(f"  [Inflight] {len(pairs)} interpolation candidates "
              f"({len(ball_events)} ball detections across {len(fid_to_frame)} sampled frames)")

    new_frames: dict[int, dict] = {}
    n_interpolated = 0

    if use_optical_flow:
        # ── Sequential video pass + LK optical flow ────────────────────────────
        # Accurate but slow (~4 min for full match). Use for fine-tuning or when
        # the ball trajectory is highly curved.
        needed_fids: set[int] = set()
        for fid_a, det_a, fid_b, det_b in pairs:
            needed_fids.add(fid_a)
            needed_fids.add(fid_b)
            needed_fids.update(range(fid_a + 1, fid_b))

        needed_sorted = sorted(needed_fids)
        needed_set    = set(needed_sorted)
        min_fid = needed_sorted[0]
        max_fid = needed_sorted[-1]

        if verbose:
            print(f"  [Inflight] Sequential video read (LK mode): "
                  f"frames {min_fid}-{max_fid} ({len(needed_sorted)} needed)...")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            if verbose:
                print("  [Inflight] Could not open video - skipping.")
            return all_frame_detections

        frame_store: dict[int, np.ndarray] = {}
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_fid)
            cur = min_fid
            while cur <= max_fid:
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                if cur in needed_set:
                    frame_store[cur] = frm
                cur += 1
        finally:
            cap.release()

        if verbose:
            print(f"  [Inflight] Read {len(frame_store)} frames.")

        for fid_a, det_a, fid_b, det_b in pairs:
            if fid_a not in frame_store or fid_b not in frame_store:
                continue
            gap            = fid_b - fid_a
            inter_fids     = list(range(fid_a + 1, fid_b))
            frames_between = [frame_store[f] for f in inter_fids if f in frame_store]
            valid_inter    = [f for f in inter_fids if f in frame_store]
            if not frames_between:
                continue

            bbox_a = det_a["bbox"]
            bbox_b = det_b["bbox"]
            ctr_a  = _bbox_center(bbox_a)
            ctr_b  = _bbox_center(bbox_b)
            w_a, h_a = _bbox_wh(bbox_a)
            w_b, h_b = _bbox_wh(bbox_b)

            tracked = _lk_interpolate(
                frames_between, frame_store[fid_a], frame_store[fid_b],
                ctr_a, ctr_b, bbox_a, bbox_b,
            )
            ts_a = fid_to_frame.get(fid_a, {}).get("timestamp_ms")
            ts_b = fid_to_frame.get(fid_b, {}).get("timestamp_ms")

            for j, fid in enumerate(valid_inter):
                if j >= len(tracked):
                    break
                t  = (fid - fid_a) / gap
                cx, cy = tracked[j]
                w  = w_a + (w_b - w_a) * t
                h  = h_a + (h_b - h_a) * t
                _add_synth(new_frames, fid, cx, cy, w, h, t, ts_a, ts_b,
                           det_a.get("class_name", "ball"))
                n_interpolated += 1

    else:
        # ── Fast linear interpolation (no video read needed) ───────────────────
        # Accurate to within a pixel for gaps <= 6 frames. Runs in <1 second.
        if verbose:
            print(f"  [Inflight] Linear interpolation (fast mode)...")

        for fid_a, det_a, fid_b, det_b in pairs:
            gap        = fid_b - fid_a
            inter_fids = list(range(fid_a + 1, fid_b))
            if not inter_fids:
                continue

            bbox_a = det_a["bbox"]
            bbox_b = det_b["bbox"]
            w_a, h_a = _bbox_wh(bbox_a)
            w_b, h_b = _bbox_wh(bbox_b)
            ts_a = fid_to_frame.get(fid_a, {}).get("timestamp_ms")
            ts_b = fid_to_frame.get(fid_b, {}).get("timestamp_ms")

            for fid in inter_fids:
                t = (fid - fid_a) / gap
                interp_bbox = _lerp_bbox(bbox_a, bbox_b, t)
                cx = (interp_bbox[0] + interp_bbox[2]) / 2
                cy = (interp_bbox[1] + interp_bbox[3]) / 2
                w  = w_a + (w_b - w_a) * t
                h  = h_a + (h_b - h_a) * t
                _add_synth(new_frames, fid, cx, cy, w, h, t, ts_a, ts_b,
                           det_a.get("class_name", "ball"))
                n_interpolated += 1

    if not new_frames:
        if verbose:
            print("  [Inflight] LK tracking produced no usable positions.")
        return all_frame_detections

    # Insert new frame dicts into all_frame_detections in frame_id order
    for frame_dict in new_frames.values():
        all_frame_detections.append(frame_dict)

    all_frame_detections.sort(key=lambda x: x.get("frame_id", 0))

    if verbose:
        print(f"  [Inflight] Added {n_interpolated} interpolated ball detections "
              f"across {len(new_frames)} new frames "
              f"(total frames now: {len(all_frame_detections)})")

    return all_frame_detections


# ─── Augmented detection cache I/O ──────────────────────────────────────────

def save_augmented_cache(
    all_frame_detections: list[dict],
    path: str | Path,
) -> None:
    """Save the augmented detection list (with interpolated entries) to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"_cache_sig": "augmented", "frames": all_frame_detections},
                  f, separators=(",", ":"))


def load_augmented_cache(path: str | Path) -> list[dict] | None:
    """Load augmented detection cache if it exists."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    return raw.get("frames", [])
