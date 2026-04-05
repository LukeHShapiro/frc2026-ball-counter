"""
src/field_calibration.py - Automatic field zone detection

auto_detect_scoring_zones() samples frames from the match video and
locates scoring zones using two complementary signals:

  Signal A — Ball disappearance clustering
    Balls that exist in frame N but vanish by frame N+2 were likely scored.
    Cluster the disappearance locations -> zone centres.

  Signal B — Structural edge analysis
    FRC goals are elevated rectangular structures on the alliance walls
    (far left and far right of the field interior). Detect them via
    vertical edge density on the left and right thirds of the frame.

  Signal C — Cached detection heatmap (if detections.json exists)
    Uses saved Roboflow detections to build a density map of ball positions.
    High-density, stationary regions that are NOT near robots = scoring zones.

Result is written directly to configs/field_config.json so the pipeline
picks it up automatically. No user input required.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


# ---- helpers ----------------------------------------------------------------

def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _cluster_points(points: list[tuple[float, float]],
                    radius: float = 80.0) -> list[tuple[float, float]]:
    """
    Simple radius-based clustering. Returns list of cluster centroids.
    """
    if not points:
        return []
    clusters: list[list[tuple[float, float]]] = []
    for p in points:
        placed = False
        for c in clusters:
            cx = sum(q[0] for q in c) / len(c)
            cy = sum(q[1] for q in c) / len(c)
            if ((p[0]-cx)**2 + (p[1]-cy)**2) ** 0.5 < radius:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return [(sum(q[0] for q in c)/len(c),
             sum(q[1] for q in c)/len(c)) for c in clusters]


def _centre_to_bbox(cx: float, cy: float,
                    w: float, h: float,
                    frame_w: int, frame_h: int) -> list[int]:
    """Expand a centre point to a bbox, clamped to frame bounds."""
    x1 = max(0, int(cx - w/2))
    y1 = max(0, int(cy - h/2))
    x2 = min(frame_w, int(cx + w/2))
    y2 = min(frame_h, int(cy + h/2))
    return [x1, y1, x2, y2]


# ---- Signal A: ball disappearance -------------------------------------------

def _disappearance_signal(video_path: Path,
                           n_samples: int,
                           frame_w: int, frame_h: int) -> list[tuple[float, float]]:
    """
    Find positions where balls disappear between sampled frames.
    Uses background subtraction and motion detection on a small thumbnail.
    """
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // n_samples)

    # Collect ball-like blob centres across frames
    prev_centres: list[tuple[float, float]] = []
    disappeared: list[tuple[float, float]]  = []

    # Scale factor from thumbnail (320x?) to full frame
    thumb_w = 320
    thumb_h = int(frame_h * (thumb_w / frame_w))
    sx = frame_w / thumb_w
    sy = frame_h / thumb_h

    def _find_ball_blobs(frame_bgr: np.ndarray) -> list[tuple[float, float]]:
        """Find yellow/green ball-like blobs in a thumbnail frame."""
        thumb = cv2.resize(frame_bgr, (thumb_w, thumb_h))
        hsv   = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
        # Yellow balls (FRC 2026 Reefscape uses yellow/green game pieces)
        mask_y = cv2.inRange(hsv, (18, 80, 80), (38, 255, 255))
        # Green tint variant
        mask_g = cv2.inRange(hsv, (38, 60, 80), (80, 255, 255))
        mask   = cv2.bitwise_or(mask_y, mask_g)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                   np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        centres = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = (M["m10"] / M["m00"]) * sx
            cy = (M["m01"] / M["m00"]) * sy
            centres.append((cx, cy))
        return centres

    prev_centres = []
    for i in range(n_samples):
        fid = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        curr_centres = _find_ball_blobs(frame)

        # Which prev centres have no nearby match in curr? -> disappeared
        for pc in prev_centres:
            found = False
            for cc in curr_centres:
                if ((pc[0]-cc[0])**2 + (pc[1]-cc[1])**2)**0.5 < 40*sx:
                    found = True
                    break
            if not found:
                disappeared.append(pc)

        prev_centres = curr_centres

    cap.release()
    return disappeared


# ---- Signal B: structural vertical edges ------------------------------------

def _structural_signal(video_path: Path,
                        frame_w: int, frame_h: int) -> list[tuple[float, float]]:
    """
    Find elevated rectangular structures (goals) via vertical edge density
    in the left-third and right-third of the frame.
    Returns approximate centre points of detected goal structures.
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES,
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.15))  # 15% into match
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return []

    # Work at 1/4 resolution
    small = cv2.resize(frame, (frame_w // 4, frame_h // 4))
    sw, sh = frame_w // 4, frame_h // 4
    sx = 4.0

    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    # Vertical Sobel to emphasise vertical edges (goal posts)
    sobel_v = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_v = np.abs(sobel_v).astype(np.uint8)

    # Scan left third and right third for strong vertical edge columns
    # Field interior occupies roughly y: 15%-75% of frame height
    field_top    = int(sh * 0.15)
    field_bottom = int(sh * 0.75)
    field_crop   = sobel_v[field_top:field_bottom, :]

    col_sums = field_crop.mean(axis=0)

    centres = []

    # Left goal: peak vertical edge density in left 30% of frame
    left_sums  = col_sums[:int(sw * 0.30)]
    right_sums = col_sums[int(sw * 0.70):]

    if len(left_sums) > 0:
        peak_col_l = int(np.argmax(left_sums))
        peak_row_l = field_top + (field_bottom - field_top) // 3
        centres.append((peak_col_l * sx, peak_row_l * sx))

    if len(right_sums) > 0:
        peak_col_r = int(sw * 0.70) + int(np.argmax(right_sums))
        peak_row_r = field_top + (field_bottom - field_top) // 3
        centres.append((peak_col_r * sx, peak_row_r * sx))

    return centres


# ---- Signal C: detection heatmap --------------------------------------------

def _detection_heatmap_signal(detections_path: Path,
                               frame_w: int,
                               frame_h: int) -> list[tuple[float, float]]:
    """
    Build a heatmap from cached Roboflow detections.
    High-density regions far from robots = likely scoring zones.
    """
    if not detections_path.exists():
        return []

    with open(detections_path) as f:
        raw = json.load(f)

    # detections.json may be a bare list OR a cache dict {"frames": [...], ...}
    frames = raw.get("frames", raw) if isinstance(raw, dict) else raw

    ball_classes  = {"Fuel", "fuel", "ball", "Ball", "game_piece"}
    robot_classes = {"robot", "Robot", "Blue_Robot", "Red_Robot",
                     "blue_robot", "red_robot"}

    heat   = np.zeros((frame_h // 8, frame_w // 8), dtype=np.float32)
    sx, sy = frame_w / 8, frame_h / 8

    robot_positions: list[tuple[float, float]] = []

    for frame_data in frames:
        for det in frame_data.get("detections", []):
            cx, cy = det.get("cx", 0), det.get("cy", 0)
            if det["class_name"] in ball_classes:
                xi = min(int(cx / sx), heat.shape[1] - 1)
                yi = min(int(cy / sy), heat.shape[0] - 1)
                heat[yi, xi] += 1.0
            elif det["class_name"] in robot_classes:
                robot_positions.append((cx, cy))

    if heat.max() == 0:
        return []

    # Blur to create smooth density map
    heat_blur = cv2.GaussianBlur(heat, (15, 15), 0)

    # Zero out robot-adjacent regions (balls near robots = possession, not scoring)
    robot_mask = np.ones_like(heat_blur)
    for rx, ry in robot_positions:
        xi = min(int(rx / sx), heat.shape[1] - 1)
        yi = min(int(ry / sy), heat.shape[0] - 1)
        r  = int(150 / max(sx, sy))
        cv2.circle(robot_mask, (xi, yi), r, 0, -1)
    heat_blur *= robot_mask

    if heat_blur.max() == 0:
        return []

    # Find peaks
    threshold = heat_blur.max() * 0.50
    peak_mask = (heat_blur > threshold).astype(np.uint8) * 255
    peak_mask = cv2.morphologyEx(peak_mask, cv2.MORPH_OPEN,
                                  np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(peak_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    centres = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = (M["m10"] / M["m00"]) * sx
        cy = (M["m01"] / M["m00"]) * sy
        centres.append((cx, cy))

    return centres


# ---- Main auto-detection entry point ----------------------------------------

def auto_detect_scoring_zones(
    video_path:       str | Path,
    config_path:      str | Path = "configs/field_config.json",
    detections_path:  str | Path = "data/detections.json",
    n_samples:        int = 40,
    zone_box_w:       int | None = None,
    zone_box_h:       int | None = None,
    save:             bool = True,
) -> dict[str, list[int]]:
    """
    Automatically detect scoring zone bounding boxes from a match video.

    Uses three signals (ball disappearance, structural edges, detection
    heatmap) and combines them. The two strongest clusters that are on
    opposite sides of the field are labelled red_goal and blue_goal.

    Args:
        video_path:      Path to match video.
        config_path:     field_config.json to update.
        detections_path: Cached Roboflow detections (optional, improves accuracy).
        n_samples:       Number of frames to sample for disappearance analysis.
        zone_box_w/h:    Override zone bbox size in pixels. Auto-sized if None.
        save:            Write result to field_config.json.

    Returns:
        {"red_high_goal": [x1,y1,x2,y2], "blue_high_goal": [...], ...}
    """
    video_path      = Path(video_path)
    config_path     = Path(config_path)
    detections_path = Path(detections_path)

    # Get frame dimensions
    cap     = cv2.VideoCapture(str(video_path))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  [FieldCal] Video: {frame_w}x{frame_h}")
    print(f"  [FieldCal] Running auto-detection with {n_samples} sample frames...")

    # Default zone box size: ~10% of frame width x ~15% of field height
    field_h      = int(frame_h * 0.76)   # approximate field area height
    box_w = zone_box_w or max(80,  int(frame_w * 0.10))
    box_h = zone_box_h or max(100, int(field_h * 0.18))

    # ---- Gather signals ----
    all_candidates: list[tuple[float, float]] = []

    print("  [FieldCal] Signal A: ball disappearance analysis...")
    sig_a = _disappearance_signal(video_path, n_samples, frame_w, frame_h)
    all_candidates.extend(sig_a)
    print(f"             -> {len(sig_a)} disappearance points")

    print("  [FieldCal] Signal B: structural edge analysis...")
    sig_b = _structural_signal(video_path, frame_w, frame_h)
    all_candidates.extend(sig_b)
    print(f"             -> {len(sig_b)} structural points")

    print("  [FieldCal] Signal C: detection heatmap...")
    sig_c = _detection_heatmap_signal(detections_path, frame_w, frame_h)
    all_candidates.extend(sig_c)
    print(f"             -> {len(sig_c)} heatmap points")

    if not all_candidates:
        print("  [FieldCal] (!) No candidates found; using default zone positions.")
        return _default_zones(frame_w, frame_h, box_w, box_h)

    # ---- Cluster all points ----
    cluster_radius = max(box_w, box_h) * 0.8
    clusters = _cluster_points(all_candidates, radius=cluster_radius)
    print(f"  [FieldCal] Clusters found: {len(clusters)}")

    # ---- Split into left / right (red / blue) ----
    # Red alliance = left side of frame, Blue = right side
    centre_x = frame_w / 2
    left_clusters  = [(cx, cy) for cx, cy in clusters if cx <  centre_x]
    right_clusters = [(cx, cy) for cx, cy in clusters if cx >= centre_x]

    # Pick strongest cluster per side: closest to alliance wall
    # Left: smallest cx; Right: largest cx
    red_centre  = min(left_clusters,  key=lambda p: p[0]) \
                  if left_clusters  else (frame_w * 0.08, frame_h * 0.35)
    blue_centre = max(right_clusters, key=lambda p: p[0]) \
                  if right_clusters else (frame_w * 0.92, frame_h * 0.35)

    red_bbox  = _centre_to_bbox(*red_centre,  box_w, box_h, frame_w, frame_h)
    blue_bbox = _centre_to_bbox(*blue_centre, box_w, box_h, frame_w, frame_h)

    zones = {
        "red_goal":  red_bbox,
        "blue_goal": blue_bbox,
    }

    print(f"  [FieldCal] Detected zones:")
    for name, bbox in zones.items():
        print(f"             {name}: {bbox}")

    if save:
        _write_zones_to_config(zones, config_path)

    return zones


def _default_zones(frame_w: int, frame_h: int,
                   box_w: int, box_h: int) -> dict[str, list[int]]:
    """Fallback zone positions at 7% and 93% of frame width."""
    field_top = int(frame_h * 0.15)
    return {
        "red_goal":  _centre_to_bbox(int(frame_w*0.07), field_top + box_h//2,
                                     box_w, box_h, frame_w, frame_h),
        "blue_goal": _centre_to_bbox(int(frame_w*0.93), field_top + box_h//2,
                                     box_w, box_h, frame_w, frame_h),
    }


def _write_zones_to_config(zones: dict[str, list[int]],
                            config_path: Path) -> None:
    """Write auto-detected scoring zones ONLY if no valid zones already exist.

    'Valid' means every existing zone has y1 < 700 (i.e. covers real field
    area, not the scoreboard strip at the bottom of the frame).  If the user
    already set zones via Box Goals or a previous calibration, we leave them
    alone and only update the scoreboard bbox.
    """
    with open(config_path) as f:
        cfg = json.load(f)

    existing = cfg.get("scoring_zones", {})
    valid_zones = {k: v for k, v in existing.items()
                   if not k.startswith("_") and isinstance(v, list)
                   and len(v) == 4 and v[1] < 700}   # y1 < 700 = on the field

    if valid_zones:
        print(f"  [FieldCal] Keeping existing scoring zones "
              f"(user-calibrated): {list(valid_zones.keys())}")
        return   # do not overwrite

    # No valid zones present — write auto-detected ones
    new_sz = {"_comment": cfg.get("scoring_zones", {}).get(
        "_comment",
        "Auto-detected by field_calibration.py. Use Box Goals to override."
    )}
    new_sz.update(zones)
    cfg["scoring_zones"] = new_sz

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"  [FieldCal] Scoring zones saved -> {config_path}")


# ---- Auto-detect scoreboard (also wired in here for one-stop calibration) ---

def auto_detect_scoreboard(
    video_path:  str | Path,
    config_path: str | Path = "configs/field_config.json",
    save:        bool = True,
) -> list[int] | None:
    """
    Detect the scoreboard overlay bbox from a sample frame and optionally
    save it to field_config.json.

    Returns:
        [x1, y1, x2, y2] or None.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from scoreboard import locate_scoreboard

    cap = cv2.VideoCapture(str(Path(video_path)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bbox = None
    for frac in (0.20, 0.40, 0.60):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
        ret, frame = cap.read()
        if not ret:
            continue
        bbox = locate_scoreboard(frame, None)
        if bbox:
            break
    cap.release()

    if bbox and save:
        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = json.load(f)
        frame_w = int(cv2.VideoCapture(str(video_path)).get(cv2.CAP_PROP_FRAME_WIDTH)) or bbox[2]
        cfg["scoreboard"]["bbox"]           = bbox
        cfg["scoreboard"]["auto_detected"]  = True
        cfg["scoreboard"]["red_score_region"]  = [0, bbox[1], int(frame_w * 0.42), bbox[3]]
        cfg["scoreboard"]["blue_score_region"] = [int(frame_w * 0.58), bbox[1], frame_w, bbox[3]]
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"  [FieldCal] Scoreboard bbox {bbox} saved -> {config_path}")

    return bbox


# ---- Combined calibration entry point ---------------------------------------

def calibrate_field(
    video_path:      str | Path,
    config_path:     str | Path = "configs/field_config.json",
    detections_path: str | Path = "data/detections.json",
    n_samples:       int = 40,
) -> dict:
    """
    Run all field auto-calibration steps:
      1. Auto-detect scoreboard bbox
      2. Auto-detect scoring zones

    Results are written to field_config.json automatically.

    Args:
        video_path:      Match video path.
        config_path:     field_config.json path.
        detections_path: Cached detections (optional).
        n_samples:       Frames to sample for zone detection.

    Returns:
        {"scoreboard_bbox": [...], "scoring_zones": {...}}
    """
    print("\n" + "=" * 60)
    print("FIELD AUTO-CALIBRATION")
    print("=" * 60)

    sb_bbox = auto_detect_scoreboard(video_path, config_path, save=True)
    zones   = auto_detect_scoring_zones(
        video_path, config_path, detections_path, n_samples, save=True
    )

    print("\n  [FieldCal] Calibration complete.")
    print(f"  Scoreboard : {sb_bbox}")
    print(f"  Zones      : {list(zones.keys())}")

    return {"scoreboard_bbox": sb_bbox, "scoring_zones": zones}
