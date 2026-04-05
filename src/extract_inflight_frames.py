"""
src/extract_inflight_frames.py - Generate in-flight ball training data.

Scans the match video at full frame rate, uses Hough circle detection +
optical flow to find FUEL in-flight, and outputs:
  - data/inflight_training/images/*.jpg  (cropped frame context)
  - data/inflight_training/labels/*.txt  (YOLOv8 format annotations)
  - data/inflight_training/review/       (annotated frames for human review)
  - data/inflight_training/dataset.yaml  (ready for Roboflow upload or local train)

Upload the images/ + labels/ folders to Roboflow, correct any bad labels,
then export in YOLOv8 format and run:
    py -3.13 main.py --train-inflight

This replaces the generic FUEL model with one that specialises in in-flight
detection, expected to push coverage from ~12% to >70%.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


# ─── Config ──────────────────────────────────────────────────────────────────

# Frame sampling for the scan pass.
# Use 1 (every frame) for maximum coverage, 2 for faster scanning.
SCAN_SAMPLE = 1

# Hough circle detection params (tuned for FUEL: 5.91" sphere)
# Estimated pixel diameter at mid-field: ~20-50 px
_HOUGH_DP        = 1.2
_HOUGH_MIN_DIST  = 15      # px between circle centers
_HOUGH_PARAM1    = 60      # Canny high threshold
_HOUGH_PARAM2    = 18      # accumulator threshold (lower = more circles detected)
_HOUGH_MIN_R     = 8       # pixels
_HOUGH_MAX_R     = 40      # pixels

# Optical flow: minimum displacement to count as "in-flight"
_MIN_FLOW_MAG_PX = 3.0     # px / frame

# Exclude balls sitting on the floor (below this y-fraction of frame)
_FLOOR_Y_FRAC = 0.80

# Minimum frames a candidate must be consistently detected to save it
_MIN_PERSIST_FRAMES = 2

# How many pixels of padding to add around the ball when saving crop
_CONTEXT_PAD_FRAC = 3.0    # crop = ball_diameter * this value

# Output directory
_OUT_DIR = Path("data/inflight_training")

# YOLO class index for ball (must match your dataset.yaml)
_BALL_CLASS_IDX = 0

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hough_balls(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Return list of (cx, cy, r) from Hough circle detection."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=_HOUGH_DP,
        minDist=_HOUGH_MIN_DIST,
        param1=_HOUGH_PARAM1,
        param2=_HOUGH_PARAM2,
        minRadius=_HOUGH_MIN_R,
        maxRadius=_HOUGH_MAX_R,
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def _flow_at_point(
    flow: np.ndarray, cx: float, cy: float, r: float
) -> float:
    """Mean optical flow magnitude inside a circle of radius r at (cx, cy)."""
    h, w = flow.shape[:2]
    x1 = max(0, int(cx - r))
    y1 = max(0, int(cy - r))
    x2 = min(w, int(cx + r) + 1)
    y2 = min(h, int(cy + r) + 1)
    roi = flow[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mag = np.sqrt(roi[..., 0]**2 + roi[..., 1]**2)
    return float(np.mean(mag))


def _to_yolo_label(cx: float, cy: float, r: float, W: int, H: int) -> str:
    """Convert circle params to YOLOv8 normalised label string."""
    diam = 2 * r
    nx   = cx / W
    ny   = cy / H
    nw   = diam / W
    nh   = diam / H
    # Clamp to [0, 1]
    nx  = max(0.0, min(1.0, nx))
    ny  = max(0.0, min(1.0, ny))
    nw  = max(0.001, min(1.0, nw))
    nh  = max(0.001, min(1.0, nh))
    return f"{_BALL_CLASS_IDX} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def _draw_annotation(
    frame: np.ndarray,
    detections: list[tuple[float, float, float]],
    flow_mags: list[float],
) -> np.ndarray:
    """Draw annotated preview frame."""
    vis = frame.copy()
    for (cx, cy, r), mag in zip(detections, flow_mags):
        color = (0, 255, 0) if mag >= _MIN_FLOW_MAG_PX else (0, 165, 255)
        cv2.circle(vis, (int(cx), int(cy)), int(r), color, 2)
        cv2.putText(
            vis, f"{mag:.1f}px/f",
            (int(cx) - 20, int(cy) - int(r) - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
    return vis


# ─── Main function ────────────────────────────────────────────────────────────

def extract_inflight_frames(
    video_path: str | Path,
    out_dir: str | Path = _OUT_DIR,
    scan_sample: int = SCAN_SAMPLE,
    save_review: bool = True,
    max_frames: int | None = None,
) -> dict:
    """
    Scan video at near-full rate, detect in-flight FUEL balls using Hough
    circles + optical flow, and write YOLOv8-format training data.

    Args:
        video_path:   Path to match video (.mp4/.mov/.avi).
        out_dir:      Root output directory.
        scan_sample:  Read every N-th frame (1 = every frame).
        save_review:  Write annotated preview images for human inspection.
        max_frames:   Stop after this many source frames (None = whole video).

    Returns:
        Summary dict with counts of frames saved, detections found, etc.
    """
    video_path = Path(video_path)
    out_dir    = Path(out_dir)
    img_dir    = out_dir / "images"
    lbl_dir    = out_dir / "labels"
    rev_dir    = out_dir / "review"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    if save_review:
        rev_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  [ExtractInflight] {video_path.name}")
    print(f"  [ExtractInflight] {total_frames} frames @ {fps:.1f} fps, {W}x{H}")
    print(f"  [ExtractInflight] Scan sample: every {scan_sample} frame(s)")
    print(f"  [ExtractInflight] Output: {out_dir}")

    n_saved       = 0
    n_detections  = 0
    prev_gray     = None
    frame_idx     = 0

    # Track candidates across frames: (cx, cy) -> frames_seen
    # simple IoU-free proximity tracker
    active: dict[int, dict] = {}   # track_id -> {cx, cy, r, last_frame, frames}
    next_tid = 0
    PROX = 30   # px: same circle within this distance = same track

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if max_frames and frame_idx >= max_frames:
                break
            if frame_idx % scan_sample != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Compute dense optical flow from previous frame
            flow: np.ndarray | None = None
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray,
                    None,
                    pyr_scale=0.5, levels=3, winsize=13,
                    iterations=3, poly_n=5, poly_sigma=1.2,
                    flags=0,
                )

            # Detect circles
            candidates = _hough_balls(gray)
            floor_y    = H * _FLOOR_Y_FRAC

            inflight: list[tuple[float, float, float]] = []
            flow_mags: list[float] = []
            for (cx, cy, r) in candidates:
                if cy > floor_y:
                    continue   # on floor, skip
                mag = _flow_at_point(flow, cx, cy, r) if flow is not None else 0.0
                flow_mags.append(mag)
                if mag >= _MIN_FLOW_MAG_PX or prev_gray is None:
                    inflight.append((cx, cy, r))

            # Update persistence tracker
            matched_tids: set[int] = set()
            for (cx, cy, r) in inflight:
                best_tid, best_dist = None, PROX + 1
                for tid, trk in active.items():
                    d = math.hypot(trk["cx"] - cx, trk["cy"] - cy)
                    if d < best_dist:
                        best_dist = d
                        best_tid  = tid
                if best_tid is not None:
                    active[best_tid].update({"cx": cx, "cy": cy, "r": r,
                                             "last_frame": frame_idx})
                    active[best_tid]["frames"] += 1
                    matched_tids.add(best_tid)
                else:
                    active[next_tid] = {"cx": cx, "cy": cy, "r": r,
                                        "last_frame": frame_idx, "frames": 1}
                    matched_tids.add(next_tid)
                    next_tid += 1

            # Expire old tracks
            expired = [t for t, trk in active.items()
                       if frame_idx - trk["last_frame"] > 5]
            for t in expired:
                del active[t]

            # Save frames that have in-flight balls with sufficient persistence
            persistent = [
                active[t] for t in matched_tids
                if active[t]["frames"] >= _MIN_PERSIST_FRAMES
            ]
            if persistent:
                stem   = f"frame_{frame_idx:07d}"
                labels = []
                for trk in persistent:
                    cx, cy, r = trk["cx"], trk["cy"], trk["r"]
                    labels.append(_to_yolo_label(cx, cy, r, W, H))
                    n_detections += 1

                # Save full frame (not crop - YOLO needs full frame + label)
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                (lbl_dir / f"{stem}.txt").write_text("\n".join(labels))

                if save_review:
                    vis = _draw_annotation(
                        frame,
                        [(trk["cx"], trk["cy"], trk["r"]) for trk in persistent],
                        [trk.get("flow_mag", 0.0) for trk in persistent],
                    )
                    cv2.imwrite(str(rev_dir / f"{stem}_review.jpg"), vis,
                                [cv2.IMWRITE_JPEG_QUALITY, 80])

                n_saved += 1

            prev_gray  = gray
            frame_idx += 1

            if frame_idx % 500 == 0:
                print(f"  [ExtractInflight] Frame {frame_idx}/{total_frames} "
                      f"- {n_saved} frames saved so far")

    finally:
        cap.release()

    # Write dataset.yaml
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# In-flight FUEL ball dataset\n"
        f"# Auto-generated by extract_inflight_frames.py\n"
        f"# Review images in review/ before uploading to Roboflow.\n\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images\n"
        f"val:   images\n"
        f"test:  images\n\n"
        f"nc: 1\n"
        f"names: ['ball']\n"
    )

    # Write summary JSON
    summary = {
        "video":          str(video_path),
        "total_frames":   total_frames,
        "scan_sample":    scan_sample,
        "frames_saved":   n_saved,
        "detections":     n_detections,
        "output_dir":     str(out_dir.resolve()),
        "images_dir":     str(img_dir.resolve()),
        "labels_dir":     str(lbl_dir.resolve()),
        "review_dir":     str(rev_dir.resolve()) if save_review else None,
        "dataset_yaml":   str(yaml_path.resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  [ExtractInflight] Done.")
    print(f"  [ExtractInflight] {n_saved} frames saved, {n_detections} ball detections")
    print(f"  [ExtractInflight] Images  : {img_dir}")
    print(f"  [ExtractInflight] Labels  : {lbl_dir}")
    if save_review:
        print(f"  [ExtractInflight] Review  : {rev_dir}")
    print(f"  [ExtractInflight] YAML    : {yaml_path}")
    print()
    print("  Next steps:")
    print("  1. Check review/ images - delete any bad labels (non-ball circles)")
    print("  2. Upload images/ + labels/ to Roboflow as a new dataset version")
    print("  3. Add augmentations: motion blur (kernel 7-15), brightness +/-30%,")
    print("     horizontal flip, rotation +/-15 deg")
    print("  4. Train YOLOv8s on Roboflow (or locally: see --train-inflight flag)")
    print("  5. Download best.pt -> models/trained/balls/best.pt")
    print("  6. Re-run pipeline without --skip-detect to use the new model")

    return summary


# ─── Local training shortcut ─────────────────────────────────────────────────

def train_inflight_model(
    data_dir: str | Path = _OUT_DIR,
    epochs: int = 50,
    imgsz: int = 640,
    base_weights: str = "yolov8s.pt",
) -> Path:
    """
    Train a YOLOv8s in-flight ball model locally on the extracted dataset.

    Requires Ultralytics (pip install ultralytics).
    GPU strongly recommended; will work on CPU but slowly.

    Args:
        data_dir:      Directory produced by extract_inflight_frames().
        epochs:        Training epochs (50 is usually enough for fine-tuning).
        imgsz:         Input image size (640 standard).
        base_weights:  Starting weights. Use existing ball model if available.

    Returns:
        Path to best.pt weights file.
    """
    from ultralytics import YOLO

    data_dir  = Path(data_dir)
    yaml_path = data_dir / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run extract_inflight_frames() first."
        )

    # Prefer existing ball model as starting point (transfer learning)
    root = Path(__file__).parent.parent
    existing_ball = root / "models" / "trained" / "balls" / "best.pt"
    if existing_ball.exists():
        base_weights = str(existing_ball)
        print(f"  [TrainInflight] Fine-tuning from existing ball model: {base_weights}")
    else:
        print(f"  [TrainInflight] Starting from base weights: {base_weights}")

    out_dir = root / "models" / "trained" / "balls_inflight"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [TrainInflight] Training for {epochs} epochs on {yaml_path}")

    model = YOLO(base_weights)
    results = model.train(
        data       = str(yaml_path),
        epochs     = epochs,
        imgsz      = imgsz,
        project    = str(out_dir),
        name       = "train",
        exist_ok   = True,
        augment    = True,
        # Motion-blur augmentation: critical for in-flight detection
        degrees    = 15.0,       # rotation
        fliplr     = 0.5,        # horizontal flip
        scale      = 0.3,        # scale jitter
        mosaic     = 0.5,
        # YOLOv8 built-in motion blur aug
        erasing    = 0.2,
    )

    # Copy best weights to standard location
    trained_best = out_dir / "train" / "weights" / "best.pt"
    final_dest   = root / "models" / "trained" / "balls" / "best.pt"
    final_dest.parent.mkdir(parents=True, exist_ok=True)

    if trained_best.exists():
        import shutil as _shutil
        _shutil.copy2(trained_best, final_dest)
        print(f"  [TrainInflight] Best weights -> {final_dest}")
        print(f"  [TrainInflight] Re-run pipeline without --skip-detect to use new model.")
        return final_dest
    else:
        print(f"  [TrainInflight] (!) Could not find best.pt at {trained_best}")
        return trained_best


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract in-flight FUEL frames for training")
    parser.add_argument("video", help="Path to match video")
    parser.add_argument("--out",     default=str(_OUT_DIR), help="Output directory")
    parser.add_argument("--sample",  type=int, default=1,   help="Scan every N frames (default 1)")
    parser.add_argument("--train",   action="store_true",   help="Train model after extraction")
    parser.add_argument("--epochs",  type=int, default=50,  help="Training epochs (default 50)")
    parser.add_argument("--no-review", action="store_true", help="Skip saving review images")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N source frames")
    args = parser.parse_args()

    summary = extract_inflight_frames(
        args.video,
        out_dir    = args.out,
        scan_sample= args.sample,
        save_review= not args.no_review,
        max_frames = args.max_frames,
    )

    if args.train:
        if summary["frames_saved"] < 50:
            print(f"  (!) Only {summary['frames_saved']} frames extracted. "
                  "Recommend at least 200 before training.")
            ans = input("  Train anyway? (yes/no): ").strip().lower()
            if ans != "yes":
                print("  Training skipped.")
                raise SystemExit(0)
        train_inflight_model(out_dir=args.out, epochs=args.epochs)
