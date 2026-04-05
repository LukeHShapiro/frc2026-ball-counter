"""
src/ingest.py — Phase 2: Video Ingestion

Functions:
  extract_frames()       — decode video -> individual frame images at target fps
  filter_duplicates()    — perceptual-hash deduplication (target 500–2000 frames)
  log_video_metadata()   — save metadata.json to data/

Depends on: Phase 1 environment (opencv-python, Pillow installed).

(!)️ STOP after this module is complete —
   "PHASE 2 COMPLETE. Ready for INPUT CHECKPOINT #1 (match video)."
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path
from typing import Tuple

import cv2


# ── Supported formats ─────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi"}


# ─────────────────────────────────────────────────────────────────────────────
# extract_frames
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: float = 10.0,
) -> Tuple[int, float, Tuple[int, int]]:
    """
    Decode a video file and write one frame image per interval to output_dir.

    Accepts .mp4, .mov, .avi.  Frames are written as
    output_dir/frame_{frame_number:06d}.jpg

    Args:
        video_path:  Path to source video (.mp4 / .mov / .avi).
        output_dir:  Directory to write frame images into.
        fps:         Target extraction rate in frames per second.
                     e.g. fps=10 -> one frame every 0.1 s of video time.

    Returns:
        (total_frames_written, duration_seconds, (width, height))

    Raises:
        ValueError:  Unsupported file extension or video cannot be opened.

    Depends on: Phase 1 (opencv-python).
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format '{video_path.suffix}'. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration        = total_src_frames / source_fps if source_fps > 0 else 0.0

    # How many source frames to skip between each saved frame
    frame_interval = max(1, round(source_fps / fps))

    print(f"  [Ingest] Source: {video_path.name}")
    print(f"  [Ingest] {width}x{height}  |  {source_fps:.2f} fps  |  "
          f"{duration:.1f}s  |  {total_src_frames} total frames")
    print(f"  [Ingest] Extracting every {frame_interval} frames "
          f"(target {fps} fps)...")

    # Cap write resolution at 1080p to avoid multi-GB disk writes for 4K video.
    # Detection reads the original video directly — these frames are for labeling only.
    MAX_WRITE_HEIGHT = 1080
    if height > MAX_WRITE_HEIGHT:
        scale = MAX_WRITE_HEIGHT / height
        write_w = int(width  * scale)
        write_h = MAX_WRITE_HEIGHT
        print(f"  [Ingest] Resizing frames to {write_w}x{write_h} for storage")
    else:
        write_w, write_h = width, height

    # Skip extraction if frames already exist from a previous run
    existing = list(output_dir.glob("frame_*.jpg"))
    if existing:
        print(f"  [Ingest] Found {len(existing)} existing frames - skipping re-extraction.")
        cap.release()
        return len(existing), duration, (width, height)

    frames_written = 0
    src_frame_idx  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if src_frame_idx % frame_interval == 0:
            if write_w != width or write_h != height:
                frame = cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_AREA)
            out_path = output_dir / f"frame_{src_frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frames_written += 1

        src_frame_idx += 1

    cap.release()

    print(f"  [Ingest] Wrote {frames_written} frames -> {output_dir}")
    return frames_written, duration, (width, height)


# ─────────────────────────────────────────────────────────────────────────────
# _phash  — simple 8×8 perceptual hash
# ─────────────────────────────────────────────────────────────────────────────

def _phash(image_path: Path) -> int:
    """
    Compute an 8×8 average-hash (pHash-lite) for a JPEG frame.

    Returns a 64-bit integer hash.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean  = small.mean()
    bits  = (small > mean).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    x = a ^ b
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# filter_duplicates
# ─────────────────────────────────────────────────────────────────────────────

def filter_duplicates(
    frames_dir: str | Path,
    similarity_threshold: float = 0.95,
) -> int:
    """
    Remove near-duplicate frames using mean-absolute-difference (MAD) on
    downscaled greyscale images compared frame-to-frame.

    FRC footage has a static background (field carpet/walls) that defeats
    whole-frame perceptual hashing. MAD on consecutive frames correctly
    identifies only truly frozen/static segments (paused feed, replays,
    scoreboard cutaways) while keeping every frame where robots moved.

    similarity_threshold maps to a MAD pixel-difference floor:
      0.95 -> frames must differ by > 5%  of 255 to be kept  (aggressive)
      0.85 -> frames must differ by > 15% of 255 to be kept  (default tuning)

    Duplicate frames are moved to frames_dir/duplicates/ (reversible).
    Target: 500–2000 kept frames for labeling.

    Args:
        frames_dir:           Directory of extracted .jpg frames.
        similarity_threshold: 0.0–1.0. Higher = more aggressive deduplication.

    Returns:
        Number of frames kept after deduplication.

    Depends on: Phase 2 extract_frames() output.
    """
    frames_dir = Path(frames_dir)
    dup_dir    = frames_dir / "duplicates"
    dup_dir.mkdir(exist_ok=True)

    # MAD threshold: fraction of max pixel value (255) that must differ
    # for a frame to be considered "new".
    # similarity_threshold=0.95 -> mad_floor = (1-0.95)*255 = 12.75
    mad_floor = (1.0 - similarity_threshold) * 255.0
    mad_floor = max(1.0, mad_floor)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    total_in    = len(frame_paths)

    if total_in == 0:
        print("  [Ingest] No frames found for deduplication.")
        return 0

    # Stage 1: remove truly frozen frames (MAD < 1.0/255 ≈ no change at all).
    # FRC robots are small vs the full frame, so we only flag actual freezes
    # (paused feed, replay hold, scoreboard cutaway) — not normal play.
    frozen_mad = 2.0   # pixel units out of 255; anything below this = frozen

    print(f"  [Ingest] Stage 1 - removing frozen frames (MAD < {frozen_mad}/255)...")

    import numpy as np

    kept:    list[Path] = []
    removed: list[Path] = []
    prev_small = None

    # Fixed thumbnail size — must be identical for every frame so MAD subtraction works
    THUMB_W, THUMB_H = 80, 45

    for path in frame_paths:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            kept.append(path)
            continue
        small = cv2.resize(img, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)

        if prev_small is None:
            kept.append(path)
            prev_small = small
            continue

        mad = float(np.mean(np.abs(small.astype(np.float32)
                                   - prev_small.astype(np.float32))))
        if mad < frozen_mad:
            dest = dup_dir / path.name
            path.replace(dest)             # atomic overwrite on Windows
            removed.append(path)
        else:
            kept.append(path)
            prev_small = small

    print(f"  [Ingest] Frozen removed: {len(removed)} | Active frames: {len(kept)}")

    # Stage 2: if still over 2000, uniformly subsample to ~1500.
    TARGET_MAX = 2000
    TARGET_SUBSAMPLE = 1500
    if len(kept) > TARGET_MAX:
        step = len(kept) / TARGET_SUBSAMPLE
        keep_set = {kept[round(i * step)] for i in range(TARGET_SUBSAMPLE)}
        extra_removed = []
        for p in kept:
            if p not in keep_set:
                dest = dup_dir / p.name
                p.replace(dest)            # atomic overwrite on Windows
                extra_removed.append(p)
        kept = [p for p in kept if p in keep_set]
        print(f"  [Ingest] Stage 2 subsample: moved {len(extra_removed)} more "
              f"-> {len(kept)} frames remain")

    kept_count = len(kept)
    print(f"  [Ingest] Kept {kept_count} frames, "
          f"moved {len(removed)} duplicates -> {dup_dir}")

    if kept_count < 500:
        print(f"  [Ingest] (!) Only {kept_count} frames kept "
              f"(target: 500-2000). Try extracting at higher fps.")
    elif kept_count > 2000:
        print(f"  [Ingest] (!) {kept_count} frames kept - slightly above target.")
    else:
        print(f"  [Ingest] OK {kept_count} frames in target range "
              f"(500-2000). Ready for labeling.")

    return kept_count


# ─────────────────────────────────────────────────────────────────────────────
# log_video_metadata
# ─────────────────────────────────────────────────────────────────────────────

def log_video_metadata(video_path: str | Path) -> Path:
    """
    Read video properties and save them to data/metadata.json.

    Written fields:
      file_name, file_size_mb, duration_seconds, source_fps,
      total_frames, width, height, codec, logged_at

    Args:
        video_path: Path to source video.

    Returns:
        Path to the written metadata.json.

    Depends on: Phase 1 (opencv-python).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video for metadata: {video_path}")

    source_fps       = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int       = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec            = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)).strip()
    duration         = total_frames / source_fps if source_fps > 0 else 0.0
    file_size_mb     = video_path.stat().st_size / (1024 * 1024) if video_path.exists() else 0.0
    cap.release()

    metadata = {
        "file_name":        video_path.name,
        "file_path":        str(video_path.resolve()),
        "file_size_mb":     round(file_size_mb, 2),
        "duration_seconds": round(duration, 2),
        "source_fps":       round(source_fps, 3),
        "total_frames":     total_frames,
        "width":            width,
        "height":           height,
        "codec":            codec,
        "logged_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path = Path("data") / "metadata.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  [Ingest] Metadata saved -> {out_path}")
    for k, v in metadata.items():
        if not k.startswith("file_path"):
            print(f"           {k}: {v}")

    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI helper (for quick testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Phase 2 — Video Ingestion")
    parser.add_argument("video", help="Path to .mp4/.mov/.avi match video")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="Frame extraction rate (default: 10)")
    parser.add_argument("--out", default="data/raw_frames",
                        help="Output directory for frames")
    parser.add_argument("--similarity", type=float, default=0.95,
                        help="Dedup similarity threshold (default: 0.95)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PHASE 2 - Video Ingestion")
    print("="*60)

    n_frames, duration, res = extract_frames(args.video, args.out, fps=args.fps)
    log_video_metadata(args.video)
    kept = filter_duplicates(args.out, similarity_threshold=args.similarity)

    print("\n" + "="*60)
    print(f"  Extracted : {n_frames} frames")
    print(f"  Duration  : {duration:.1f}s  |  Resolution: {res[0]}x{res[1]}")
    print(f"  After dedup: {kept} frames ready for labeling")
    print("="*60)
    print("\nPHASE 2 COMPLETE. Ready for INPUT CHECKPOINT #1 (match video).")
    sys.exit(0)
