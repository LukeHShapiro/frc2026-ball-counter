"""
src/scoreboard.py - Phase 8: Scoreboard OCR

Reads alliance scores from the on-screen scoreboard and validates
attribution totals against them.

Functions:
  locate_scoreboard()      - find scoreboard region in frame
  read_score()             - EasyOCR on scoreboard -> {red, blue, confidence}
  detect_score_change()    - compare current vs previous reading
  validate_attribution()   - check robot score sums against scoreboard

Depends on: Phase 5 frames, INPUT CHECKPOINT #6 (scoreboard coords or auto).

(!) STOP after locate + read confirmed on a sample frame --
   "PHASE 8 READY. Awaiting INPUT CHECKPOINT #6 (scoreboard location)."
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np


# ---- Lazy OCR reader --------------------------------------------------------

_ocr_reader = None

def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _ocr_reader


# ---- locate_scoreboard ------------------------------------------------------

def locate_scoreboard(
    frame:  np.ndarray,
    coords: list[int] | None = None,
) -> list[int] | None:
    """
    Return the bounding box [x1, y1, x2, y2] of the scoreboard region.

    If coords is provided (from INPUT CHECKPOINT #6), use those directly.
    Otherwise attempt auto-detection by looking for the dark horizontal
    banner typically found at the top of FRC broadcast overlays.

    Args:
        frame:   BGR image array.
        coords:  User-supplied [x1, y1, x2, y2], or None for auto-detect.

    Returns:
        [x1, y1, x2, y2] in pixels, or None if not found.
    """
    if coords is not None:
        return coords

    # Auto-detect: scan bottom 40% then top 20% for a wide dark horizontal band.
    # FRC broadcast overlays are typically full-width dark bars at the BOTTOM.
    h, w = frame.shape[:2]

    # Primary: scan bottom 40%
    bot_start  = int(h * 0.60)
    bot_region = frame[bot_start:, :]
    gray       = cv2.cvtColor(bot_region, cv2.COLOR_BGR2GRAY)
    row_means  = gray.mean(axis=1)
    dark_rows  = [i for i, m in enumerate(row_means) if m < 110]
    if len(dark_rows) >= 8:
        y1 = bot_start + dark_rows[0]
        y2 = h   # extend to bottom edge (overlay goes to frame edge)
        return [0, y1, w, y2]

    # Fallback: top 20%
    top_region = frame[:int(h * 0.20), :]
    gray       = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
    row_means  = gray.mean(axis=1)
    dark_rows  = [i for i, m in enumerate(row_means) if m < 110]
    if len(dark_rows) >= 8:
        y1 = dark_rows[0]
        y2 = dark_rows[-1]
        return [0, y1, w, y2]

    return None


# ---- read_score -------------------------------------------------------------

def read_score(
    frame:           np.ndarray,
    scoreboard_bbox: list[int],
) -> dict:
    """
    Run EasyOCR on the scoreboard region to extract alliance scores.

    Looks for two groups of digits separated by whitespace or a dash,
    corresponding to Red Alliance score | Blue Alliance score.

    Args:
        frame:           BGR image array.
        scoreboard_bbox: [x1, y1, x2, y2] from locate_scoreboard().

    Returns:
        {
            red_score:  int,   # -1 if not readable
            blue_score: int,   # -1 if not readable
            confidence: float, # 0.0–1.0
            raw_text:   str,
        }
    """
    x1, y1, x2, y2 = scoreboard_bbox
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"red_score": -1, "blue_score": -1, "confidence": 0.0, "raw_text": ""}

    # FRC broadcast overlay: RED score is in the left ~40% of the bar,
    # BLUE score is in the right ~40%.  The centre ~20% holds the timer.
    # Read each half independently to avoid confusing them.
    cw = crop.shape[1]
    red_crop  = crop[:, :int(cw * 0.42)]
    blue_crop = crop[:, int(cw * 0.58):]

    reader = _get_ocr()

    def _read_half(half_crop: np.ndarray) -> tuple[int, float]:
        """OCR one half, return (best_score, confidence).

        The FRC scoreboard has two rows:
          Top ~50%: team number logos row  (we IGNORE this)
          Bottom ~50%: big alliance score  (we READ this)
        We crop to the bottom 55% to avoid reading team numbers.
        """
        h_h = half_crop.shape[0]
        # Focus on bottom 55% — that's where the big score digit lives
        score_row = half_crop[int(h_h * 0.45):, :]

        # Scale up for better OCR accuracy
        scale = max(1, min(6, 180 // max(score_row.shape[0], 1)))
        if scale > 1:
            score_row = cv2.resize(score_row, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)
        gray   = cv2.cvtColor(score_row, cv2.COLOR_BGR2GRAY)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        results = reader.readtext(thr, detail=1, allowlist="0123456789")
        if not results:
            return -1, 0.0

        # Largest bounding box in this row = the main score number (1-3 digits)
        candidates = []
        for (bbox_pts, text, conf) in results:
            text = text.strip()
            if text.isdigit() and 1 <= len(text) <= 3 and 0 <= int(text) <= 999:
                pts  = bbox_pts
                area = abs((pts[2][0]-pts[0][0]) * (pts[2][1]-pts[0][1]))
                candidates.append((area, int(text), conf))

        if not candidates:
            return -1, 0.0
        candidates.sort(reverse=True)   # largest visual area first
        return candidates[0][1], candidates[0][2]

    red_score,  red_conf  = _read_half(red_crop)
    blue_score, blue_conf = _read_half(blue_crop)
    avg_conf = (red_conf + blue_conf) / 2

    all_text = f"red={red_score} blue={blue_score}"
    return {
        "red_score":  red_score,
        "blue_score": blue_score,
        "confidence": round(avg_conf, 3),
        "raw_text":   all_text,
    }


# ---- detect_score_change ----------------------------------------------------

def detect_score_change(
    score_history:  list[dict],
    current_score:  dict,
    frame_id:       int,
) -> dict:
    """
    Compare current score reading against previous readings.

    Args:
        score_history:  List of prior {red_score, blue_score, frame_id} dicts.
        current_score:  Current reading from read_score().
        frame_id:       Current frame number.

    Returns:
        {
            changed:  bool,
            delta:    int,      # Points added this change
            alliance: str,      # "red" | "blue" | "unknown"
            frame_id: int,
        }
    """
    no_change = {"changed": False, "delta": 0, "alliance": "unknown",
                 "frame_id": frame_id}

    if not score_history:
        return no_change

    prev = score_history[-1]
    red_delta  = current_score.get("red_score",  -1) - prev.get("red_score",  -1)
    blue_delta = current_score.get("blue_score", -1) - prev.get("blue_score", -1)

    # Filter noise: ignore negative deltas or impossible jumps
    red_delta  = red_delta  if 0 < red_delta  <= 50 else 0
    blue_delta = blue_delta if 0 < blue_delta <= 50 else 0

    if red_delta > 0:
        return {"changed": True, "delta": red_delta,
                "alliance": "red",  "frame_id": frame_id}
    if blue_delta > 0:
        return {"changed": True, "delta": blue_delta,
                "alliance": "blue", "frame_id": frame_id}

    return no_change


# ---- validate_attribution ---------------------------------------------------

def validate_attribution(
    robot_scores:     dict[str, int],
    scoreboard_scores: dict,
) -> dict:
    """
    Verify that attributed robot scores sum to the scoreboard total per alliance.

    Args:
        robot_scores:      {team_number: total_score}
        scoreboard_scores: Latest {red_score, blue_score} from read_score().

    Returns:
        {
            red_attributed:  int,
            red_scoreboard:  int,
            red_gap:         int,   # positive = under-attributed
            blue_attributed: int,
            blue_scoreboard: int,
            blue_gap:        int,
            balanced:        bool,
        }
    """
    # Load alliance membership from match_identity.json
    identity_path = Path("configs/match_identity.json")
    red_teams:  set[str] = set()
    blue_teams: set[str] = set()

    if identity_path.exists():
        with open(identity_path) as f:
            identity = json.load(f)
        for robot in identity.get("robots", []):
            if robot.get("alliance") == "red":
                red_teams.add(str(robot["team_number"]))
            elif robot.get("alliance") == "blue":
                blue_teams.add(str(robot["team_number"]))

    red_attr  = sum(v for k, v in robot_scores.items() if k in red_teams)
    blue_attr = sum(v for k, v in robot_scores.items() if k in blue_teams)
    unalloc   = sum(v for k, v in robot_scores.items()
                    if k not in red_teams and k not in blue_teams)

    red_sb  = scoreboard_scores.get("red_score",  0)
    blue_sb = scoreboard_scores.get("blue_score", 0)

    result = {
        "red_attributed":  red_attr,
        "red_scoreboard":  red_sb,
        "red_gap":         red_sb - red_attr,
        "blue_attributed": blue_attr,
        "blue_scoreboard": blue_sb,
        "blue_gap":        blue_sb - blue_attr,
        "unallocated":     unalloc,
        "balanced":        (red_sb == red_attr and blue_sb == blue_attr),
    }

    if not result["balanced"]:
        print("  [Scoreboard] (!) Attribution mismatch:")
        if result["red_gap"] != 0:
            print(f"               Red  : attributed={red_attr}  "
                  f"scoreboard={red_sb}  gap={result['red_gap']}")
        if result["blue_gap"] != 0:
            print(f"               Blue : attributed={blue_attr}  "
                  f"scoreboard={blue_sb}  gap={result['blue_gap']}")
        if unalloc:
            print(f"               Unallocated (unknown alliance): {unalloc}")
    else:
        print("  [Scoreboard] OK Attribution matches scoreboard totals.")

    return result


# ---- Sample a few frames from video to confirm scoreboard location ----------

def sample_scoreboard_frames(
    video_path:    str | Path,
    coords:        list[int] | None = None,
    n_samples:     int = 5,
    output_dir:    str | Path = "data/scoreboard_samples",
) -> list[dict]:
    """
    Extract N frames from the video, locate the scoreboard, and return
    score readings. Used to confirm coords before full processing.

    Args:
        video_path: Path to match video.
        coords:     User-supplied [x1,y1,x2,y2], or None for auto-detect.
        n_samples:  Number of evenly spaced frames to sample.
        output_dir: Where to save debug crops.

    Returns:
        List of {frame_id, bbox, red_score, blue_score, confidence}.
    """
    video_path = Path(video_path)
    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap         = cv2.VideoCapture(str(video_path))
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_ids  = [int(total * i / n_samples) for i in range(n_samples)]
    readings    = []

    for fid in sample_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        bbox = locate_scoreboard(frame, coords)
        if bbox is None:
            readings.append({"frame_id": fid, "bbox": None,
                              "red_score": -1, "blue_score": -1, "confidence": 0.0})
            continue

        score = read_score(frame, bbox)
        readings.append({
            "frame_id":  fid,
            "bbox":      bbox,
            **score,
        })

        # Save debug crop
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        cv2.imwrite(str(out_dir / f"scoreboard_{fid:06d}.jpg"), crop)

    cap.release()

    print(f"  [Scoreboard] Sampled {len(readings)} frames:")
    for r in readings:
        print(f"               frame {r['frame_id']:6d}  "
              f"red={r['red_score']}  blue={r['blue_score']}  "
              f"conf={r.get('confidence', 0):.2f}")

    return readings
