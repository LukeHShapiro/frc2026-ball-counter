"""
src/detect.py - Phase 5: Detection

Primary path: local YOLOv8s models (models/trained/balls/best.pt +
              models/trained/robots/best.pt) via ultralytics.
Fallback path: Roboflow Workflows HTTP API (Python 3.13 compatible,
               no SDK required) -- used when local weights are absent
               and a valid Roboflow API key is configured.

Functions:
  load_roboflow_config()    - load configs/roboflow_config.json
  encode_frame()            - BGR numpy array -> base64 JPEG string
  run_workflow_on_frame()   - POST one frame to the Roboflow workflow endpoint
  parse_predictions()       - normalise workflow response -> [{bbox, confidence, class_name}]
  run_ball_detection()      - detect game pieces in a frame (local or API)
  run_robot_detection()     - detect robots in a frame (local or API)
  read_bumper_number()      - EasyOCR on a cropped robot image -> team number
  process_video()           - stream all frames through detector, collect results
  save_detection_cache()    - persist detection results to data/detections.json

Depends on: Phase 3/4 models OR Roboflow workflow trained.

(!) STOP after this module is integrated --
   detections feed into src/track.py (Phase 5b).
"""

from __future__ import annotations

import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


# ---- Real-time frame preview callback (#12) ---------------------------------
# Set this callable before calling process_video() to receive annotated JPEG
# bytes for each processed batch.  Signature: callback(jpeg_bytes: bytes) -> None
# The GUI worker sets this to emit a pyqtSignal; leave None to disable.
_frame_callback = None   # type: ignore[assignment]


def set_frame_callback(fn) -> None:
    """Register a callable that receives JPEG bytes for each processed batch (#12)."""
    global _frame_callback
    _frame_callback = fn


def clear_frame_callback() -> None:
    """Unregister the frame callback (call after pipeline finishes)."""
    global _frame_callback
    _frame_callback = None


# ---- Config ----------------------------------------------------------------

def load_roboflow_config(config_path: str | Path = "configs/roboflow_config.json") -> dict:
    """
    Load Roboflow API settings from configs/roboflow_config.json.

    Returns:
        Config dict with api_url, api_key, workspace, workflow_id, etc.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Roboflow config not found: {path}\n"
            "Create configs/roboflow_config.json with api_key, workspace, workflow_id."
        )
    with open(path) as f:
        return json.load(f)


# ---- Frame encoding --------------------------------------------------------

def encode_frame(frame: np.ndarray, quality: int = 90) -> str:
    """
    Encode a BGR numpy frame to a base64 JPEG string for the Roboflow API.

    Args:
        frame:   BGR image array (H x W x 3).
        quality: JPEG compression quality (1-100).

    Returns:
        Base64-encoded JPEG string.
    """
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


# ---- Workflow HTTP call -----------------------------------------------------

def run_workflow_on_frame(
    frame:       np.ndarray,
    config:      dict,
    session:     requests.Session | None = None,
    timeout:     int = 30,
) -> dict:
    """
    POST a single frame to the Roboflow Workflows HTTP endpoint.

    Endpoint: POST {api_url}/{workspace}/workflows/{workflow_id}
              ?api_key={api_key}

    Request body:
        {"inputs": {"image": {"type": "base64", "value": "<b64>"}}}

    Args:
        frame:   BGR image array.
        config:  Roboflow config dict (from load_roboflow_config).
        session: Optional requests.Session for connection reuse.
        timeout: Request timeout in seconds.

    Returns:
        Raw workflow response dict (outputs key contains per-step results).

    Raises:
        requests.HTTPError: On 4xx/5xx API responses.
    """
    b64 = encode_frame(frame)

    url = (
        f"{config['api_url'].rstrip('/')}/"
        f"{config['workspace']}/workflows/{config['workflow_id']}"
    )
    params = {"api_key": config["api_key"]}
    body   = {
        "inputs": {
            config.get("image_input_name", "image"): {
                "type":  "base64",
                "value": b64,
            }
        }
    }

    requester = session or requests
    resp = requester.post(url, params=params, json=body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ---- Parse predictions ------------------------------------------------------

def parse_predictions(
    workflow_response: dict,
    config:            dict,
    conf_threshold:    float | None = None,
) -> list[dict]:
    """
    Normalise a Roboflow workflow response into a flat list of detections.

    Each detection dict:
        {
          "bbox":        [x1, y1, x2, y2],   # pixel coords
          "confidence":  float,
          "class_name":  str,
          "class_id":    int,
          "cx":          float,               # centre x
          "cy":          float,               # centre y
          "width":       float,               # bbox width
          "height":      float,               # bbox height
        }

    Args:
        workflow_response: Raw dict returned by run_workflow_on_frame.
        config:            Roboflow config (for output key names).
        conf_threshold:    Min confidence to keep (default: config value).

    Returns:
        List of detection dicts.
    """
    if conf_threshold is None:
        conf_threshold = config.get("confidence_threshold", 0.40)

    detections: list[dict] = []

    # The Roboflow Workflows API returns:
    # {"outputs": [{"predictions": {...}, "output_image": {...}, "count_objects": ...}]}
    outputs = workflow_response.get("outputs", [])
    if not outputs:
        # Fallback: some versions wrap directly
        outputs = [workflow_response]

    predictions_key = config.get("predictions_output", "predictions")

    for output in outputs:
        raw_preds = output.get(predictions_key)
        if raw_preds is None:
            continue

        # Roboflow returns predictions in two formats depending on workflow step type:
        # Format A: {"predictions": [{"x", "y", "width", "height", "confidence", "class", ...}]}
        # Format B: {"value": {"predictions": [...]}}
        if isinstance(raw_preds, dict):
            if "value" in raw_preds:
                raw_preds = raw_preds["value"]
            preds_list = raw_preds.get("predictions", [])
        elif isinstance(raw_preds, list):
            preds_list = raw_preds
        else:
            continue

        for pred in preds_list:
            conf = float(pred.get("confidence", 0.0))
            if conf < conf_threshold:
                continue

            # Roboflow uses centre-x, centre-y, width, height (pixels)
            cx = float(pred.get("x", 0))
            cy = float(pred.get("y", 0))
            w  = float(pred.get("width",  0))
            h  = float(pred.get("height", 0))
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": conf,
                "class_name": str(pred.get("class", "unknown")),
                "class_id":   int(pred.get("class_id", 0)),
                "cx": cx, "cy": cy,
                "width": w, "height": h,
            })

    return detections


def parse_count(workflow_response: dict, config: dict) -> int:
    """
    Extract the object count from a workflow response.

    Returns:
        Integer count of detected objects, or -1 if not present.
    """
    outputs = workflow_response.get("outputs", [{}])
    count_key = config.get("count_output", "count_objects")
    for output in outputs:
        raw = output.get(count_key)
        if raw is None:
            continue
        if isinstance(raw, (int, float)):
            return int(raw)
        if isinstance(raw, dict):
            # {"value": 5} format
            return int(raw.get("value", -1))
    return -1


def parse_annotated_frame(workflow_response: dict, config: dict) -> np.ndarray | None:
    """
    Decode the annotated output image from a workflow response.

    Returns:
        BGR numpy array, or None if not present.
    """
    outputs = workflow_response.get("outputs", [{}])
    img_key = config.get("output_image_name", "output_image")
    for output in outputs:
        raw = output.get(img_key)
        if raw is None:
            continue
        # {"type": "base64", "value": "<b64>"} or just a b64 string
        if isinstance(raw, dict):
            b64 = raw.get("value", "")
        else:
            b64 = str(raw)
        if not b64:
            continue
        try:
            arr = np.frombuffer(base64.b64decode(b64), np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            return None
    return None


# ---- High-level detection helpers ------------------------------------------

def run_ball_detection(
    frame:        np.ndarray,
    config:       dict,
    session:      requests.Session | None = None,
    project_root: str | Path = ".",
) -> list[dict]:
    """
    Detect game pieces (balls/Fuel) in a single frame.

    Uses local YOLO model if weights are present, otherwise Roboflow API.

    Args:
        frame:        BGR image array.
        config:       Roboflow config dict.
        session:      Optional requests.Session (API path only).
        project_root: Project root for locating local weights.

    Returns:
        List of detections [{bbox, confidence, class_name, ...}].
        Filtered to ball-class detections only.
    """
    conf = float(config.get("confidence_threshold", 0.40))
    if _local_models_available(project_root):
        all_dets = _run_local_inference(frame, conf, project_root)
    else:
        response = run_workflow_on_frame(frame, config, session)
        all_dets = parse_predictions(response, config)
    return [d for d in all_dets
            if d["class_name"].lower() in {c.lower() for c in _LOCAL_BALL_CLASSES}]


def run_robot_detection(
    frame:        np.ndarray,
    config:       dict,
    session:      requests.Session | None = None,
    project_root: str | Path = ".",
) -> list[dict]:
    """
    Detect robots in a single frame.

    Uses local YOLO model if weights are present, otherwise Roboflow API.

    Args:
        frame:        BGR image array.
        config:       Roboflow config dict.
        session:      Optional requests.Session (API path only).
        project_root: Project root for locating local weights.

    Returns:
        List of detections [{bbox, confidence, class_name, ...}].
        Filtered to robot-class detections only.
    """
    conf = float(config.get("confidence_threshold", 0.40))
    if _local_models_available(project_root):
        all_dets = _run_local_inference(frame, conf, project_root)
    else:
        response = run_workflow_on_frame(frame, config, session)
        all_dets = parse_predictions(response, config)
    return [d for d in all_dets
            if d["class_name"].lower() in {c.lower() for c in _LOCAL_ROBOT_CLASSES}]


# ---- OCR: bumper number reading ─────────────────────────────────────────────
#
# Multi-strategy approach:
#   • 8 preprocessed image variants per strip (white mask, adaptive thresh, etc.)
#   • Multiple crop zones (bottom 45/55/65/75%, top 25%, full crop)
#   • Primary + secondary OCR engine ensemble (PaddleOCR + EasyOCR)
#   • Common OCR character corrections (I→1, O→0, B→8, etc.)
#   • TBA roster validation with edit-distance-1 fuzzy matching
#   • Sharpness-weighted voting across all readings per track
# ─────────────────────────────────────────────────────────────────────────────

_paddle_ocr  = None
_easy_ocr    = None   # secondary ensemble engine
_ocr_cache: dict[int, dict] = {}
_tba_roster: set[str] = set()   # loaded by set_tba_roster()

_OCR_RULES = {
    "min_frames_for_confidence": 2,    # lower = commit faster
    "majority_vote_threshold":   0.35, # weighted-vote share to accept result
    "fallback_if_unknown":       "UNKNOWN",
    "blur_rejection_threshold":  5,    # Laplacian variance; only reject truly blurry
}


# ── TBA roster validation ─────────────────────────────────────────────────────

def set_tba_roster(teams: list[str]) -> None:
    """
    Populate the global TBA roster used to validate OCR output.

    Call once at pipeline startup after fetching event teams.
    Any OCR reading not in the roster (or within edit-distance 1) is rejected.

    Args:
        teams: List of team number strings, e.g. ["1719", "2377", ...].
    """
    global _tba_roster
    _tba_roster = {str(t) for t in teams if t}
    print(f"  [OCR] TBA roster loaded: {len(_tba_roster)} teams for validation")


def _ocr_char_corrections(text: str) -> str:
    """Apply common OCR character → digit substitutions."""
    _subs = {
        'O': '0', 'o': '0', 'D': '0', 'Q': '0',
        'I': '1', 'l': '1', 'i': '1', '|': '1',
        'B': '8',
        'S': '5', 's': '5',
        'G': '6', 'b': '6',
        'Z': '2', 'z': '2',
        'q': '9', 'g': '9',
        'A': '4',
        ' ': '', '-': '', '.': '',
    }
    return ''.join(_subs.get(c, c) for c in text)


def _is_valid_frc_number(text: str) -> bool:
    """True if text is a plausible FRC team number (1–9999, no leading zeros)."""
    if not text or not text.isdigit() or not (2 <= len(text) <= 4):
        return False
    n = int(text)
    return 1 <= n <= 9999 and not (len(text) > 1 and text[0] == '0')


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (capped at 2 for speed)."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 1:
        return 99
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[n]


def _match_to_roster(text: str) -> str | None:
    """
    Validate a candidate team number against the TBA roster.

    • Exact match → accepted immediately.
    • Edit-distance 1 → corrected to the canonical roster number.
    • No roster loaded → accept any valid FRC number.
    • Returns None if nothing matches.
    """
    if not _is_valid_frc_number(text):
        return None
    if not _tba_roster:
        return text  # no roster; trust FRC range check
    if text in _tba_roster:
        return text
    for team in _tba_roster:
        if _edit_distance(text, team) <= 1:
            return team
    return None


# ── OCR engine loaders ────────────────────────────────────────────────────────

def _get_paddle_ocr():
    """Lazy-load primary OCR engine: PaddleOCR → EasyOCR fallback."""
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            engine = PaddleOCR(use_textline_orientation=False, lang="en",
                               ocr_version="PP-OCRv4", show_log=False)
            import numpy as _np
            engine.ocr(_np.zeros((20, 40, 3), dtype=_np.uint8))
            _paddle_ocr = engine
            print("  [OCR] Primary: PaddleOCR PP-OCRv4")
            return _paddle_ocr
        except Exception:
            pass
        try:
            import torch, easyocr
            _paddle_ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available(),
                                          verbose=False)
            print("  [OCR] Primary: EasyOCR (%s)" %
                  ("GPU" if torch.cuda.is_available() else "CPU"))
        except Exception as exc:
            print(f"  [OCR] (!) No OCR engine available: {exc}")
            _paddle_ocr = None
    return _paddle_ocr


def _get_easy_ocr():
    """Lazy-load secondary EasyOCR for ensemble (skipped if primary is already EasyOCR)."""
    global _easy_ocr
    if _easy_ocr is False:
        return None
    if _easy_ocr is None:
        # Don't load secondary if primary is already EasyOCR
        try:
            import easyocr as _emod
            if isinstance(_paddle_ocr, _emod.Reader):
                _easy_ocr = False
                return None
        except ImportError:
            pass
        try:
            import torch, easyocr
            _easy_ocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available(),
                                        verbose=False)
            print("  [OCR] Secondary ensemble: EasyOCR")
        except Exception:
            _easy_ocr = False
    return _easy_ocr if _easy_ocr is not False else None


def _run_engine(engine, crop: np.ndarray) -> list[tuple[str, float]]:
    """
    Run one OCR engine and return (raw_text, confidence) pairs.
    Accepts strings 2–5 chars long (character correction happens later).
    """
    if engine is None or crop is None or crop.size == 0:
        return []
    results: list[tuple[str, float]] = []
    try:
        import easyocr as _emod
        if isinstance(engine, _emod.Reader):
            # allowlist broadened to include common OCR confusables
            raw = engine.readtext(
                crop, detail=1,
                allowlist="0123456789OIlBSGZbqgsDQo ")
            for (_, text, conf) in raw:
                t = text.strip().replace(" ", "")
                if 2 <= len(t) <= 5:
                    results.append((t, float(conf)))
            return results
    except (ImportError, Exception):
        pass
    # PaddleOCR 3.x
    try:
        result = engine.ocr(crop)
        if not result:
            return results
        rows = result[0] if (result and isinstance(result[0], list)) else result
        for item in (rows or []):
            if item is None:
                continue
            try:
                text = item[1][0].strip().replace(" ", "")
                conf = float(item[1][1])
            except (IndexError, TypeError, ValueError):
                continue
            if 2 <= len(text) <= 5:
                results.append((text, conf))
    except Exception:
        pass
    return results


# ── Image preprocessing pipeline ─────────────────────────────────────────────

def _bumper_strips(robot_crop: np.ndarray) -> list[np.ndarray]:
    """
    Slice the robot bounding box into candidate bumper strips.

    Tries bottom (most common), top (some robots), and full crop.
    """
    h, w = robot_crop.shape[:2]
    strips: list[np.ndarray] = []
    # Bottom strips at multiple heights
    for frac in (0.55, 0.65, 0.72, 0.80, 0.42):
        strip = robot_crop[int(h * frac):, :]
        if strip.shape[0] >= 6 and strip.shape[1] >= 10:
            strips.append(strip)
    # Top strips
    for frac in (0.0, 0.08):
        strip = robot_crop[int(h * frac): int(h * (frac + 0.28)), :]
        if strip.shape[0] >= 6 and strip.shape[1] >= 10:
            strips.append(strip)
    # Full crop as last resort
    if robot_crop.shape[0] >= 6 and robot_crop.shape[1] >= 10:
        strips.append(robot_crop)
    return strips


def _preprocess_strip(strip: np.ndarray) -> list[np.ndarray]:
    """
    Generate 8 preprocessed variants of a bumper strip.

    Each variant targets a different visual characteristic:
      V1  4× CLAHE          — general contrast normalisation
      V2  6× CLAHE          — more pixels for small strips
      V3  white-text mask   — white digits on colored bumper → black on white
      V4  yellow-text mask  — yellow digits on dark bumper
      V5  adaptive thresh   — works well for lit text
      V6  inverted thresh   — handles dark-on-light
      V7  Otsu threshold    — bimodal intensity split
      V8  unsharp + gray    — sharpened grayscale

    Returns list of BGR images ready for OCR.
    """
    if strip is None or strip.size == 0:
        return []
    h, w = strip.shape[:2]
    if h < 4 or w < 4:
        return []

    def _up(img, s):
        return cv2.resize(img, (img.shape[1] * s, img.shape[0] * s),
                          interpolation=cv2.INTER_CUBIC)

    def _clahe_bgr(bgr):
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        ll, a, b = cv2.split(lab)
        ll = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(ll)
        return cv2.cvtColor(cv2.merge([ll, a, b]), cv2.COLOR_LAB2BGR)

    up4 = _up(strip, 4)
    hsv4 = cv2.cvtColor(up4, cv2.COLOR_BGR2HSV)
    gray4 = cv2.cvtColor(up4, cv2.COLOR_BGR2GRAY)

    variants: list[np.ndarray] = []

    # V1: 4× CLAHE
    variants.append(_clahe_bgr(up4))

    # V2: 6× CLAHE
    variants.append(_clahe_bgr(_up(strip, 6)))

    # V3: white-text mask (white digits on colored bumper)
    white_mask = cv2.inRange(hsv4, np.array([0, 0, 165]), np.array([180, 60, 255]))
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    white_mask = cv2.dilate(white_mask, kern, iterations=1)
    variants.append(cv2.cvtColor(cv2.bitwise_not(white_mask), cv2.COLOR_GRAY2BGR))

    # V4: yellow-text mask
    yellow_mask = cv2.inRange(hsv4, np.array([18, 80, 150]), np.array([38, 255, 255]))
    variants.append(cv2.cvtColor(cv2.bitwise_not(yellow_mask), cv2.COLOR_GRAY2BGR))

    # V5: adaptive threshold on sharpened gray
    blur = cv2.GaussianBlur(gray4, (0, 0), 1.5)
    sharp = cv2.addWeighted(gray4, 2.0, blur, -1.0, 0)
    thresh = cv2.adaptiveThreshold(sharp, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 15, 4)
    variants.append(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # V6: inverted adaptive threshold
    variants.append(cv2.cvtColor(cv2.bitwise_not(thresh), cv2.COLOR_GRAY2BGR))

    # V7: Otsu threshold
    _, otsu = cv2.threshold(
        cv2.GaussianBlur(gray4, (3, 3), 0), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

    # V8: unsharp + grayscale
    blur8 = cv2.GaussianBlur(gray4, (0, 0), 2)
    unsharp = cv2.addWeighted(gray4, 1.5, blur8, -0.5, 0)
    variants.append(cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR))

    return variants


# ── Main OCR entry point ──────────────────────────────────────────────────────

def read_bumper_number(
    robot_crop: np.ndarray,
    track_id:   int,
) -> str:
    """
    Multi-strategy bumper OCR with TBA roster validation.

    Per crop:
      1. Slice 7 strip positions (bottom at 5 heights, top, full).
      2. Generate 8 preprocessed variants per strip.
      3. Run primary + secondary OCR engines on each variant.
      4. Apply character corrections (I→1, O→0, B→8 …).
      5. Fuzzy-match each reading against TBA roster (edit-dist ≤ 1).
      6. Accumulate sharpness-weighted vote per validated team number.

    Returns team number string when weighted confidence is sufficient,
    otherwise "UNKNOWN_{track_id}".

    Args:
        robot_crop: BGR crop of full robot bounding box.
        track_id:   DeepSORT track ID (key for caching across frames).
    """
    primary   = _get_paddle_ocr()
    secondary = _get_easy_ocr()

    if track_id not in _ocr_cache:
        _ocr_cache[track_id] = {
            "weighted_counts": {},
            "n_reads": 0,
            "team_number": None,
            "confidence": 0.0,
        }
    cache = _ocr_cache[track_id]

    h, w = robot_crop.shape[:2]
    if h < 20 or w < 20:
        return cache["team_number"] or f"UNKNOWN_{track_id}"

    # Overall crop sharpness (scales vote weights)
    gray_full = cv2.cvtColor(robot_crop, cv2.COLOR_BGR2GRAY)
    crop_sharpness = max(1.0, float(cv2.Laplacian(gray_full, cv2.CV_64F).var()))

    strips = _bumper_strips(robot_crop)
    new_hits = 0

    for strip in strips:
        sg = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        strip_sharp = float(cv2.Laplacian(sg, cv2.CV_64F).var())
        if strip_sharp < _OCR_RULES["blur_rejection_threshold"]:
            continue

        for variant in _preprocess_strip(strip):
            for engine in (primary, secondary):
                if engine is None:
                    continue
                for raw_text, ocr_conf in _run_engine(engine, variant):
                    # Try corrected and original
                    for candidate in (
                        _ocr_char_corrections(raw_text),
                        raw_text,
                    ):
                        if not candidate.isdigit() or len(candidate) < 2:
                            continue
                        matched = _match_to_roster(candidate)
                        if matched is not None:
                            weight = strip_sharp * max(0.1, ocr_conf)
                            cache["weighted_counts"][matched] = (
                                cache["weighted_counts"].get(matched, 0.0) + weight
                            )
                            new_hits += 1

    cache["n_reads"] += 1
    wc = cache["weighted_counts"]
    if not wc:
        return cache["team_number"] or f"UNKNOWN_{track_id}"

    total = sum(wc.values())
    best  = max(wc, key=wc.get)
    conf  = wc[best] / total if total > 0 else 0.0

    # Commit result if confidence threshold met
    if cache["n_reads"] >= _OCR_RULES["min_frames_for_confidence"]:
        if conf >= _OCR_RULES["majority_vote_threshold"]:
            cache["team_number"] = best
            cache["confidence"]  = conf
            return best

    # Soft fallback: return leading candidate even below threshold
    # (build_robot_identity_map will do its own final voting)
    if new_hits > 0 or cache["n_reads"] >= 2:
        return best

    return cache["team_number"] or f"UNKNOWN_{track_id}"


def clear_ocr_cache(track_id: int | None = None) -> None:
    """
    Clear the per-track OCR cache.

    Args:
        track_id: Specific track to clear, or None to clear all.
    """
    global _ocr_cache
    if track_id is None:
        _ocr_cache.clear()
    else:
        _ocr_cache.pop(track_id, None)


# ---- Local YOLO inference ---------------------------------------------------

_LOCAL_BALL_CLASSES  = {"Fuel", "fuel", "ball", "Ball", "game_piece"}
_LOCAL_ROBOT_CLASSES = {"robot", "Robot", "Blue_Robot", "Red_Robot",
                        "blue_robot", "red_robot"}

_local_ball_model  = None
_local_robot_model = None


_YOLO_VARIANT = "yolov8m.pt"   # medium model — better accuracy than yolov8s

# Separate confidence thresholds (#9)
# Ball threshold raised: ground balls get weak detections; in-flight balls score higher
_BALL_CONF  = 0.50
_ROBOT_CONF = 0.50
# NMS IoU threshold: lower = fewer duplicate robot detections (#10)
_NMS_IOU    = 0.35


def _load_local_models(project_root: str | Path = ".") -> tuple:
    """
    Lazy-load local YOLOv8m ball and robot models.
    Prefers TensorRT .engine files for maximum GPU throughput (#1).

    Returns:
        (ball_model, robot_model) -- either may be None if weights absent.
    """
    global _local_ball_model, _local_robot_model

    root = Path(project_root)
    # Prefer TensorRT engine if available, fall back to .pt (#1)
    ball_weights  = (root / "models" / "trained" / "balls"  / "best.engine"
                     if (root / "models" / "trained" / "balls"  / "best.engine").exists()
                     else root / "models" / "trained" / "balls"  / "best.pt")
    robot_weights = (root / "models" / "trained" / "robots" / "best.engine"
                     if (root / "models" / "trained" / "robots" / "best.engine").exists()
                     else root / "models" / "trained" / "robots" / "best.pt")

    if _local_ball_model is None and ball_weights.exists():
        try:
            from ultralytics import YOLO
            _local_ball_model = YOLO(str(ball_weights))
            fmt = "TensorRT" if str(ball_weights).endswith(".engine") else "PyTorch"
            print(f"  [Detect] Loaded ball model ({fmt}): {ball_weights}")
        except Exception as exc:
            print(f"  [Detect] (!) Could not load ball model: {exc}")

    if _local_robot_model is None and robot_weights.exists():
        try:
            from ultralytics import YOLO
            _local_robot_model = YOLO(str(robot_weights))
            fmt = "TensorRT" if str(robot_weights).endswith(".engine") else "PyTorch"
            print(f"  [Detect] Loaded robot model ({fmt}): {robot_weights}")
        except Exception as exc:
            print(f"  [Detect] (!) Could not load robot model: {exc}")

    return _local_ball_model, _local_robot_model


def export_tensorrt(project_root: str | Path = ".") -> None:
    """
    Export trained YOLO models to TensorRT .engine format (#1).
    Run once after training — subsequent pipeline runs auto-detect and use .engine files.

    Requires: TensorRT installed (ships with CUDA toolkit).
    """
    root = Path(project_root)
    for name in ("balls", "robots"):
        pt_path = root / "models" / "trained" / name / "best.pt"
        eng_path = root / "models" / "trained" / name / "best.engine"
        if not pt_path.exists():
            print(f"  [TRT] No weights found at {pt_path} - skipping.")
            continue
        if eng_path.exists():
            print(f"  [TRT] Engine already exists: {eng_path} - skipping.")
            continue
        try:
            from ultralytics import YOLO
            model = YOLO(str(pt_path))
            print(f"  [TRT] Exporting {name} model to TensorRT (half=True)...")
            model.export(format="engine", half=True, device=0)
            print(f"  [TRT] Done: {eng_path}")
        except Exception as exc:
            print(f"  [TRT] Export failed for {name}: {exc}")


def _yolo_results_to_dets(yolo_result, conf_threshold: float = 0.40) -> list[dict]:
    """
    Convert a single ultralytics Results object to the standard detection format.

    Returns:
        List of detection dicts: {bbox, confidence, class_name, class_id, cx, cy, width, height}
    """
    dets: list[dict] = []
    if yolo_result is None or yolo_result.boxes is None:
        return dets

    boxes  = yolo_result.boxes
    names  = yolo_result.names  # {class_id: class_name}

    for i in range(len(boxes)):
        conf = float(boxes.conf[i])
        if conf < conf_threshold:
            continue
        cls_id = int(boxes.cls[i])
        cls_name = names.get(cls_id, str(cls_id))

        # xyxy format
        x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1

        dets.append({
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(conf, 4),
            "class_name": cls_name,
            "class_id":   cls_id,
            "cx": cx, "cy": cy,
            "width": w, "height": h,
        })
    return dets


def _run_local_inference(
    frame: np.ndarray,
    conf_threshold: float = 0.40,
    project_root: str | Path = ".",
) -> list[dict]:
    """
    Run both local YOLO models on a single frame (used by run_*_detection helpers).
    For bulk video processing, use _run_batched_inference() instead.
    """
    return _run_batched_inference([frame], conf_threshold, project_root)[0]


def _run_batched_inference(
    frames: list[np.ndarray],
    conf_threshold: float = 0.40,   # kept for API compat; per-model thresholds used internally
    project_root: str | Path = ".",
) -> list[list[dict]]:
    """
    Run both YOLO models on a batch of frames in one GPU call per model.

    Upgrades applied:
      - FP16 half-precision inference (#2) — ~40% faster on RTX GPUs
      - Separate ball/robot conf thresholds (#9) — 0.35 balls / 0.50 robots
      - NMS IoU=0.35 (#10) — fewer duplicate detections on overlapping robots
    """
    ball_model, robot_model = _load_local_models(project_root)
    n = len(frames)
    per_frame: list[list[dict]] = [[] for _ in range(n)]

    import torch
    use_half = torch.cuda.is_available()  # FP16 only on GPU (#2)

    if ball_model is not None:
        try:
            batch_results = ball_model(
                frames, verbose=False,
                conf=_BALL_CONF, iou=_NMS_IOU,   # #9, #10
                half=use_half,                    # #2
                stream=True,
            )
            for i, r in enumerate(batch_results):
                if i < n:
                    per_frame[i].extend(_yolo_results_to_dets(r, _BALL_CONF))
        except Exception as exc:
            print(f"  [Detect] (!) Ball batch failed: {exc}")

    if robot_model is not None:
        try:
            batch_results = robot_model(
                frames, verbose=False,
                conf=_ROBOT_CONF, iou=_NMS_IOU,  # #9, #10
                half=use_half,                    # #2
                stream=True,
            )
            for i, r in enumerate(batch_results):
                if i < n:
                    robot_dets = _yolo_results_to_dets(r, _ROBOT_CONF)
                    for d in robot_dets:
                        if d["class_name"].lower() not in {c.lower() for c in _LOCAL_BALL_CLASSES}:
                            d["class_name"] = "robot"
                    per_frame[i].extend(robot_dets)
        except Exception as exc:
            print(f"  [Detect] (!) Robot batch failed: {exc}")

    return per_frame


def _local_models_available(project_root: str | Path = ".") -> bool:
    """Return True if at least one local YOLO model weight file exists."""
    root = Path(project_root)
    ball_w  = root / "models" / "trained" / "balls"  / "best.pt"
    robot_w = root / "models" / "trained" / "robots" / "best.pt"
    return ball_w.exists() or robot_w.exists()


# ---- GPU frame decoding (#6) ------------------------------------------------

def _try_nvdec_reader(
    video_path:     Path,
    sample_every_n: int,
    source_fps:     float,
) -> list[tuple[int, np.ndarray, float]] | None:
    """
    Read sampled video frames using ffmpeg NVIDIA hardware decode (NVDEC) (#6).

    Decoding on the GPU is ~2× faster than cv2.VideoCapture for 1080p/4K video,
    which reduces the frame-collection phase before batched YOLO inference.

    Falls back gracefully (returns None) if:
      - ffmpeg is not in PATH
      - NVDEC is unavailable (no NVIDIA GPU or driver too old)
      - Any other error occurs

    The caller should then fall back to cv2.VideoCapture.

    Args:
        video_path:     Path to the video file.
        sample_every_n: Take every N-th frame.
        source_fps:     Frames-per-second of the source video.

    Returns:
        List of (frame_id, BGR ndarray, timestamp_ms) tuples, or None on failure.
    """
    import subprocess
    import json as _json

    # Step 1: probe for width/height
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams", "-select_streams", "v:0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        info   = _json.loads(probe.stdout)
        stream = info["streams"][0]
        width  = int(stream["width"])
        height = int(stream["height"])
    except Exception:
        return None   # ffprobe not available or can't parse

    frame_size = width * height * 3

    # Step 2: decode with NVDEC; select every N-th frame on the CPU side.
    # -hwaccel cuda decodes on GPU; frames are transferred to CPU before the
    # select filter runs, so we still avoid software codec overhead.
    select_expr = f"not(mod(n\\,{sample_every_n}))" if sample_every_n > 1 else "1"
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", str(video_path),
        "-vf", f"select='{select_expr}'",
        "-vsync", "0",
        "-pix_fmt", "bgr24",
        "-f", "rawvideo",
        "pipe:1",
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=frame_size * 4,
        )
    except FileNotFoundError:
        return None   # ffmpeg not installed

    frames: list[tuple[int, np.ndarray, float]] = []
    sample_idx = 0  # counts selected frames (0, 1, 2, ...)

    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            arr    = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            orig_fid = sample_idx * sample_every_n
            ts_ms    = (orig_fid / source_fps) * 1000.0
            frames.append((orig_fid, arr.copy(), ts_ms))
            sample_idx += 1
    except Exception:
        proc.kill()
        return None
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()

    if not frames:
        return None   # NVDEC failed silently — fall back to cv2

    return frames


# ---- Full video processing --------------------------------------------------

def process_video(
    video_path:     str | Path,
    config:         dict,
    sample_every_n: int   = 1,
    max_workers:    int   = 4,
    save_annotated: bool  = False,
    output_dir:     str | Path | None = None,
    progress:       bool  = True,
    project_root:   str | Path = ".",
) -> list[dict]:
    """
    Stream all frames of a video through the detector.

    Detection backend selection (automatic):
      1. Local YOLO models (models/trained/balls/best.pt +
         models/trained/robots/best.pt) -- used if weights exist.
      2. Roboflow Workflows HTTP API -- fallback when local weights are absent
         and config["api_key"] is a valid (non-placeholder) key.

    For each sampled frame returns:
        {
            frame_id:        int,
            timestamp_ms:    float,
            detections:      list[dict],   # standard detection dicts
            count:           int,
            annotated_frame: np.ndarray or None,
        }

    Args:
        video_path:      Path to .mp4 / .mov / .avi.
        config:          Roboflow config dict (from load_roboflow_config).
        sample_every_n:  Process every N-th frame (1 = all frames).
        max_workers:     Parallel workers (threads for API; processes for local).
        save_annotated:  If True, save annotated frames to output_dir.
        output_dir:      Where to save annotated frames (API path only).
        progress:        Print progress every 50 frames.
        project_root:    Project root directory (for locating model weights).

    Returns:
        List of per-frame result dicts, sorted by frame_id.
    """
    video_path   = Path(video_path)
    project_root = Path(project_root)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if save_annotated and output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Choose backend
    use_local = _local_models_available(project_root)
    api_key   = config.get("api_key", "")
    use_api   = (not use_local) and bool(api_key) and api_key != "YOUR_ROBOFLOW_KEY_HERE"

    if use_local:
        backend = "local YOLO"
        # Pre-load models before starting (avoids repeated loads in threads)
        _load_local_models(project_root)
    elif use_api:
        backend = "Roboflow API"
    else:
        backend = "none (no local weights and no valid API key)"

    conf_threshold = float(config.get("confidence_threshold", 0.40))

    print(f"  [Detect] Video: {video_path.name}  |  "
          f"{total_frames} frames  |  {source_fps:.1f} fps")
    print(f"  [Detect] Backend: {backend}  |  "
          f"Sampling every {sample_every_n} frame(s)")

    # Collect (frame_id, frame, timestamp_ms) tuples
    # Attempt GPU-accelerated decode via ffmpeg NVDEC (#6); fall back to cv2.
    frames_to_process: list[tuple[int, np.ndarray, float]] = []

    nvdec_frames = _try_nvdec_reader(video_path, sample_every_n, source_fps)
    if nvdec_frames is not None:
        frames_to_process = nvdec_frames
        cap.release()
        print(f"  [Detect] Frame decode: ffmpeg NVDEC  ({len(frames_to_process)} frames selected)")
    else:
        # cv2 fallback
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_every_n == 0:
                ts_ms = (frame_idx / source_fps) * 1000.0
                frames_to_process.append((frame_idx, frame.copy(), ts_ms))
            frame_idx += 1
        cap.release()
        print(f"  [Detect] Frame decode: cv2.VideoCapture  ({len(frames_to_process)} frames selected)")

    print(f"  [Detect] Processing {len(frames_to_process)} frames...")

    if not use_local and not use_api:
        print("  [Detect] (!) No detection backend available - returning empty results.")
        return [
            {"frame_id": fid, "timestamp_ms": ts,
             "detections": [], "count": 0, "annotated_frame": None}
            for fid, _, ts in frames_to_process
        ]

    # Checkpoint/resume (#7): load partial results if they exist
    ckpt_path = project_root / "data" / "detections_checkpoint.json"
    results: list[dict] = []
    completed_fids: set[int] = set()
    if ckpt_path.exists():
        try:
            ckpt_raw = json.loads(ckpt_path.read_text())
            results = ckpt_raw if isinstance(ckpt_raw, list) else ckpt_raw.get("frames", [])
            completed_fids = {r["frame_id"] for r in results}
            print(f"  [Detect] Checkpoint found - resuming from {len(completed_fids)} "
                  f"already-processed frames.")
            frames_to_process = [(fid, frm, ts) for fid, frm, ts in frames_to_process
                                 if fid not in completed_fids]
        except Exception:
            results = []

    CKPT_INTERVAL = 200   # save checkpoint every N frames

    t0 = time.time()

    # ---- Local YOLO path: batched GPU inference with pipeline threading --------
    if use_local:
        # Batching sends N frames to the GPU in one call, maximising GPU occupancy.
        # A producer thread pre-loads the next batch from disk while the GPU
        # processes the current one, hiding frame decode latency.
        BATCH_SIZE = 32   # RTX 5090 has 25.7 GB VRAM — 32 saturates GPU without OOM

        # Producer: fill a queue with batches of (meta, frames)
        batch_q: "queue.Queue[list | None]" = queue.Queue(maxsize=4)

        def _producer():
            try:
                items = frames_to_process
                for start in range(0, len(items), BATCH_SIZE):
                    batch_q.put(items[start : start + BATCH_SIZE])
            except Exception as exc:
                print(f"  [Detect] (!) Producer thread error: {exc}")
            finally:
                batch_q.put(None)   # always send sentinel so consumer doesn't block

        prod_thread = threading.Thread(target=_producer, daemon=True)
        prod_thread.start()

        done = 0
        while True:
            batch = batch_q.get()
            if batch is None:
                break
            batch_frames = [frm for _, frm, _ in batch]
            try:
                batch_dets = _run_batched_inference(batch_frames, conf_threshold, project_root)
            except Exception as exc:
                print(f"  [Detect] (!) Batch failed: {exc}")
                batch_dets = [[] for _ in batch]

            preview_frame: np.ndarray | None = None
            for (fid, raw_frm, ts), dets in zip(batch, batch_dets):
                results.append({
                    "frame_id":        fid,
                    "timestamp_ms":    ts,
                    "detections":      dets,
                    "count":           len(dets),
                    "annotated_frame": None,
                })
                done += 1
                if _frame_callback is not None and preview_frame is None:
                    preview_frame = raw_frm   # use first frame of batch for annotation

            # Emit annotated preview for the GUI (#12)
            if _frame_callback is not None and preview_frame is not None:
                try:
                    # Draw detection boxes on a copy of the frame
                    annotated = preview_frame.copy()
                    last_batch_dets = batch_dets[0] if batch_dets else []
                    for det in last_batch_dets:
                        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                        is_robot = det["class_name"].lower() in {"robot", "blue_robot", "red_robot"}
                        color = (0, 120, 255) if is_robot else (0, 255, 80)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, det["class_name"][:6],
                                    (x1, max(y1 - 4, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                                    cv2.LINE_AA)
                    # Resize to max 640 wide to keep JPEG small
                    h, w = annotated.shape[:2]
                    if w > 640:
                        scale = 640 / w
                        annotated = cv2.resize(annotated,
                                               (640, int(h * scale)),
                                               interpolation=cv2.INTER_AREA)
                    _, buf = cv2.imencode(".jpg", annotated,
                                         [cv2.IMWRITE_JPEG_QUALITY, 70])
                    _frame_callback(buf.tobytes())
                except Exception:
                    pass   # never let preview crash the pipeline

            if progress:
                elapsed = time.time() - t0
                fps_est = done / elapsed if elapsed > 0 else 0
                total_so_far = len(completed_fids) + done
                grand_total  = len(completed_fids) + len(frames_to_process)
                print(f"  [Detect] {total_so_far}/{grand_total} frames  "
                      f"({fps_est:.1f} frames/s)")

            # Checkpoint save every CKPT_INTERVAL frames (#7)
            if done % CKPT_INTERVAL == 0:
                try:
                    ckpt_path.write_text(json.dumps(results))
                except Exception:
                    pass

        prod_thread.join()
        # Remove checkpoint file on successful completion
        if ckpt_path.exists():
            ckpt_path.unlink(missing_ok=True)

    # ---- Roboflow API path (parallel HTTP) -------------------------------------
    else:
        def _process_one_api(item: tuple[int, np.ndarray, float]) -> dict:
            fid, frm, ts = item
            with requests.Session() as s:
                try:
                    response  = run_workflow_on_frame(frm, config, s)
                    dets      = parse_predictions(response, config, conf_threshold)
                    count     = parse_count(response, config)
                    annotated = parse_annotated_frame(response, config) if save_annotated else None
                except Exception as exc:
                    print(f"  [Detect] (!) Frame {fid} failed: {exc}")
                    return {
                        "frame_id": fid, "timestamp_ms": ts,
                        "detections": [], "count": 0, "annotated_frame": None,
                        "error": str(exc),
                    }
            if save_annotated and annotated is not None and output_dir:
                cv2.imwrite(
                    str(Path(output_dir) / f"annotated_{fid:06d}.jpg"),
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
            return {
                "frame_id":        fid,
                "timestamp_ms":    ts,
                "detections":      dets,
                "count":           count,
                "annotated_frame": annotated,
            }

        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one_api, item): item[0]
                       for item in frames_to_process}
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if progress and done % 50 == 0:
                    elapsed = time.time() - t0
                    fps_est = done / elapsed if elapsed > 0 else 0
                    print(f"  [Detect] {done}/{len(frames_to_process)} frames  "
                          f"({fps_est:.1f} frames/s)")

    results.sort(key=lambda r: r["frame_id"])
    elapsed    = time.time() - t0
    total_dets = sum(len(r["detections"]) for r in results)
    print(f"  [Detect] Done: {len(results)} frames in {elapsed:.1f}s  |  "
          f"{total_dets} total detections  |  backend: {backend}")

    return results


# ---- Cache detections to disk -----------------------------------------------

def _cache_signature(video_path: str | Path, project_root: str | Path = ".") -> str:
    """
    Build a short hash string from video file size + mtime + model mtimes (#13).
    If any of these change the cache is considered stale.
    """
    import hashlib
    vp = Path(video_path)
    parts = []
    if vp.exists():
        st = vp.stat()
        parts.append(f"{st.st_size}:{st.st_mtime}")
    root = Path(project_root)
    for name in ("balls", "robots"):
        for ext in ("engine", "pt"):
            wp = root / "models" / "trained" / name / f"best.{ext}"
            if wp.exists():
                parts.append(f"{wp}:{wp.stat().st_mtime}")
                break
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def save_detection_cache(
    results:      list[dict],
    out_path:     str | Path = "data/detections.json",
    video_path:   str | Path | None = None,
    project_root: str | Path = ".",
) -> Path:
    """
    Persist detection results to disk with a cache-version signature (#13).

    Args:
        results:      List of per-frame dicts from process_video().
        out_path:     Destination JSON path.
        video_path:   Source video (used to compute cache signature).
        project_root: Project root (used to hash model weights).

    Returns:
        Path to written file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sig = _cache_signature(video_path, project_root) if video_path else ""

    serialisable = []
    for r in results:
        serialisable.append({
            "frame_id":     r["frame_id"],
            "timestamp_ms": r["timestamp_ms"],
            "detections":   r["detections"],
            "count":        r["count"],
            "error":        r.get("error"),
        })

    payload = {"_cache_sig": sig, "frames": serialisable}
    with open(out_path, "w") as f:
        json.dump(payload, f)

    print(f"  [Detect] Detection cache saved -> {out_path} "
          f"({len(serialisable)} frames, sig={sig})")
    return out_path


def load_detection_cache(
    path:         str | Path = "data/detections.json",
    video_path:   str | Path | None = None,
    project_root: str | Path = ".",
) -> list[dict]:
    """
    Load a previously saved detection cache (#13).
    Warns if the cache signature doesn't match the current video/model.

    Returns:
        List of per-frame detection dicts.
    """
    with open(path) as f:
        raw = json.load(f)

    # Support both old format (plain list) and new format (dict with _cache_sig)
    if isinstance(raw, list):
        return raw

    if video_path:
        stored_sig = raw.get("_cache_sig", "")
        current_sig = _cache_signature(video_path, project_root)
        if stored_sig and stored_sig != current_sig:
            print(f"  [Detect] (!) Cache signature mismatch "
                  f"(stored={stored_sig}, current={current_sig}). "
                  f"Video or model may have changed — consider re-running detection.")

    return raw.get("frames", [])


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Phase 5 -- Detection via Roboflow")
    parser.add_argument("video", help="Path to match video (.mp4/.mov/.avi)")
    parser.add_argument("--config", default="configs/roboflow_config.json")
    parser.add_argument("--sample", type=int, default=3,
                        help="Process every N-th frame (default: 3)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--out", default="data/detections.json")
    args = parser.parse_args()

    cfg     = load_roboflow_config(args.config)
    results = process_video(
        args.video, cfg,
        sample_every_n = args.sample,
        max_workers    = args.workers,
        save_annotated = args.save_annotated,
        output_dir     = "data/annotated_frames" if args.save_annotated else None,
    )
    save_detection_cache(results, args.out)
    print(f"\nPhase 5 detection complete. Results -> {args.out}")
    sys.exit(0)
