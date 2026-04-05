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


# ---- OCR: bumper number reading (PaddleOCR) ---------------------------------

_paddle_ocr  = None
_ocr_cache: dict[int, dict] = {}    # track_id -> {team_number, confidence, frames}
_OCR_RULES = {
    "min_frames_for_confidence": 5,
    "majority_vote_threshold":   0.55,
    "fallback_if_unknown":       "UNKNOWN",
    "blur_rejection_threshold":  20,
}


def _get_paddle_ocr():
    """
    Lazy-load OCR engine.

    Tries PaddleOCR 3.x first (faster on compatible systems).
    Falls back to EasyOCR (GPU) which is stable on all platforms.
    """
    global _paddle_ocr
    if _paddle_ocr is None:
        # Try PaddleOCR 3.x
        try:
            from paddleocr import PaddleOCR
            # Use ocr_version PP-OCRv4 (mobile/standard) — much faster load than server
            engine = PaddleOCR(
                use_textline_orientation=False,
                lang="en",
                ocr_version="PP-OCRv4",   # lightweight, fast
                show_log=False,
            )
            # Smoke-test
            import numpy as _np
            _blank = _np.zeros((20, 40, 3), dtype=_np.uint8)
            engine.ocr(_blank)
            _paddle_ocr = engine
            print("  [OCR] Using PaddleOCR (PP-OCRv4, fast)")
            return _paddle_ocr
        except Exception:
            pass  # fall through to EasyOCR

        # Fallback: EasyOCR with GPU
        try:
            import torch, easyocr
            _paddle_ocr = easyocr.Reader(
                ["en"], gpu=torch.cuda.is_available(), verbose=False)
            print("  [OCR] Using EasyOCR (GPU)" if torch.cuda.is_available()
                  else "  [OCR] Using EasyOCR (CPU)")
        except Exception as e:
            print(f"  [OCR] (!) No OCR engine available: {e}")
            _paddle_ocr = None
    return _paddle_ocr


def _paddle_readtext(ocr_engine, crop: np.ndarray) -> list[str]:
    """
    Run OCR on a crop and return 3-4 digit candidate strings.
    Handles both EasyOCR (Reader) and PaddleOCR 3.x engines.
    """
    if ocr_engine is None:
        return []
    try:
        import easyocr as _easyocr_mod
        if isinstance(ocr_engine, _easyocr_mod.Reader):
            results = ocr_engine.readtext(crop, detail=0, allowlist="0123456789")
            return [r.strip() for r in results
                    if r.strip().isdigit() and 3 <= len(r.strip()) <= 4]
    except ImportError:
        pass

    try:
        # PaddleOCR 3.x — result is a list of OCRResult objects or nested lists
        result = ocr_engine.ocr(crop)
        candidates: list[str] = []
        if not result:
            return candidates
        # Flatten: result may be [[lines]] or a list of OCRResult
        rows = result[0] if isinstance(result[0], list) else result
        for item in rows:
            if item is None:
                continue
            # Each item: [ [box_points], (text, conf) ]
            try:
                text = item[1][0].strip().replace(" ", "")
            except (IndexError, TypeError):
                continue
            if text.isdigit() and 3 <= len(text) <= 4:
                candidates.append(text)
        return candidates
    except Exception:
        return []


def _preprocess_bumper_crop(crop: np.ndarray, zone_start: float = 0.60) -> np.ndarray:
    """
    Prepare a robot bounding-box crop for OCR (#3).

    Steps:
      1. Take the bottom section of the bbox (zone_start–100%) — bumper location.
      2. Upscale 4× with bicubic interpolation so digits are large enough for OCR.
      3. Apply CLAHE to normalise contrast (handles bright/dark field lighting).

    Args:
        zone_start: Fraction of bbox height to start the bumper strip (default 0.60
                    = bottom 40%).  Callers can try multiple values.

    Returns:
        Preprocessed BGR image ready for OCR.
    """
    h, w = crop.shape[:2]
    bumper = crop[int(h * zone_start):, :]
    if bumper.shape[0] < 4:
        bumper = crop
    # Upscale 4×
    up = cv2.resize(bumper, (bumper.shape[1] * 4, bumper.shape[0] * 4),
                    interpolation=cv2.INTER_CUBIC)
    # CLAHE on L channel
    lab  = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    up = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return up


def read_bumper_number(
    robot_crop: np.ndarray,
    track_id:   int,
) -> str:
    """
    Run OCR on a cropped robot image to read the bumper team number (#3).

    Preprocessing: bottom-30% crop → 3× upscale → CLAHE before OCR.
    Caches results per track_id — does NOT re-OCR every frame.

    Args:
        robot_crop: BGR crop of the full robot bounding box.
        track_id:   Track ID for this robot.

    Returns:
        Team number string (e.g. "1234") or "UNKNOWN_{track_id}".
    """
    # Try multiple bumper crop zones (bottom 40%, 30%, 20%) — pick first with OCR hit
    ocr_engine = _get_paddle_ocr()
    candidates: list[str] = []
    for zone_start in (0.60, 0.70, 0.80, 0.50):
        processed = _preprocess_bumper_crop(robot_crop, zone_start)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < _OCR_RULES["blur_rejection_threshold"]:
            continue  # this zone too blurry; try next
        candidates = _paddle_readtext(ocr_engine, processed)
        if candidates:
            break  # got something — stop trying zones

    if not candidates:
        fallback = f"{_OCR_RULES['fallback_if_unknown']}_{track_id}"
        return _ocr_cache.get(track_id, {}).get("team_number") or fallback

    reading = candidates[0]

    # Update per-track cache
    if track_id not in _ocr_cache:
        _ocr_cache[track_id] = {"readings": [], "team_number": None, "confidence": 0.0}
    _ocr_cache[track_id]["readings"].append(reading)

    readings = _ocr_cache[track_id]["readings"]
    if len(readings) >= _OCR_RULES["min_frames_for_confidence"]:
        from collections import Counter
        most_common, count = Counter(readings).most_common(1)[0]
        conf = count / len(readings)
        if conf >= _OCR_RULES["majority_vote_threshold"]:
            _ocr_cache[track_id]["team_number"] = most_common
            _ocr_cache[track_id]["confidence"]  = conf
            return most_common

    return f"{_OCR_RULES['fallback_if_unknown']}_{track_id}"


def clear_ocr_cache(track_id: int | None = None) -> None:
    """
    Clear OCR cache (call when a robot disappears for 60+ frames).

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
