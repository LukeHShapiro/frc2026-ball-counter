"""
src/train.py - Phase 4: Model Training

Functions:
  estimate_training_time()  - detect GPU/CPU, estimate time, confirm with user
  train_model()             - train one YOLOv8s model (call twice: balls, robots)
  train_all()               - train ball detector then robot detector in sequence

Minimum acceptable mAP50 before proceeding:
  Ball detector:  >= 0.70
  Robot detector: >= 0.65

Depends on: Phase 3 (data splits + YAML configs ready).

(!) STOP after this module is complete --
   "PHASE 4 COMPLETE. Ball mAP: [X]. Robot mAP: [X]. Awaiting INPUT CHECKPOINT #5."
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ---- Minimum accuracy thresholds -------------------------------------------

MAP50_MIN = {
    "balls":   0.70,
    "robots":  0.65,
}


# ---- estimate_training_time -------------------------------------------------

def estimate_training_time(
    n_images: int = 50,
    epochs:   int = 75,
) -> dict:
    """
    Detect GPU/CPU environment and estimate training time.

    Args:
        n_images: Number of training images (for scaling estimate).
        epochs:   Planned epoch count.

    Returns:
        {"device": str, "cuda": bool, "gpu_name": str or None,
         "est_seconds_per_epoch": float, "est_total_minutes": float}
    """
    cuda = False
    gpu_name = None
    est_sec_per_epoch = 60.0   # conservative CPU baseline

    if TORCH_AVAILABLE:
        cuda = torch.cuda.is_available()
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            # RTX-class GPU: ~3-8s/epoch on 50 images at imgsz=640
            est_sec_per_epoch = max(3.0, n_images * 0.10)
        else:
            # CPU: ~30-120s/epoch depending on core count
            est_sec_per_epoch = max(30.0, n_images * 1.5)

    device = f"cuda ({gpu_name})" if cuda else "cpu"
    est_total_minutes = (est_sec_per_epoch * epochs) / 60.0

    result = {
        "device":                device,
        "cuda":                  cuda,
        "gpu_name":              gpu_name,
        "est_seconds_per_epoch": round(est_sec_per_epoch, 1),
        "est_total_minutes":     round(est_total_minutes, 1),
    }

    print(f"  [Train] Device            : {device}")
    print(f"  [Train] Est. time/epoch   : {est_sec_per_epoch:.1f}s")
    print(f"  [Train] Est. total (x2)   : ~{est_total_minutes*2:.0f} min "
          f"(both models, {epochs} epochs each)")

    return result


# ---- train_model ------------------------------------------------------------

def train_model(
    dataset_yaml:   str | Path,
    base_weights:   str | Path,
    model_name:     str,
    project_root:   str | Path = ".",
    epochs:         int   = 75,
    imgsz:          int   = 1280,
    batch:          int   = -1,       # -1 = auto
    patience:       int   = 20,
    confirm:        bool  = True,
) -> dict:
    """
    Train one YOLOv8s model and save best weights.

    Best weights are saved to models/trained/{model_name}/best.pt
    Results are saved to models/trained/{model_name}/results.json

    Args:
        dataset_yaml:  Path to configs/dataset_{model_name}.yaml
        base_weights:  Path to YOLOv8s base weights (models/base/yolov8s.pt)
        model_name:    "balls" or "robots"
        project_root:  Project root.
        epochs:        Training epochs.
        imgsz:         Input image size (px).
        batch:         Batch size (-1 = auto-detect).
        patience:      Early stopping patience.
        confirm:       If True, ask user to confirm before starting.

    Returns:
        {"model_name": str, "map50": float, "precision": float,
         "recall": float, "weights_path": str, "passed_threshold": bool}

    Raises:
        RuntimeError: If ultralytics not installed.
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    dataset_yaml = Path(dataset_yaml)
    base_weights = Path(base_weights)
    root         = Path(project_root).resolve()
    out_dir      = root / "models" / "trained" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  [Train] --- {model_name.upper()} DETECTOR ---")
    print(f"  [Train] Dataset YAML : {dataset_yaml}")
    print(f"  [Train] Base weights : {base_weights}")
    print(f"  [Train] Epochs       : {epochs}  |  imgsz: {imgsz}")
    print(f"  [Train] Output dir   : {out_dir}")

    # Estimate time before starting
    device_info = estimate_training_time(epochs=epochs)

    if confirm:
        print(f"\n  Estimated ~{device_info['est_total_minutes']:.0f} min on "
              f"{device_info['device']}.")
        ans = input("  Start training? (yes/no): ").strip().lower()
        if ans not in ("yes", "y"):
            print("  Training cancelled.")
            return {}

    # Load model + train
    device = "0" if device_info["cuda"] else "cpu"
    model  = YOLO(str(base_weights))

    t0 = time.time()
    results = model.train(
        data      = str(dataset_yaml),
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        patience  = patience,
        device    = device,
        project   = str(out_dir.parent),
        name      = model_name,
        exist_ok  = True,
        verbose   = True,
    )
    elapsed = time.time() - t0

    # --- Extract metrics from results ---
    # ultralytics Results object exposes .results_dict or .box.map50
    map50       = 0.0
    precision   = 0.0
    recall      = 0.0

    try:
        metrics = results.results_dict
        map50     = float(metrics.get("metrics/mAP50(B)", 0.0))
        precision = float(metrics.get("metrics/precision(B)", 0.0))
        recall    = float(metrics.get("metrics/recall(B)", 0.0))
    except Exception:
        # Fallback: try validator results
        try:
            map50     = float(results.box.map50)
            precision = float(results.box.mp)
            recall    = float(results.box.mr)
        except Exception:
            pass

    # Copy best weights to canonical location
    # ultralytics saves to project/name/weights/best.pt
    trained_best = out_dir / model_name / "weights" / "best.pt"
    canonical    = out_dir / "best.pt"
    if trained_best.exists() and not canonical.exists():
        import shutil
        shutil.copy2(trained_best, canonical)

    # Also check the direct path ultralytics sometimes uses
    alt_best = out_dir / "weights" / "best.pt"
    if alt_best.exists() and not canonical.exists():
        import shutil
        shutil.copy2(alt_best, canonical)

    threshold = MAP50_MIN.get(model_name, 0.65)
    passed    = map50 >= threshold

    result = {
        "model_name":        model_name,
        "map50":             round(map50, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "elapsed_seconds":   round(elapsed, 1),
        "weights_path":      str(canonical),
        "passed_threshold":  passed,
        "threshold":         threshold,
    }

    # Save results JSON
    results_json = out_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  [Train] {model_name.upper()} training complete "
          f"({elapsed/60:.1f} min)")
    print(f"  [Train] mAP50     : {map50:.4f}  (threshold: {threshold})")
    print(f"  [Train] Precision : {precision:.4f}")
    print(f"  [Train] Recall    : {recall:.4f}")
    status = "OK  Threshold met." if passed else \
        f"(!) mAP50 {map50:.4f} < {threshold} -- consider retraining or proceeding with caution."
    print(f"  [Train] Status    : {status}")

    return result


# ---- train_all --------------------------------------------------------------

def train_all(
    project_root: str | Path = ".",
    epochs:       int   = 75,
    imgsz:        int   = 1280,
    confirm:      bool  = True,
) -> dict:
    """
    Train both detectors in sequence: ball detector first, then robot detector.

    Args:
        project_root: Project root.
        epochs:       Training epochs for each model.
        imgsz:        Input image size.
        confirm:      Ask user to confirm before each training run.

    Returns:
        {"balls": result_dict, "robots": result_dict}
    """
    root         = Path(project_root).resolve()
    base_weights = root / "models" / "base" / "yolov8s.pt"

    if not base_weights.exists():
        raise FileNotFoundError(
            f"Base weights not found: {base_weights}\n"
            "Run setup_check.py first to download them."
        )

    print("\n" + "=" * 60)
    print("PHASE 4 -- Model Training")
    print("=" * 60)

    results = {}

    # --- Model A: Ball Detector ---
    ball_yaml = root / "configs" / "dataset_balls.yaml"
    if not ball_yaml.exists():
        raise FileNotFoundError(f"Ball dataset YAML not found: {ball_yaml}")

    results["balls"] = train_model(
        dataset_yaml  = ball_yaml,
        base_weights  = base_weights,
        model_name    = "balls",
        project_root  = root,
        epochs        = epochs,
        imgsz         = imgsz,
        confirm       = confirm,
    )

    # --- Model B: Robot Detector ---
    robot_yaml = root / "configs" / "dataset_robots.yaml"
    if not robot_yaml.exists():
        raise FileNotFoundError(f"Robot dataset YAML not found: {robot_yaml}")

    results["robots"] = train_model(
        dataset_yaml  = robot_yaml,
        base_weights  = base_weights,
        model_name    = "robots",
        project_root  = root,
        epochs        = epochs,
        imgsz         = imgsz,
        confirm       = confirm,
    )

    # --- Final summary ---
    ball_map  = results["balls"].get("map50",  0.0)
    robot_map = results["robots"].get("map50", 0.0)
    ball_pass  = results["balls"].get("passed_threshold",  False)
    robot_pass = results["robots"].get("passed_threshold", False)

    print("\n" + "=" * 60)
    print(f"  Ball detector  mAP50: {ball_map:.4f}  "
          f"({'OK' if ball_pass else 'BELOW THRESHOLD'})")
    print(f"  Robot detector mAP50: {robot_map:.4f}  "
          f"({'OK' if robot_pass else 'BELOW THRESHOLD'})")
    print("=" * 60)
    print(f"\nPHASE 4 COMPLETE. Ball mAP: {ball_map:.4f}. "
          f"Robot mAP: {robot_map:.4f}. Awaiting INPUT CHECKPOINT #5.")

    if not ball_pass:
        print("(!) Ball detector mAP below 0.70. "
              "Consider adding more labeled data or retraining.")
    if not robot_pass:
        print("(!) Robot detector mAP below 0.65. "
              "Consider adding more labeled data or retraining.")

    return results


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 4 -- Model Training")
    parser.add_argument("--root",    default=".",  help="Project root")
    parser.add_argument("--epochs",  type=int, default=75)
    parser.add_argument("--imgsz",   type=int, default=1280)
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompts (for CI)")
    args = parser.parse_args()

    train_all(
        project_root = args.root,
        epochs       = args.epochs,
        imgsz        = args.imgsz,
        confirm      = not args.no_confirm,
    )
