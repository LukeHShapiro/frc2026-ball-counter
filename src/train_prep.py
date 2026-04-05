"""
src/train_prep.py - Phase 3: Training Data Preparation

Functions:
  ingest_roboflow_export()  - copy + remap a combined Roboflow dataset into
                              data/labeled/balls/ and data/labeled/robots/
  validate_labels()         - check YOLOv8 format (.txt class cx cy w h)
  split_dataset()           - 80/10/10 train/val/test split per model
  generate_dataset_yaml()   - write configs/dataset_balls.yaml and
                              configs/dataset_robots.yaml

Depends on: Phase 2 (environment ready), INPUT CHECKPOINTS #2, #3, #4.

(!) STOP after this module is complete --
   "PHASE 3 COMPLETE. Ready for Phase 4 (model training)."
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Optional

import yaml


# ---- Class remapping constants ----------------------------------------------

# Roboflow export class IDs (from classes.txt)
ROBOFLOW_CLASSES = {
    0: "Blue_Robot",
    1: "Fuel",
    2: "Red_Robot",
}

# For ball model: keep only Fuel (class 1), remap to class 0
BALL_CLASSES = ["Fuel"]
BALL_KEEP_IDS = {1}           # original class IDs to include
BALL_REMAP = {1: 0}           # original_id -> new_id

# For robot model: keep Blue_Robot (0) + Red_Robot (2), remap both to 0
ROBOT_CLASSES = ["robot"]
ROBOT_KEEP_IDS = {0, 2}
ROBOT_REMAP = {0: 0, 2: 0}


# ---- ingest_roboflow_export -------------------------------------------------

def ingest_roboflow_export(
    roboflow_images_dir: str | Path,
    project_root: str | Path = ".",
) -> tuple[int, int]:
    """
    Copy images and remap labels from a combined Roboflow YOLOv8 export into:
      data/labeled/balls/   - images + labels for ball detector
      data/labeled/robots/  - images + labels for robot detector

    The Roboflow export is expected to have:
      {roboflow_images_dir}/           <- images (*.jpg / *.png)
      {roboflow_images_dir}/../labels/ <- YOLOv8 .txt label files
      (or labels/ next to images/)

    Label remapping:
      Ball model   : class 1 (Fuel)      -> 0
      Robot model  : class 0/2 (R/B bot) -> 0  (single "robot" class)

    Args:
        roboflow_images_dir: Path to the images/ folder from the Roboflow export.
        project_root:        Root of the frc2026-ball-counter project.

    Returns:
        (ball_samples, robot_samples) - number of images written for each model.
    """
    images_dir = Path(roboflow_images_dir)
    labels_dir = images_dir.parent / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    root = Path(project_root)
    balls_out  = root / "data" / "labeled" / "balls"
    robots_out = root / "data" / "labeled" / "robots"
    balls_out.mkdir(parents=True, exist_ok=True)
    robots_out.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    ball_count   = 0
    robot_count  = 0
    skipped      = 0

    print(f"  [Phase 3] Ingesting Roboflow export from: {images_dir}")
    print(f"  [Phase 3] Found {len(image_files)} images.")

    for img_path in image_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        raw_lines = label_path.read_text().splitlines()
        ball_lines  = []
        robot_lines = []

        for line in raw_lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            coords = " ".join(parts[1:])
            if cls_id in BALL_KEEP_IDS:
                ball_lines.append(f"{BALL_REMAP[cls_id]} {coords}")
            if cls_id in ROBOT_KEEP_IDS:
                robot_lines.append(f"{ROBOT_REMAP[cls_id]} {coords}")

        if ball_lines:
            dest_img = balls_out / img_path.name
            dest_lbl = balls_out / (img_path.stem + ".txt")
            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(ball_lines))
            ball_count += 1

        if robot_lines:
            dest_img = robots_out / img_path.name
            dest_lbl = robots_out / (img_path.stem + ".txt")
            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("\n".join(robot_lines))
            robot_count += 1

    print(f"  [Phase 3] Ball samples written  : {ball_count}")
    print(f"  [Phase 3] Robot samples written : {robot_count}")
    if skipped:
        print(f"  [Phase 3] Images skipped (no label file): {skipped}")

    return ball_count, robot_count


# ---- validate_labels --------------------------------------------------------

def validate_labels(
    labeled_dir: str | Path,
    expected_classes: list[str],
    verbose: bool = True,
) -> dict:
    """
    Validate YOLOv8-format label files in labeled_dir.

    Checks:
      - Each .txt file has >= 1 annotation line
      - Each line has exactly 5 fields: class cx cy w h
      - class_id is within [0, len(expected_classes))
      - cx, cy, w, h are floats in (0, 1]

    Args:
        labeled_dir:      Directory containing image + .txt label files.
        expected_classes: Class name list (e.g. ["Fuel"] or ["robot"]).

    Returns:
        {
            "total_images":   int,
            "valid_labels":   int,
            "bad_labels":     int,
            "error_details":  [str],   # one string per bad line
        }
    """
    labeled_dir = Path(labeled_dir)
    n_classes   = len(expected_classes)

    label_files = sorted(labeled_dir.glob("*.txt"))
    total = len(label_files)
    valid = 0
    bad   = 0
    errors: list[str] = []

    for lbl_path in label_files:
        lines = lbl_path.read_text().splitlines()
        if not lines:
            bad += 1
            errors.append(f"{lbl_path.name}: EMPTY label file")
            continue

        file_ok = True
        for lineno, line in enumerate(lines, 1):
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(
                    f"{lbl_path.name}:{lineno}: expected 5 fields, got {len(parts)}"
                )
                file_ok = False
                continue
            cls_id = int(parts[0])
            if not (0 <= cls_id < n_classes):
                errors.append(
                    f"{lbl_path.name}:{lineno}: class_id {cls_id} out of range "
                    f"[0, {n_classes})"
                )
                file_ok = False
                continue
            coords = [float(x) for x in parts[1:]]
            for val in coords:
                if not (0.0 < val <= 1.0):
                    errors.append(
                        f"{lbl_path.name}:{lineno}: coordinate {val} out of (0,1]"
                    )
                    file_ok = False

        if file_ok:
            valid += 1
        else:
            bad += 1

    result = {
        "total_images": total,
        "valid_labels": valid,
        "bad_labels":   bad,
        "error_details": errors,
    }

    if verbose:
        status = "OK" if bad == 0 else "WARN"
        print(f"  [Phase 3] Validate {labeled_dir.name}:"
              f"  {valid}/{total} valid  [{status}]")
        for e in errors[:10]:
            print(f"            {e}")
        if len(errors) > 10:
            print(f"            ... and {len(errors)-10} more errors.")

    return result


# ---- split_dataset ----------------------------------------------------------

def split_dataset(
    labeled_dir:  str | Path,
    model_name:   str,
    project_root: str | Path = ".",
    train: float = 0.80,
    val:   float = 0.10,
    test:  float = 0.10,
    seed:  int   = 42,
) -> dict[str, int]:
    """
    Split a labeled directory into train / val / test subsets.

    Images and their matching .txt labels are copied into:
      data/splits/{model_name}/train/
      data/splits/{model_name}/val/
      data/splits/{model_name}/test/

    Args:
        labeled_dir:  Directory with interleaved image + .txt files.
        model_name:   "balls" or "robots" (subfolder name under data/splits/).
        project_root: Root of project.
        train/val/test: Split fractions (must sum to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        {"train": int, "val": int, "test": int}
    """
    assert abs(train + val + test - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    labeled_dir = Path(labeled_dir)
    root        = Path(project_root)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        p for p in labeled_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    rng = random.Random(seed)
    rng.shuffle(images)

    n       = len(images)
    n_train = round(n * train)
    n_val   = round(n * val)
    splits  = {
        "train": images[:n_train],
        "val":   images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    counts: dict[str, int] = {}
    for split_name, split_images in splits.items():
        out_dir = root / "data" / "splits" / model_name / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in split_images:
            shutil.copy2(img_path, out_dir / img_path.name)
            lbl_src = labeled_dir / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, out_dir / lbl_src.name)
        counts[split_name] = len(split_images)

    print(f"  [Phase 3] Split '{model_name}': "
          f"train={counts['train']}  val={counts['val']}  test={counts['test']}")
    return counts


# ---- generate_dataset_yaml --------------------------------------------------

def generate_dataset_yaml(
    class_names:  list[str],
    model_name:   str,
    output_path:  str | Path,
    project_root: str | Path = ".",
) -> Path:
    """
    Write a YOLOv8-compatible dataset YAML config.

    Args:
        class_names:  e.g. ["Fuel"] or ["robot"]
        model_name:   "balls" or "robots" (matches split subfolder)
        output_path:  Where to write the YAML (e.g. configs/dataset_balls.yaml)
        project_root: Absolute path to project root.

    Returns:
        Path to the written YAML file.
    """
    root     = Path(project_root).resolve()
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    splits_base = root / "data" / "splits" / model_name

    cfg = {
        "path":  str(splits_base),
        "train": "train",
        "val":   "val",
        "test":  "test",
        "nc":    len(class_names),
        "names": class_names,
    }

    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"  [Phase 3] Dataset YAML written -> {out_path}")
    print(f"            classes ({len(class_names)}): {class_names}")
    print(f"            path: {splits_base}")

    return out_path


# ---- CLI / Phase 3 runner ---------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Phase 3 - Training Data Preparation")
    parser.add_argument(
        "roboflow_images",
        help="Path to the images/ folder from the Roboflow export",
    )
    parser.add_argument(
        "--root", default=".",
        help="Project root directory (default: current dir)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 3 -- Training Data Preparation")
    print("=" * 60)

    root = Path(args.root).resolve()

    # Step 1: Ingest + remap the Roboflow export
    print("\n[Step 1] Ingesting Roboflow export...")
    ball_n, robot_n = ingest_roboflow_export(args.roboflow_images, root)

    # Step 2: Validate both labeled sets
    print("\n[Step 2] Validating labels...")
    balls_dir  = root / "data" / "labeled" / "balls"
    robots_dir = root / "data" / "labeled" / "robots"

    ball_validation  = validate_labels(balls_dir,  BALL_CLASSES)
    robot_validation = validate_labels(robots_dir, ROBOT_CLASSES)

    if ball_validation["bad_labels"] > 0:
        print(f"  (!) {ball_validation['bad_labels']} bad ball label files -- review errors above.")
    if robot_validation["bad_labels"] > 0:
        print(f"  (!) {robot_validation['bad_labels']} bad robot label files -- review errors above.")

    # Step 3: Split datasets
    print("\n[Step 3] Splitting datasets (80/10/10)...")
    ball_splits  = split_dataset(balls_dir,  "balls",  root)
    robot_splits = split_dataset(robots_dir, "robots", root)

    # Step 4: Generate YAML configs
    print("\n[Step 4] Generating dataset YAML configs...")
    generate_dataset_yaml(
        BALL_CLASSES,
        "balls",
        root / "configs" / "dataset_balls.yaml",
        root,
    )
    generate_dataset_yaml(
        ROBOT_CLASSES,
        "robots",
        root / "configs" / "dataset_robots.yaml",
        root,
    )

    # Summary
    print("\n" + "=" * 60)
    print("  Ball samples  :", ball_n,
          f"-> train={ball_splits['train']}  val={ball_splits['val']}  test={ball_splits['test']}")
    print("  Robot samples :", robot_n,
          f"-> train={robot_splits['train']}  val={robot_splits['val']}  test={robot_splits['test']}")
    print("=" * 60)
    print("\nPHASE 3 COMPLETE. Ready for Phase 4 (model training).")
    sys.exit(0)
