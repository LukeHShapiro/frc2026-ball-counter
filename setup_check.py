"""
setup_check.py — Phase 1 environment verification for FRC 2026 Ball Counter.

Verifies:
  - All required packages are importable
  - CUDA availability (GPU vs CPU)
  - YOLOv8s base weights are downloaded to models/base/
  - All required project directories exist

Prints PASS/FAIL for each check. Exit code 0 if all pass, 1 if any fail.
"""

from pathlib import Path
import sys

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(label: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    results.append(ok)


# ── 1. Required directories ────────────────────────────────────────────────────
print("\n=== Directory Structure ===")
required_dirs = [
    "data/raw_frames",
    "data/labeled/balls",
    "data/labeled/robots",
    "data/labeled/bumpers",
    "data/splits/train",
    "data/splits/val",
    "data/splits/test",
    "models/base",
    "models/trained/ball_detector",
    "models/trained/robot_detector",
    "src",
    "configs",
    "tests",
]
for d in required_dirs:
    check(d, Path(d).is_dir())


# ── 2. Package imports ─────────────────────────────────────────────────────────
print("\n=== Package Imports ===")

packages = [
    ("ultralytics",          "ultralytics"),
    ("opencv-python",        "cv2"),
    ("deep-sort-realtime",   "deep_sort_realtime"),
    ("gradio",               "gradio"),
    ("pandas",               "pandas"),
    ("numpy",                "numpy"),
    ("torch",                "torch"),
    ("torchvision",          "torchvision"),
    ("Pillow",               "PIL"),
    ("PyYAML",               "yaml"),
    ("easyocr",              "easyocr"),
    ("scipy",                "scipy"),
    ("filterpy",             "filterpy"),
]

for pkg_name, import_name in packages:
    try:
        __import__(import_name)
        check(pkg_name, True)
    except ImportError as e:
        check(pkg_name, False, str(e))


# ── 3. CUDA availability ───────────────────────────────────────────────────────
print("\n=== Hardware ===")
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    device_detail = torch.cuda.get_device_name(0) if cuda_ok else "CPU only"
    check("CUDA / GPU", cuda_ok, device_detail)
    if not cuda_ok:
        print("  (INFO) No GPU detected — training will run on CPU and will be slow.")
except Exception as e:
    check("CUDA / GPU", False, str(e))


# ── 4. YOLOv8s base weights ────────────────────────────────────────────────────
print("\n=== Base Weights ===")
weights_path = Path("models/base/yolov8s.pt")

if not weights_path.exists():
    print("  (INFO) yolov8s.pt not found — downloading now …")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8s.pt")          # downloads to cache first
        import shutil
        # Ultralytics caches in ~/.cache/ultralytics — find and copy
        import torch.hub as hub
        cache_dirs = [
            Path.home() / ".cache" / "ultralytics" / "assets",
            Path.home() / ".cache" / "ultralytics",
            Path("yolov8s.pt"),             # sometimes saved locally
        ]
        copied = False
        for candidate in cache_dirs:
            src = candidate if candidate.suffix == ".pt" else candidate / "yolov8s.pt"
            if src.exists():
                shutil.copy(src, weights_path)
                copied = True
                break
        if not copied:
            # Try saving via model.save()
            model.save(str(weights_path))
            copied = weights_path.exists()
        check("yolov8s.pt downloaded", copied,
              str(weights_path) if copied else "copy failed — check cache manually")
    except Exception as e:
        check("yolov8s.pt downloaded", False, str(e))
else:
    size_mb = weights_path.stat().st_size / (1024 * 1024)
    check("yolov8s.pt present", True, f"{size_mb:.1f} MB at {weights_path}")


# ── Summary ────────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
failed = total - passed

print(f"\n{'='*40}")
print(f"  Results: {passed}/{total} checks passed")
if failed:
    print(f"  {failed} check(s) FAILED — fix above errors before Phase 2.")
    print("="*40)
    sys.exit(1)
else:
    print("  All checks PASSED.")
    print("="*40)
    print("\nPHASE 1 COMPLETE. Run setup_check.py and confirm all PASS.")
    sys.exit(0)
