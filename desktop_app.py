"""
desktop_app.py — Entry point for OmniScouter desktop application.

Usage:
    python desktop_app.py
    py desktop_app.py        (Windows py launcher)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

# Ensure src/ is on the path so gui sub-package imports resolve
_ROOT = Path(__file__).parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import QApplication

from gui.window import MainWindow


def _prewarm_models() -> None:
    """
    Load local YOLO models in a daemon background thread (#11).

    Runs immediately on app launch so the first pipeline run doesn't pay
    the model-load latency (typically 2-5 s on disk, longer for TensorRT).
    Failures are silently ignored — the main pipeline will handle them.
    """
    try:
        from detect import _load_local_models
        _load_local_models(project_root=_ROOT)
    except Exception:
        pass   # models not trained yet — no-op


def main() -> int:
    # High-DPI support (PyQt6 enables it by default, but be explicit)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Start model pre-warm before the window is shown (#11)
    threading.Thread(target=_prewarm_models, daemon=True).start()

    app = QApplication(sys.argv)
    app.setApplicationName("OmniScouter")
    app.setOrganizationName("FRC Scouting")
    app.setApplicationVersion("0.1.0")

    # App icon (taskbar + title bar)
    _icon_path = _ROOT / "assets" / "icon.ico"
    if _icon_path.exists():
        app.setWindowIcon(QIcon(str(_icon_path)))

    # Default font — Segoe UI on Windows, system fallback elsewhere
    font = QFont("Segoe UI", 10)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
