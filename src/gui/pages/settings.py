"""
src/gui/pages/settings.py — App settings and data reset
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from ..theme import C

_ROOT = Path(__file__).parent.parent.parent.parent


def _lbl(text: str, obj: str = "") -> QLabel:
    w = QLabel(text)
    if obj:
        w.setObjectName(obj)
    return w


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color: {C['border']};")
    return f


class _ActionRow(QWidget):
    """A labelled row with a description and an action button."""

    def __init__(self, title: str, description: str, btn_text: str,
                 btn_style: str = "", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color: {C['text']}; font-weight: 600; font-size: 13px;")
        text_col.addWidget(title_lbl)
        desc_lbl = QLabel(description)
        desc_lbl.setObjectName("muted")
        desc_lbl.setWordWrap(True)
        text_col.addWidget(desc_lbl)
        lay.addLayout(text_col, 1)

        self.btn = QPushButton(btn_text)
        if btn_style:
            self.btn.setObjectName(btn_style)
        self.btn.setFixedWidth(130)
        self.btn.setFixedHeight(34)
        lay.addWidget(self.btn, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 36, 48, 36)
        root.setSpacing(0)

        root.addWidget(_lbl("Settings", "h1"))
        root.addSpacing(4)
        root.addWidget(_lbl("Manage pipeline data and application state.", "muted"))
        root.addSpacing(28)

        # ── Data Management ───────────────────────────────────────────────────
        root.addWidget(_lbl("DATA MANAGEMENT", "kpi_lbl"))
        root.addSpacing(12)

        card = QFrame()
        card.setObjectName("card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(16)

        # Clear robot identity
        row1 = _ActionRow(
            "Clear Robot Identity",
            "Remove saved team-number assignments. The next pipeline run will "
            "re-run OCR and ask you to reassign team numbers.",
            "Clear Identity",
            "sec",
        )
        row1.btn.clicked.connect(self._clear_identity)
        cl.addWidget(row1)
        cl.addWidget(_sep())

        # Clear detection cache
        row2 = _ActionRow(
            "Clear Detection Cache",
            "Delete cached YOLO detections (data/detections.json). Forces a full "
            "re-detection on the next run. Useful after changing sample rate.",
            "Clear Cache",
            "sec",
        )
        row2.btn.clicked.connect(self._clear_detection_cache)
        cl.addWidget(row2)
        cl.addWidget(_sep())

        # Clear all results
        row3 = _ActionRow(
            "Clear All Results",
            "Delete scores, timeline, driving report, and possession log. "
            "Keeps detection cache and robot identity intact.",
            "Clear Results",
            "sec",
        )
        row3.btn.clicked.connect(self._clear_results)
        cl.addWidget(row3)
        cl.addWidget(_sep())

        # Full reset
        row4 = _ActionRow(
            "Full Reset",
            "Delete ALL pipeline outputs — detection cache, results, identity map, "
            "possession log, and exported files. Returns app to a clean state.",
            "Full Reset",
            "danger",
        )
        row4.btn.clicked.connect(self._full_reset)
        cl.addWidget(row4)

        root.addWidget(card)
        root.addStretch()

        # ── About ─────────────────────────────────────────────────────────────
        root.addWidget(_sep())
        root.addSpacing(12)
        about = QLabel("OmniScouter  v0.1.0-alpha  •  FRC 2026 Analysis Platform")
        about.setObjectName("muted")
        about.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(about)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _confirm(self, title: str, message: str) -> bool:
        dlg = QMessageBox(self)
        dlg.setWindowTitle(title)
        dlg.setText(message)
        dlg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        dlg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        dlg.setIcon(QMessageBox.Icon.Warning)
        return dlg.exec() == QMessageBox.StandardButton.Yes

    def _delete(self, paths: list[Path]) -> int:
        deleted = 0
        for p in paths:
            if p.exists():
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                deleted += 1
        return deleted

    def _info(self, message: str):
        QMessageBox.information(self, "Done", message)

    def _clear_identity(self):
        if not self._confirm("Clear Robot Identity",
                             "This will delete configs/match_identity.json.\n"
                             "Team number assignments will be lost."):
            return
        n = self._delete([_ROOT / "configs" / "match_identity.json"])
        self._info(f"Cleared robot identity ({n} file removed).")

    def _clear_detection_cache(self):
        if not self._confirm("Clear Detection Cache",
                             "This will delete data/detections.json.\n"
                             "The next run will re-detect all frames (may take several minutes)."):
            return
        n = self._delete([_ROOT / "data" / "detections.json"])
        self._info(f"Detection cache cleared ({n} file removed).")

    def _clear_results(self):
        if not self._confirm("Clear All Results",
                             "This will delete scores, timeline, driving report, "
                             "possession log, and all exported files.\n"
                             "Detection cache and robot identity are kept."):
            return
        targets = [
            _ROOT / "data" / "exports",
            _ROOT / "data" / "score_timeline.json",
            _ROOT / "data" / "possession_log.json",
            _ROOT / "data" / "possession_log.sig",
        ]
        n = self._delete(targets)
        self._info(f"Results cleared ({n} items removed).")

    def _full_reset(self):
        if not self._confirm("Full Reset",
                             "This will delete ALL pipeline data:\n"
                             "  • Detection cache\n"
                             "  • Robot identity map\n"
                             "  • Scores, timeline, driving report\n"
                             "  • Possession log\n"
                             "  • All exported files\n\n"
                             "This cannot be undone."):
            return
        targets = [
            _ROOT / "data" / "detections.json",
            _ROOT / "data" / "exports",
            _ROOT / "data" / "score_timeline.json",
            _ROOT / "data" / "possession_log.json",
            _ROOT / "data" / "possession_log.sig",
            _ROOT / "configs" / "match_identity.json",
        ]
        n = self._delete(targets)
        self._info(f"Full reset complete ({n} items removed).")
