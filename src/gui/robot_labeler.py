"""
src/gui/robot_labeler.py — Visual robot labeling dialog

Shows a video frame. User draws bounding boxes around each robot,
types team numbers, picks red/blue alliance. Saves to match_identity.json.

Max 6 boxes (3 red + 3 blue). Existing tracks from match_identity.json
are pre-loaded so auto-detected robots don't need to be re-drawn.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPixmap,
)
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSlider, QSpinBox, QVBoxLayout, QWidget, QGridLayout,
    QMessageBox,
)

_ROOT = Path(__file__).parent.parent.parent

_ALLIANCE_COLOR = {
    "red":  QColor(220, 38,  38,  200),
    "blue": QColor(37,  99,  235, 200),
}
_FILL_COLOR = {
    "red":  QColor(220, 38,  38,  40),
    "blue": QColor(37,  99,  235, 40),
}
_MAX_ROBOTS = 6


# ── Canvas widget ─────────────────────────────────────────────────────────────

class _Canvas(QLabel):
    """
    Displays a video frame and lets the user draw bounding boxes.
    Emits no signals — the dialog polls self.boxes directly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setMouseTracking(True)

        self._base_pixmap: Optional[QPixmap] = None   # unmodified frame
        self._scale: float = 1.0

        # Each box: {x1,y1,x2,y2 in original coords, team, alliance}
        self.boxes: list[dict] = []

        self._drawing = False
        self._drag_start: Optional[QPoint] = None
        self._drag_end:   Optional[QPoint] = None
        self._pending_alliance = "red"

    # ── public ────────────────────────────────────────────────────────────────

    def set_frame(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img   = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._base_pixmap = QPixmap.fromImage(img)
        # Scale to fit 900px wide max
        max_w = 900
        if w > max_w:
            self._scale = max_w / w
            self._base_pixmap = self._base_pixmap.scaledToWidth(
                max_w, Qt.TransformationMode.SmoothTransformation)
        else:
            self._scale = 1.0
        self.setFixedSize(self._base_pixmap.size())
        self._redraw()

    def set_pending_alliance(self, alliance: str):
        self._pending_alliance = alliance

    def remove_last_box(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw()

    def clear_boxes(self):
        self.boxes.clear()
        self._redraw()

    # ── mouse events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap:
            self._drawing   = True
            self._drag_start = event.position().toPoint()
            self._drag_end   = self._drag_start

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._drag_end = event.position().toPoint()
            self._redraw()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing  = False
            self._drag_end = event.position().toPoint()

            # Hard cap at 6 robots
            if len(self.boxes) >= 6:
                self._drag_start = self._drag_end = None
                self._redraw()
                return

            # Convert to original-image coordinates
            r = self._norm_rect(self._drag_start, self._drag_end)
            if r.width() > 8 and r.height() > 8:
                x1 = int(r.left()   / self._scale)
                y1 = int(r.top()    / self._scale)
                x2 = int(r.right()  / self._scale)
                y2 = int(r.bottom() / self._scale)
                self.boxes.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "team":     "",
                    "alliance": self._pending_alliance,
                    "source":   "manual",
                })
            self._drag_start = self._drag_end = None
            self._redraw()

    # ── drawing ───────────────────────────────────────────────────────────────

    def _norm_rect(self, a: QPoint, b: QPoint) -> QRect:
        return QRect(
            min(a.x(), b.x()), min(a.y(), b.y()),
            abs(b.x() - a.x()), abs(b.y() - a.y()),
        )

    def _redraw(self):
        if self._base_pixmap is None:
            return
        pm = QPixmap(self._base_pixmap)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(font)

        # Draw committed boxes
        for i, box in enumerate(self.boxes):
            al    = box.get("alliance", "red")
            color = _ALLIANCE_COLOR.get(al, _ALLIANCE_COLOR["red"])
            fill  = _FILL_COLOR.get(al,  _FILL_COLOR["red"])

            sx1 = int(box["x1"] * self._scale)
            sy1 = int(box["y1"] * self._scale)
            sx2 = int(box["x2"] * self._scale)
            sy2 = int(box["y2"] * self._scale)
            rect = QRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

            painter.fillRect(rect, fill)
            painter.setPen(QPen(color, 2))
            painter.drawRect(rect)

            # Label
            team = box.get("team") or f"#{i+1}"
            lbl  = f"{al[0].upper()} {team}"
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            bg = QRect(sx1, sy1 - 18, len(lbl) * 8 + 6, 18)
            painter.fillRect(bg, color)
            painter.drawText(bg, Qt.AlignmentFlag.AlignCenter, lbl)

        # Draw in-progress drag rect
        if self._drag_start and self._drag_end:
            al    = self._pending_alliance
            color = _ALLIANCE_COLOR.get(al, _ALLIANCE_COLOR["red"])
            r     = self._norm_rect(self._drag_start, self._drag_end)
            painter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            painter.drawRect(r)

        painter.end()
        self.setPixmap(pm)


# ── Row widget for each box ───────────────────────────────────────────────────

class _BoxRow(QWidget):
    def __init__(self, index: int, box: dict, on_change, parent=None):
        super().__init__(parent)
        self._box       = box
        self._on_change = on_change
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(8)

        idx_lbl = QLabel(f"Box {index + 1}")
        idx_lbl.setFixedWidth(48)
        idx_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        lay.addWidget(idx_lbl)

        self._team_edit = QLineEdit(box.get("team", ""))
        self._team_edit.setPlaceholderText("Team # (e.g. 1234)")
        self._team_edit.setFixedWidth(130)
        self._team_edit.textChanged.connect(self._update)
        lay.addWidget(self._team_edit)

        self._al_combo = QComboBox()
        self._al_combo.addItems(["red", "blue"])
        self._al_combo.setCurrentText(box.get("alliance", "red"))
        self._al_combo.setFixedWidth(70)
        self._al_combo.currentTextChanged.connect(self._update)
        lay.addWidget(self._al_combo)

        src = box.get("source", "manual")
        src_lbl = QLabel("auto" if src != "manual" else "drawn")
        src_lbl.setStyleSheet(
            "color: #22c55e; font-size: 10px;" if src != "manual"
            else "color: #6366f1; font-size: 10px;"
        )
        src_lbl.setFixedWidth(44)
        lay.addWidget(src_lbl)

    def _update(self):
        self._box["team"]     = self._team_edit.text().strip()
        self._box["alliance"] = self._al_combo.currentText()
        self._on_change()


# ── Main dialog ───────────────────────────────────────────────────────────────

class RobotLabelerDialog(QDialog):
    """
    Shows a video frame. User draws boxes around robots, assigns numbers.
    Accepts → writes to configs/match_identity.json.
    """

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Robot Labeler — Draw boxes around each robot")
        self.setMinimumSize(980, 720)
        self._video_path = video_path
        self._cap        = cv2.VideoCapture(video_path)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame_idx = 0
        self._current_frame: Optional[np.ndarray] = None
        self._build_ui()
        self._load_existing_identity()
        self._goto_frame(0)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(12)
        root.setContentsMargins(16, 16, 16, 12)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.addWidget(self._lbl("Frame:", "muted"))

        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setRange(0, max(1, self._total_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setFixedWidth(420)
        self._frame_slider.valueChanged.connect(self._goto_frame)
        top.addWidget(self._frame_slider)

        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, max(1, self._total_frames - 1))
        self._frame_spin.setFixedWidth(80)
        self._frame_spin.valueChanged.connect(self._goto_frame)
        top.addWidget(self._frame_spin)

        top.addSpacing(16)
        top.addWidget(self._lbl("Draw as:", "muted"))

        self._al_sel = QComboBox()
        self._al_sel.addItems(["red", "blue"])
        self._al_sel.setFixedWidth(70)
        self._al_sel.currentTextChanged.connect(
            lambda t: self._canvas.set_pending_alliance(t))
        top.addWidget(self._al_sel)

        undo_btn = QPushButton("Undo last")
        undo_btn.setObjectName("sec")
        undo_btn.setFixedWidth(80)
        undo_btn.clicked.connect(self._undo)
        top.addWidget(undo_btn)

        clear_btn = QPushButton("Clear all")
        clear_btn.setObjectName("danger")
        clear_btn.setFixedWidth(76)
        clear_btn.clicked.connect(self._clear)
        top.addWidget(clear_btn)

        top.addStretch()
        self._box_count_lbl = self._lbl("0 / 6 boxes", "muted")
        top.addWidget(self._box_count_lbl)

        root.addLayout(top)

        # ── Hint ──────────────────────────────────────────────────────────────
        hint = self._lbl(
            "Scrub to a frame where all robots are visible. "
            "Click & drag to draw a box around each robot. Max 3 red + 3 blue.",
            "muted"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        # ── Canvas in scroll area ─────────────────────────────────────────────
        self._canvas_scroll = QScrollArea()
        self._canvas_scroll.setWidgetResizable(True)
        self._canvas_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._canvas = _Canvas()
        self._canvas.set_pending_alliance("red")
        # Hook into canvas box changes to update the row list
        self._canvas_orig_release = self._canvas.mouseReleaseEvent
        def _patched_release(e):
            self._canvas_orig_release(e)
            self._refresh_rows()
        self._canvas.mouseReleaseEvent = _patched_release
        self._canvas_scroll.setWidget(self._canvas)
        self._canvas_scroll.setMinimumHeight(380)
        root.addWidget(self._canvas_scroll)

        # ── "Done drawing" confirm button ─────────────────────────────────────
        confirm_row = QHBoxLayout()
        confirm_row.addStretch()
        self._confirm_btn = QPushButton("Done Drawing  →  Enter Team Numbers")
        self._confirm_btn.setFixedHeight(38)
        self._confirm_btn.clicked.connect(self._confirm_drawing)
        confirm_row.addWidget(self._confirm_btn)
        confirm_row.addStretch()
        root.addLayout(confirm_row)

        # ── Box list (hidden until confirmed) ─────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        root.addWidget(sep)

        self._team_section = QWidget()
        team_sec_lay = QVBoxLayout(self._team_section)
        team_sec_lay.setContentsMargins(0, 0, 0, 0)
        team_sec_lay.setSpacing(6)
        team_sec_lay.addWidget(self._lbl("ASSIGN TEAM NUMBERS  (max 3 red + 3 blue)", "kpi_lbl"))

        self._rows_container = QWidget()
        self._rows_lay = QVBoxLayout(self._rows_container)
        self._rows_lay.setSpacing(4)
        self._rows_lay.setContentsMargins(0, 0, 0, 0)
        team_sec_lay.addWidget(self._rows_container)
        self._team_section.setVisible(False)
        root.addWidget(self._team_section)

        # ── Dialog buttons ────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── Frame navigation ──────────────────────────────────────────────────────

    def _goto_frame(self, idx: int):
        if idx == self._current_frame_idx and self._current_frame is not None:
            return
        self._current_frame_idx = idx
        # Sync slider and spinbox without re-triggering
        self._frame_slider.blockSignals(True)
        self._frame_spin.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_spin.setValue(idx)
        self._frame_slider.blockSignals(False)
        self._frame_spin.blockSignals(False)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if ret and frame is not None:
            self._current_frame = frame
            self._canvas.set_frame(frame)

    # ── Box management ────────────────────────────────────────────────────────

    def _undo(self):
        self._canvas.remove_last_box()
        self._refresh_rows()

    def _clear(self):
        self._canvas.clear_boxes()
        self._refresh_rows()

    def _confirm_drawing(self):
        """User finished drawing — reveal team number inputs and scroll to them."""
        if not self._canvas.boxes:
            QMessageBox.information(self, "No boxes", "Draw at least one box first.")
            return
        self._team_section.setVisible(True)
        self._confirm_btn.setText(f"Boxes confirmed ({len(self._canvas.boxes)}) — edit below")
        self._confirm_btn.setEnabled(False)
        self._refresh_rows()
        # Scroll to team number section
        self._canvas_scroll.verticalScrollBar().setValue(
            self._canvas_scroll.verticalScrollBar().maximum()
        )
        # Focus first empty team input
        for i in range(self._rows_lay.count()):
            w = self._rows_lay.itemAt(i).widget()
            if w:
                edit = w.findChild(QLineEdit)
                if edit and not edit.text():
                    edit.setFocus()
                    break

    def _refresh_rows(self):
        # Remove old row widgets
        while self._rows_lay.count():
            item = self._rows_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, box in enumerate(self._canvas.boxes):
            row = _BoxRow(i, box, self._canvas._redraw)
            self._rows_lay.addWidget(row)

        n = len(self._canvas.boxes)
        self._box_count_lbl.setText(f"{n} / {_MAX_ROBOTS} boxes")
        color = "#22c55e" if n <= _MAX_ROBOTS else "#ef4444"
        self._box_count_lbl.setStyleSheet(f"color: {color}; font-size: 12px;")

        # Re-enable confirm button if user draws more boxes after confirming
        if n > 0:
            self._confirm_btn.setEnabled(True)
            self._confirm_btn.setText("Done Drawing  →  Enter Team Numbers")

    # ── Load existing identity (pre-populate boxes from auto-detection) ────────

    def _load_existing_identity(self):
        path = _ROOT / "configs" / "match_identity.json"
        if not path.exists():
            return
        try:
            data   = json.loads(path.read_text())
            robots = data.get("robots", [])
        except Exception:
            return

        for robot in robots[:_MAX_ROBOTS]:   # hard cap at 6
            team  = robot.get("team_number", "")
            if str(team).startswith("UNKNOWN"):
                team = ""
            self._canvas.boxes.append({
                "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                "team":     team,
                "alliance": robot.get("alliance", "unknown"),
                "source":   "auto",
                "track_id": robot.get("track_id"),
            })
        if self._canvas.boxes:
            self._team_section.setVisible(True)
        self._refresh_rows()

    # ── Save ──────────────────────────────────────────────────────────────────

    def _save_and_accept(self):
        boxes = self._canvas.boxes

        # Validate: each drawn box (non-auto) needs a team number
        manual = [b for b in boxes if b.get("source") == "manual"]
        missing = [i + 1 for i, b in enumerate(boxes)
                   if b.get("source") == "manual" and not b.get("team")]
        if missing:
            QMessageBox.warning(
                self, "Missing team numbers",
                f"Please fill in team numbers for box(es): {missing}"
            )
            return

        red_manual  = [b for b in manual if b["alliance"] == "red"]
        blue_manual = [b for b in manual if b["alliance"] == "blue"]
        if len(red_manual) > 3 or len(blue_manual) > 3:
            QMessageBox.warning(
                self, "Too many robots",
                "Maximum 3 red + 3 blue robots. Please remove extra boxes."
            )
            return

        # Build robot list
        robots_json = []
        for i, box in enumerate(boxes):
            team = box.get("team") or f"UNKNOWN_{i}"
            robots_json.append({
                "track_id":         box.get("track_id", i + 1),
                "team_number":      team,
                "alliance":         box.get("alliance", "unknown"),
                "confidence":       1.0 if box.get("source") == "manual" else 0.0,
                "frames_confirmed": 0,
                "user_corrected":   True,
                "bbox_sample":      [box["x1"], box["y1"], box["x2"], box["y2"]],
                "label_frame":      self._current_frame_idx,
            })

        # Load existing file if present (preserve metadata)
        path = _ROOT / "configs" / "match_identity.json"
        try:
            existing = json.loads(path.read_text()) if path.exists() else {}
        except Exception:
            existing = {}

        existing["robots"]         = robots_json
        existing["user_confirmed"] = True
        existing["labeled_manually"] = True

        path.write_text(json.dumps(existing, indent=2))
        self.accept()

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        l = QLabel(text)
        if obj:
            l.setObjectName(obj)
        return l

    def closeEvent(self, event):
        self._cap.release()
        super().closeEvent(event)
