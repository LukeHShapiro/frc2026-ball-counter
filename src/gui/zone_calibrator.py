"""
src/gui/zone_calibrator.py — Draw scoring-zone boxes on a video frame.

Usage:
    dlg = ZoneCalibratorDialog(video_path, config_path, parent)
    dlg.exec()

The user drags two rectangles (red_goal, blue_goal) on a sample frame.
Confirmed coordinates are saved to field_config.json["scoring_zones"].
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QPoint, QRect, QSize
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
    QSizePolicy, QFrame,
)

from .theme import C

_ZONE_META = {
    "red_goal":  {"color": QColor(220, 60,  60,  180), "label": "Red Goal"},
    "blue_goal": {"color": QColor(60,  100, 220, 180), "label": "Blue Goal"},
}


# ── Frame canvas ──────────────────────────────────────────────────────────────

class _FrameCanvas(QLabel):
    """
    Displays a video frame and lets the user drag rectangles to define zones.

    active_zone: which zone key is currently being drawn ("red_goal" / "blue_goal")
    zones:       {zone_key: QRect (in *display* coordinates)}
    """

    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._src_pixmap = pixmap
        self._scale = 1.0          # display → source scale factor
        self._offset = QPoint(0, 0)
        self.zones: dict[str, QRect] = {}
        self.active_zone: str = "red_goal"
        self._drag_start: QPoint | None = None
        self._drag_current: QRect | None = None

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._repaint()

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _fit(self):
        """Compute scale + top-left offset so image fits the widget."""
        sw, sh = self._src_pixmap.width(), self._src_pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / sw, wh / sh)
        dw, dh = int(sw * scale), int(sh * scale)
        ox = (ww - dw) // 2
        oy = (wh - dh) // 2
        self._scale  = scale
        self._offset = QPoint(ox, oy)
        return scale, ox, oy, dw, dh

    def _to_src(self, pt: QPoint) -> QPoint:
        """Convert widget-space point → source-image-space point."""
        return QPoint(
            int((pt.x() - self._offset.x()) / self._scale),
            int((pt.y() - self._offset.y()) / self._scale),
        )

    def _to_disp(self, r: QRect) -> QRect:
        """Convert source-image QRect → display-space QRect."""
        x = int(r.x() * self._scale) + self._offset.x()
        y = int(r.y() * self._scale) + self._offset.y()
        w = int(r.width()  * self._scale)
        h = int(r.height() * self._scale)
        return QRect(x, y, w, h)

    # ── Paint ─────────────────────────────────────────────────────────────────

    def _repaint(self):
        scale, ox, oy, dw, dh = self._fit()
        scaled = self._src_pixmap.scaled(
            dw, dh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QPixmap(self.width(), self.height())
        canvas.fill(QColor(C["bg_base"]))

        p = QPainter(canvas)
        p.drawPixmap(ox, oy, scaled)

        # Draw confirmed zones
        for key, rect_src in self.zones.items():
            if rect_src.isNull():
                continue
            color = _ZONE_META[key]["color"]
            label = _ZONE_META[key]["label"]
            r = self._to_disp(rect_src)

            pen = QPen(color, 2)
            p.setPen(pen)
            p.drawRect(r)

            fill = QColor(color)
            fill.setAlpha(40)
            p.fillRect(r, fill)

            # Label text
            p.setPen(QColor(255, 255, 255))
            p.drawText(r.adjusted(4, 2, 0, 0), Qt.AlignmentFlag.AlignTop, label)

        # Draw in-progress drag
        if self._drag_current and not self._drag_current.isNull():
            color = _ZONE_META[self.active_zone]["color"]
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(self._drag_current)

        p.end()
        self.setPixmap(canvas)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_start = ev.position().toPoint()
            self._drag_current = None

    def mouseMoveEvent(self, ev):
        if self._drag_start is None:
            return
        cur = ev.position().toPoint()
        self._drag_current = QRect(self._drag_start, cur).normalized()
        self._repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._drag_start is not None:
            cur = ev.position().toPoint()
            disp_rect = QRect(self._drag_start, cur).normalized()
            if disp_rect.width() > 10 and disp_rect.height() > 10:
                # Convert to source coordinates
                src_x = int((disp_rect.x() - self._offset.x()) / self._scale)
                src_y = int((disp_rect.y() - self._offset.y()) / self._scale)
                src_w = int(disp_rect.width()  / self._scale)
                src_h = int(disp_rect.height() / self._scale)
                self.zones[self.active_zone] = QRect(src_x, src_y, src_w, src_h)
            self._drag_start = None
            self._drag_current = None
            self._repaint()

    def resizeEvent(self, ev):
        self._repaint()

    # ── Public ────────────────────────────────────────────────────────────────

    def source_zones(self) -> dict[str, list[int]]:
        """Return zones as {name: [x1, y1, x2, y2]} in source-image pixels."""
        result = {}
        for key, r in self.zones.items():
            if not r.isNull():
                result[key] = [r.x(), r.y(), r.x() + r.width(), r.y() + r.height()]
        return result


# ── Dialog ────────────────────────────────────────────────────────────────────

class ZoneCalibratorDialog(QDialog):
    def __init__(self, video_path: str | Path, config_path: str | Path, parent=None):
        super().__init__(parent)
        self._video_path  = Path(video_path)
        self._config_path = Path(config_path)
        self.setWindowTitle("Calibrate Scoring Zones")
        self.setMinimumSize(900, 600)
        self.resize(1100, 700)
        self.setModal(True)

        frame_px = self._sample_frame()
        self._build_ui(frame_px)

    # ── Frame sampling ────────────────────────────────────────────────────────

    def _sample_frame(self) -> QPixmap:
        """Extract a frame at ~40% of video duration (mid-match action)."""
        cap = cv2.VideoCapture(str(self._video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(total * 0.40)))
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            px = QPixmap(960, 540)
            px.fill(QColor(30, 30, 30))
            return px

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(img)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self, frame_px: QPixmap):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # Instructions
        instr = QLabel(
            "Draw a box around each goal zone. "
            "Select a zone button, then click and drag on the frame. "
            "You can redraw any zone by selecting it and dragging again."
        )
        instr.setWordWrap(True)
        instr.setObjectName("muted")
        root.addWidget(instr)

        # Zone selector row
        sel_row = QHBoxLayout()
        self._zone_btns: dict[str, QPushButton] = {}
        for key, meta in _ZONE_META.items():
            btn = QPushButton(f"  {meta['label']}")
            btn.setCheckable(True)
            btn.setFixedHeight(32)
            c = meta["color"]
            color_hex = f"#{c.red():02x}{c.green():02x}{c.blue():02x}"
            btn.setStyleSheet(f"""
                QPushButton {{
                    border: 2px solid {color_hex};
                    border-radius: 4px;
                    color: {color_hex};
                    font-weight: 600;
                    padding: 0 12px;
                    background: transparent;
                    text-align: left;
                }}
                QPushButton:checked {{
                    background: {color_hex}33;
                }}
            """)
            btn.clicked.connect(lambda _, k=key: self._select_zone(k))
            sel_row.addWidget(btn)
            self._zone_btns[key] = btn
        sel_row.addStretch()

        clear_btn = QPushButton("Clear Selection")
        clear_btn.setObjectName("sec")
        clear_btn.setFixedHeight(32)
        clear_btn.clicked.connect(self._clear_active)
        sel_row.addWidget(clear_btn)
        root.addLayout(sel_row)

        # Canvas
        self._canvas = _FrameCanvas(frame_px)
        root.addWidget(self._canvas, 1)

        # Coordinate readout
        self._coord_lbl = QLabel("No zones defined yet.")
        self._coord_lbl.setObjectName("muted")
        self._coord_lbl.setStyleSheet("font-family: 'Fira Code', monospace; font-size: 11px;")
        root.addWidget(self._coord_lbl)

        # Bottom buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("sec")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._save_btn = QPushButton("Save Zones")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._save)
        btn_row.addWidget(self._save_btn)
        root.addLayout(btn_row)

        # Connect canvas change to coord readout
        # (poll via a simple override of _repaint)
        orig_repaint = self._canvas._repaint
        def _hooked_repaint():
            orig_repaint()
            self._update_coords()
        self._canvas._repaint = _hooked_repaint

        # Select red_goal by default
        self._select_zone("red_goal")

    def _select_zone(self, key: str):
        self._canvas.active_zone = key
        for k, btn in self._zone_btns.items():
            btn.setChecked(k == key)

    def _clear_active(self):
        key = self._canvas.active_zone
        self._canvas.zones.pop(key, None)
        self._canvas._repaint()

    def _update_coords(self):
        zones = self._canvas.source_zones()
        if not zones:
            self._coord_lbl.setText("No zones defined yet.")
            return
        parts = []
        for key in _ZONE_META:
            if key in zones:
                z = zones[key]
                parts.append(f"{key}: [{z[0]}, {z[1]}, {z[2]}, {z[3]}]")
            else:
                parts.append(f"{key}: not set")
        self._coord_lbl.setText("   |   ".join(parts))

    def _save(self):
        zones = self._canvas.source_zones()
        if not zones:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No zones", "Draw at least one goal zone before saving.")
            return

        # Load existing config
        cfg = {}
        if self._config_path.exists():
            try:
                cfg = json.loads(self._config_path.read_text())
            except Exception:
                pass

        existing = cfg.get("scoring_zones", {})
        # Preserve comment key; merge new zones
        comment = existing.get("_comment", "")
        existing.update(zones)
        if comment:
            existing["_comment"] = comment
        cfg["scoring_zones"] = existing

        self._config_path.write_text(json.dumps(cfg, indent=4))
        self.accept()
