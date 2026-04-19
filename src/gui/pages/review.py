"""
src/gui/pages/review.py — Flag viewer + manual override + video frame preview

Enhancements:
  • Click any row → embedded frame viewer seeks to that exact video frame
  • Scrub ±1 / ±5 / ±30 frames around the event for context
  • Frame viewer is resizable via the vertical splitter
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..theme import C, FLAG_COLOR

_ROOT = Path(__file__).parent.parent.parent.parent
_TIMELINE_PATH  = _ROOT / "data" / "score_timeline.json"
_OVERRIDES_PATH = _ROOT / "data" / "manual_overrides.json"
_RESULTS_PATH   = _ROOT / "data" / "exports" / "results.json"


def _load_timeline() -> list:
    if _TIMELINE_PATH.exists():
        try:
            return json.loads(_TIMELINE_PATH.read_text())
        except Exception:
            pass
    return []


def _save_overrides(overrides: dict):
    _OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OVERRIDES_PATH.write_text(json.dumps(overrides, indent=2))


def _get_video_path() -> str | None:
    """Read video file path from the last pipeline run's results.json."""
    if _RESULTS_PATH.exists():
        try:
            data = json.loads(_RESULTS_PATH.read_text())
            vp = data.get("video_file", "")
            return vp if vp and Path(vp).exists() else None
        except Exception:
            pass
    return None


_FLAG_FILTER_OPTIONS = [
    "All flagged",
    "AMBIGUOUS-MANUAL-REVIEW",
    "INFERRED-LOW-CONF",
    "OPR-WEIGHTED",
    "TEAM-NUMBER-UNCONFIRMED",
]

_REVIEWABLE_FLAGS = {
    "AMBIGUOUS-MANUAL-REVIEW",
    "INFERRED-LOW-CONF",
    "TEAM-NUMBER-UNCONFIRMED",
    "OPR-WEIGHTED",
}


# ── Embedded video frame viewer ───────────────────────────────────────────────

class _FrameViewer(QFrame):
    """
    Shows a single video frame for the selected scoring event.

    load_event() seeks to the event frame.
    Scrub buttons (and ← → arrow keys) step through nearby frames.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self._video_path: str | None = None
        self._cap: cv2.VideoCapture | None = None
        self._base_fid: int = 0
        self._offset:   int = 0
        self._build_ui()

    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        self._info_lbl = QLabel(
            "Click a flagged event above to preview its video frame")
        self._info_lbl.setObjectName("muted")
        self._info_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._info_lbl)

        self._frame_lbl = QLabel()
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_lbl.setMinimumHeight(260)
        self._frame_lbl.setStyleSheet(
            f"background: {C['bg_deep']}; border-radius: 4px;")
        lay.addWidget(self._frame_lbl, 1)

        ctrl = QHBoxLayout()
        ctrl.addStretch()
        for label, delta in [("◀◀ −30", -30), ("◀ −5", -5), ("◀ −1", -1)]:
            btn = QPushButton(label)
            btn.setObjectName("sec")
            btn.setFixedWidth(74)
            btn.clicked.connect(lambda _, d=delta: self._step(d))
            ctrl.addWidget(btn)

        self._fid_lbl = QLabel("—")
        self._fid_lbl.setObjectName("muted")
        self._fid_lbl.setFixedWidth(88)
        self._fid_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ctrl.addWidget(self._fid_lbl)

        for label, delta in [("+1 ▶", 1), ("+5 ▶", 5), ("+30 ▶▶", 30)]:
            btn = QPushButton(label)
            btn.setObjectName("sec")
            btn.setFixedWidth(74)
            btn.clicked.connect(lambda _, d=delta: self._step(d))
            ctrl.addWidget(btn)

        ctrl.addStretch()
        lay.addLayout(ctrl)

    # ── public ────────────────────────────────────────────────────────────────

    def load_event(self, event: dict, video_path: str):
        if video_path != self._video_path:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(video_path)
            self._video_path = video_path

        self._base_fid = event.get("frame_id", 0)
        self._offset   = 0

        team = event.get("team_number", "?")
        ts   = event.get("timestamp_s", 0)
        flag = event.get("flag", "")
        conf = event.get("confidence", 0)
        self._info_lbl.setText(
            f"Team {team}  ·  {ts:.1f}s  ·  {flag}  ·  {conf:.0%} conf  "
            f"  (← → keys or buttons to scrub)")
        self._render()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Left:
            self._step(-1)
        elif ev.key() == Qt.Key.Key_Right:
            self._step(1)
        else:
            super().keyPressEvent(ev)

    # ── private ───────────────────────────────────────────────────────────────

    def _step(self, delta: int):
        self._offset += delta
        self._render()

    def _render(self):
        if self._cap is None or not self._video_path:
            return
        fid = max(0, self._base_fid + self._offset)
        self._fid_lbl.setText(f"Frame {fid}")

        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._video_path)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._frame_lbl.setText(f"Could not read frame {fid}")
            return

        cv2.putText(frame, f"Frame {fid}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                    cv2.LINE_AA)

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)

        avail_w = max(self._frame_lbl.width(), 640)
        avail_h = max(self._frame_lbl.height(), 260)
        pix = pix.scaled(avail_w, avail_h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)
        self._frame_lbl.setPixmap(pix)


# ── Main page ─────────────────────────────────────────────────────────────────

class ReviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._overrides: dict[int, str] = {}
        self._events:    list[dict] = []
        self._override_combos: dict[int, QComboBox] = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 28, 32, 28)
        outer.setSpacing(16)

        outer.addWidget(self._lbl("Manual Review", "h1"))
        outer.addWidget(self._lbl(
            "Events flagged for low confidence or ambiguity. "
            "Click a row to preview the video frame.", "muted"))

        # Controls row
        ctrl = QHBoxLayout()
        ctrl.addWidget(self._lbl("Filter:"))
        self._filter_combo = QComboBox()
        self._filter_combo.addItems(_FLAG_FILTER_OPTIONS)
        self._filter_combo.setFixedWidth(240)
        self._filter_combo.currentIndexChanged.connect(self.reload)
        ctrl.addWidget(self._filter_combo)
        ctrl.addStretch()
        self._count_lbl = self._lbl("", "muted")
        ctrl.addWidget(self._count_lbl)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("sec")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self.reload)
        ctrl.addWidget(refresh_btn)
        save_btn = QPushButton("Apply Overrides")
        save_btn.clicked.connect(self._save_overrides)
        ctrl.addWidget(save_btn)
        outer.addLayout(ctrl)

        # Summary card
        self._summary = QFrame()
        self._summary.setObjectName("card_hi")
        sum_lay = QHBoxLayout(self._summary)
        sum_lay.setContentsMargins(16, 12, 16, 12)
        sum_lay.setSpacing(32)
        self._stat_widgets: dict[str, QLabel] = {}
        for key, label, color in [
            ("ambiguous", "Ambiguous",    C["warning"]),
            ("low_conf",  "Low Conf",     C["danger"]),
            ("opr",       "OPR-Weighted", C.get("purple", C["accent"])),
            ("unconf",    "Team Unconf",  C["text_muted"]),
        ]:
            box = QVBoxLayout()
            val = QLabel("—")
            val.setStyleSheet(
                f"font-size: 22px; font-weight: 700; color: {color}; "
                f"font-family: 'Fira Code', monospace;")
            box.addWidget(val)
            lbl = QLabel(label)
            lbl.setObjectName("kpi_lbl")
            box.addWidget(lbl)
            sum_lay.addLayout(box)
            self._stat_widgets[key] = val
        sum_lay.addStretch()
        outer.addWidget(self._summary)

        # Vertical splitter: events table ↕ frame viewer
        splitter = QSplitter(Qt.Orientation.Vertical)

        top = QWidget()
        top_lay = QVBoxLayout(top)
        top_lay.setContentsMargins(0, 0, 0, 0)
        top_lay.setSpacing(6)
        top_lay.addWidget(self._lbl("FLAGGED EVENTS", "kpi_lbl"))

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Time (s)", "Ball ID", "Current Team", "Flag",
             "Confidence", "Method", "Override To"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(200)
        self._table.setSortingEnabled(True)
        self._table.cellClicked.connect(self._on_row_clicked)
        top_lay.addWidget(self._table)
        splitter.addWidget(top)

        bot = QWidget()
        bot_lay = QVBoxLayout(bot)
        bot_lay.setContentsMargins(0, 0, 0, 0)
        bot_lay.setSpacing(4)
        bot_lay.addWidget(self._lbl("FRAME PREVIEW", "kpi_lbl"))
        self._frame_viewer = _FrameViewer()
        bot_lay.addWidget(self._frame_viewer)
        splitter.addWidget(bot)

        splitter.setSizes([300, 380])
        outer.addWidget(splitter, 1)

        self.reload()

    # ── data ──────────────────────────────────────────────────────────────────

    def reload(self):
        all_events  = _load_timeline()
        flag_filter = self._filter_combo.currentText()
        all_flagged = [e for e in all_events
                       if e.get("flag") and e["flag"] in _REVIEWABLE_FLAGS]
        flagged     = (all_flagged if flag_filter == "All flagged"
                       else [e for e in all_flagged
                             if e.get("flag") == flag_filter])
        self._events = flagged
        self._count_lbl.setText(f"{len(flagged)} events")

        counts = {
            "ambiguous": sum(1 for e in all_flagged
                             if e["flag"] == "AMBIGUOUS-MANUAL-REVIEW"),
            "low_conf":  sum(1 for e in all_flagged
                             if e["flag"] == "INFERRED-LOW-CONF"),
            "opr":       sum(1 for e in all_flagged
                             if e["flag"] == "OPR-WEIGHTED"),
            "unconf":    sum(1 for e in all_flagged
                             if e["flag"] == "TEAM-NUMBER-UNCONFIRMED"),
        }
        for key, val in counts.items():
            self._stat_widgets[key].setText(str(val))

        self._populate_table(flagged, all_events)

    def _populate_table(self, events: list, all_events: list):
        teams = sorted({
            e.get("team_number", "")
            for e in all_events
            if e.get("team_number") not in ("", "UNATTRIBUTED", "REPLACED")
        })
        override_options = ["— no change —"] + teams

        self._override_combos.clear()
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(events))

        for row, ev in enumerate(events):
            flag = ev.get("flag", "")
            conf = ev.get("confidence", 0.0)
            team = ev.get("team_number", "?")

            def cell(text, align=Qt.AlignmentFlag.AlignLeft):
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._table.setItem(row, 0, cell(f"{ev.get('timestamp_s', 0):.1f}",
                                              Qt.AlignmentFlag.AlignRight))
            self._table.setItem(row, 1, cell(str(ev.get("ball_track_id", "?")),
                                              Qt.AlignmentFlag.AlignCenter))
            self._table.setItem(row, 2, cell(team))

            flag_item = cell(flag)
            if flag in FLAG_COLOR:
                flag_item.setForeground(QColor(FLAG_COLOR[flag]))
            self._table.setItem(row, 3, flag_item)

            conf_item = cell(f"{conf:.0%}", Qt.AlignmentFlag.AlignCenter)
            col = (C["success"] if conf >= 0.85 else
                   C["warning"] if conf >= 0.50 else C["danger"])
            conf_item.setForeground(QColor(col))
            self._table.setItem(row, 4, conf_item)
            self._table.setItem(row, 5, cell(ev.get("method", "?")))

            combo = QComboBox()
            combo.addItems(override_options)
            if row in self._overrides:
                idx = combo.findText(self._overrides[row])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._table.setCellWidget(row, 6, combo)
            self._override_combos[row] = combo

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    def _on_row_clicked(self, row: int, _col: int):
        if row >= len(self._events):
            return
        ev = self._events[row]
        vp = _get_video_path()
        if vp:
            self._frame_viewer.load_event(ev, vp)
            self._frame_viewer.setFocus()
        else:
            self._frame_viewer._info_lbl.setText(
                "Video file not found — ensure the original video is still accessible "
                "at the path used when the pipeline ran.")

    def _save_overrides(self):
        overrides: dict[str, str] = {}
        for row, combo in self._override_combos.items():
            choice = combo.currentText()
            if choice and choice != "— no change —" and row < len(self._events):
                ev  = self._events[row]
                key = f"{ev.get('frame_id')}_{ev.get('ball_track_id')}"
                overrides[key] = choice
                self._overrides[row] = choice
        if overrides:
            _save_overrides(overrides)
            self._count_lbl.setText(
                f"{len(overrides)} override(s) saved → data/manual_overrides.json")

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
