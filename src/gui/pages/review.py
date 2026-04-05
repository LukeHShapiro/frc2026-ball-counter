"""
src/gui/pages/review.py — Flag viewer + manual override
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..theme import C, FLAG_COLOR

_ROOT = Path(__file__).parent.parent.parent.parent
_TIMELINE_PATH = _ROOT / "data" / "score_timeline.json"
_OVERRIDES_PATH = _ROOT / "data" / "manual_overrides.json"


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


class ReviewPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._overrides: dict[int, str] = {}   # row_index -> team override
        self._events: list[dict] = []
        self._override_combos: dict[int, QComboBox] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        root.addWidget(self._lbl("Manual Review", "h1"))
        root.addWidget(self._lbl(
            "Events flagged for low confidence or ambiguity. "
            "Override team attribution where needed.", "muted"))

        # Controls
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

        root.addLayout(ctrl)

        # Summary card
        self._summary = QFrame()
        self._summary.setObjectName("card_hi")
        sum_lay = QHBoxLayout(self._summary)
        sum_lay.setContentsMargins(16, 12, 16, 12)
        sum_lay.setSpacing(32)

        self._stat_widgets: dict[str, QLabel] = {}
        for key, label, color in [
            ("ambiguous", "Ambiguous",   C["warning"]),
            ("low_conf",  "Low Conf",    C["danger"]),
            ("opr",       "OPR-Weighted", C["purple"]),
            ("unconf",    "Team Unconf", C["text_muted"]),
        ]:
            box = QVBoxLayout()
            val = QLabel("—")
            val.setStyleSheet(
                f"font-size: 22px; font-weight: 700; color: {color}; "
                f"font-family: 'Fira Code', monospace;"
            )
            box.addWidget(val)
            lbl = QLabel(label)
            lbl.setObjectName("kpi_lbl")
            box.addWidget(lbl)
            sum_lay.addLayout(box)
            self._stat_widgets[key] = val

        sum_lay.addStretch()
        root.addWidget(self._summary)

        # Table
        root.addWidget(self._lbl("FLAGGED EVENTS", "kpi_lbl"))
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Time (s)", "Ball ID", "Current Team", "Flag",
             "Confidence", "Method", "Override To"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(340)
        root.addWidget(self._table)

        self.reload()

    def reload(self):
        all_events = _load_timeline()
        flag_filter = self._filter_combo.currentText()

        flagged = [
            e for e in all_events
            if e.get("flag") and e["flag"] in _REVIEWABLE_FLAGS
        ]

        if flag_filter != "All flagged":
            flagged = [e for e in flagged if e.get("flag") == flag_filter]

        self._events = flagged
        self._count_lbl.setText(f"{len(flagged)} events")

        # Update summary counts (always from all flagged regardless of filter)
        all_flagged = [e for e in all_events
                       if e.get("flag") and e["flag"] in _REVIEWABLE_FLAGS]
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
        # Collect unique team numbers for override options
        teams = sorted({
            e.get("team_number", "")
            for e in all_events
            if e.get("team_number") not in ("", "UNATTRIBUTED", "REPLACED")
        })
        override_options = ["— no change —"] + teams

        self._override_combos.clear()
        self._table.setRowCount(len(events))

        for row, ev in enumerate(events):
            flag  = ev.get("flag", "")
            conf  = ev.get("confidence", 0.0)
            team  = ev.get("team_number", "?")

            def cell(text, align=Qt.AlignmentFlag.AlignLeft):
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._table.setItem(row, 0, cell(
                f"{ev.get('timestamp_s', 0):.1f}",
                Qt.AlignmentFlag.AlignRight))
            self._table.setItem(row, 1, cell(
                str(ev.get("ball_track_id", "?")),
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

            # Override combo
            combo = QComboBox()
            combo.addItems(override_options)
            # Restore previous override if any
            if row in self._overrides:
                idx = combo.findText(self._overrides[row])
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._table.setCellWidget(row, 6, combo)
            self._override_combos[row] = combo

        self._table.resizeColumnsToContents()

    def _save_overrides(self):
        overrides: dict[str, str] = {}
        for row, combo in self._override_combos.items():
            choice = combo.currentText()
            if choice and choice != "— no change —" and row < len(self._events):
                ev = self._events[row]
                key = f"{ev.get('frame_id')}_{ev.get('ball_track_id')}"
                overrides[key] = choice
                self._overrides[row] = choice

        if overrides:
            _save_overrides(overrides)
            self._count_lbl.setText(
                f"{len(overrides)} override(s) saved → data/manual_overrides.json"
            )

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
