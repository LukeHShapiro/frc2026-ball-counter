"""
src/gui/pages/scores.py — Per-robot score table + charts
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..chart import ChartCanvas, score_bar_chart, score_timeline_chart
from ..theme import C, TEAM_COLORS, FLAG_COLOR

_ROOT = Path(__file__).parent.parent.parent.parent


def _load_results() -> tuple[dict, list]:
    """Return (final_scores, score_timeline) from exported JSON files."""
    scores: dict = {}
    timeline: list = []

    results_path  = _ROOT / "data" / "exports" / "results.json"
    timeline_path = _ROOT / "data" / "score_timeline.json"

    if results_path.exists():
        try:
            data   = json.loads(results_path.read_text())
            scores = data.get("final_scores", {})
        except Exception:
            pass

    if timeline_path.exists():
        try:
            timeline = json.loads(timeline_path.read_text())
        except Exception:
            pass

    return scores, timeline


class _KpiCard(QFrame):
    def __init__(self, team: str, data: dict, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(140)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(4)

        dot = QLabel("●")
        dot.setStyleSheet(f"color: {color}; font-size: 9px;")
        lay.addWidget(dot)

        lbl = QLabel("TEAM")
        lbl.setObjectName("kpi_lbl")
        lay.addWidget(lbl)

        team_lbl = QLabel(team)
        team_lbl.setObjectName("kpi_num")
        team_lbl.setStyleSheet(f"color: {color}; font-size: 22px;")
        lay.addWidget(team_lbl)

        score = QLabel(str(data.get("score", 0)))
        score.setObjectName("kpi_num")
        lay.addWidget(score)

        score_lbl = QLabel("POINTS")
        score_lbl.setObjectName("kpi_lbl")
        lay.addWidget(score_lbl)

        lay.addSpacing(6)

        conf_row = QHBoxLayout()
        for key, col, short in [
            ("high_conf", C["success"], "H"),
            ("med_conf",  C["warning"], "M"),
            ("low_conf",  C["danger"],  "L"),
        ]:
            val = data.get(key, 0)
            chip = QLabel(f"{short}:{val}")
            chip.setStyleSheet(
                f"color: {col}; font-size: 10px; font-family: 'Fira Code', monospace;"
            )
            chip.setToolTip({"H": "High confidence", "M": "Medium", "L": "Low"}[short])
            conf_row.addWidget(chip)
        conf_row.addStretch()
        lay.addLayout(conf_row)


class ScoresPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(self._lbl("Scores", "h1"))
        hdr.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("sec")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self.reload)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)

        root.addWidget(self._lbl(
            "Per-robot score attribution from the last pipeline run.", "muted"))

        # KPI cards row
        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(12)
        root.addLayout(self._cards_row)
        self._card_widgets: list[QFrame] = []

        # Charts splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._bar_canvas = ChartCanvas(height=3)
        splitter.addWidget(self._bar_canvas)

        self._timeline_canvas = ChartCanvas(height=3)
        splitter.addWidget(self._timeline_canvas)

        splitter.setSizes([400, 500])
        root.addWidget(splitter)

        # Table header row
        tbl_hdr = QHBoxLayout()
        tbl_hdr.addWidget(self._lbl("SCORING EVENTS", "kpi_lbl"))
        tbl_hdr.addStretch()
        self._detail_btn = QPushButton("Show All Events")
        self._detail_btn.setObjectName("sec")
        self._detail_btn.setFixedWidth(130)
        self._detail_btn.setCheckable(True)
        self._detail_btn.toggled.connect(self._on_detail_toggle)
        tbl_hdr.addWidget(self._detail_btn)
        root.addLayout(tbl_hdr)

        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Time (s)", "Team", "Zone", "Method", "Confidence", "Case", "Flag"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(220)
        self._table.setSortingEnabled(True)
        root.addWidget(self._table)

        self.reload()

    # ── data ─────────────────────────────────────────────────────────────────

    def reload(self):
        scores, timeline = _load_results()
        self._timeline = timeline
        self._populate_cards(scores)
        score_bar_chart(self._bar_canvas, scores)
        score_timeline_chart(self._timeline_canvas, timeline)
        self._populate_table(timeline, detail=self._detail_btn.isChecked())

    def _on_detail_toggle(self, checked: bool):
        self._detail_btn.setText("Show Summary" if checked else "Show All Events")
        self._populate_table(getattr(self, "_timeline", []), detail=checked)

    def _populate_cards(self, scores: dict):
        # Clear old cards
        for w in self._card_widgets:
            self._cards_row.removeWidget(w)
            w.deleteLater()
        self._card_widgets.clear()

        teams = [t for t in scores if t not in ("UNATTRIBUTED", "REPLACED")]
        for i, team in enumerate(teams[:6]):
            card = _KpiCard(team, scores[team],
                            TEAM_COLORS[i % len(TEAM_COLORS)])
            self._cards_row.addWidget(card)
            self._card_widgets.append(card)

        self._cards_row.addStretch()

        if not teams:
            # Check whether goal zones look like defaults (likely uncalibrated)
            import json as _json
            _zones_ok = False
            try:
                _cfg = _json.loads(
                    (_ROOT / "configs" / "field_config.json").read_text()
                )
                _sz = _cfg.get("scoring_zones", {})
                # Zones with y > 900 on typical 1080p = below most cameras → not calibrated
                for _z in _sz.values():
                    if isinstance(_z, list) and len(_z) == 4 and _z[1] < 800:
                        _zones_ok = True
                        break
            except Exception:
                pass

            if not _zones_ok:
                msg = QLabel(
                    "Goal zones not calibrated.\n"
                    "Go to Analyze → click  Box Goals  to draw\n"
                    "the red and blue goal areas on a video frame,\n"
                    "then run the pipeline again."
                )
                msg.setStyleSheet(
                    f"color: {C['warning']}; font-size: 12px; "
                    f"font-family: 'Fira Code', monospace; line-height: 1.6;"
                )
                msg.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self._cards_row.addWidget(msg)
                self._card_widgets.append(msg)
            else:
                placeholder = QLabel(
                    "No scoring events detected — pipeline ran but no balls\n"
                    "entered a goal zone. Check zone placement and ball detection."
                )
                placeholder.setObjectName("muted")
                self._cards_row.addWidget(placeholder)
                self._card_widgets.append(placeholder)

    def _populate_table(self, timeline: list, detail: bool = False):
        valid = [e for e in timeline
                 if e.get("team_number") not in ("REPLACED", "UNATTRIBUTED")]

        if not detail:
            # Default: one summary row per team showing their highest-conf event
            by_team: dict[str, list] = {}
            for ev in valid:
                t = ev.get("team_number", "?")
                by_team.setdefault(t, []).append(ev)
            # Pick representative event per team (highest confidence)
            valid = [
                max(evs, key=lambda e: e.get("confidence", 0))
                for evs in by_team.values()
            ]
            valid.sort(key=lambda e: -e.get("confidence", 0))

        self._table.setRowCount(len(valid))
        self._table.setSortingEnabled(False)

        for row, ev in enumerate(valid):
            flag = ev.get("flag", "")
            conf = ev.get("confidence", 0.0)

            def cell(text: str, align=Qt.AlignmentFlag.AlignLeft) -> QTableWidgetItem:
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._table.setItem(row, 0, cell(
                f"{ev.get('timestamp_s', 0):.1f}",
                Qt.AlignmentFlag.AlignRight))
            self._table.setItem(row, 1, cell(ev.get("team_number", "?")))
            self._table.setItem(row, 2, cell(ev.get("zone", "?")))
            self._table.setItem(row, 3, cell(ev.get("method", "?")))

            conf_item = cell(f"{conf:.0%}", Qt.AlignmentFlag.AlignCenter)
            if conf >= 0.85:
                conf_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(C["success"]))
            elif conf >= 0.50:
                conf_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(C["warning"]))
            else:
                conf_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(C["danger"]))
            self._table.setItem(row, 4, conf_item)

            self._table.setItem(row, 5, cell(str(ev.get("case", "?")),
                                              Qt.AlignmentFlag.AlignCenter))

            flag_item = cell(flag if flag else "—")
            if flag and flag in FLAG_COLOR:
                flag_item.setForeground(
                    __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(
                        FLAG_COLOR[flag]))
            self._table.setItem(row, 6, flag_item)

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
