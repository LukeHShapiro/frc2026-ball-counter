"""
src/gui/pages/scores.py — Per-robot score table + charts

Enhanced view:
  - Team selector: filter to one robot or view all at once
  - KPI cards for the selected team (or all teams in current match)
  - Bar chart: current-match scores, optionally filtered
  - Match history chart: selected team's score across every analyzed match
  - Per-match stats table: each analyzed match row with score breakdown
  - Scoring events table: individual events, filtered by team
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..chart import ChartCanvas, score_bar_chart, match_history_chart
from ..theme import C, TEAM_COLORS, FLAG_COLOR

_ROOT = Path(__file__).parent.parent.parent.parent

_ALL_TEAMS = "All Teams"


# ── Data loaders ──────────────────────────────────────────────────────────────

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


def _load_match_history() -> list[dict]:
    """Return list of analyzed matches from match_history.json."""
    path = _ROOT / "data" / "exports" / "match_history.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []


def _load_tba_teams_sync(event_key: str, api_key: str) -> list[str]:
    """Return sorted team numbers from TBA event roster (blocking, run in thread)."""
    try:
        import sys
        sys.path.insert(0, str(_ROOT / "src"))
        from tba_client import get_event_teams
        teams = get_event_teams(event_key)
        return sorted(str(t.get("team_number", "")) for t in teams if t.get("team_number"))
    except Exception:
        return []


# ── KPI card widget ───────────────────────────────────────────────────────────

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


# ── Main page ─────────────────────────────────────────────────────────────────

class ScoresPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._tba_teams: list[str] = []          # populated in background
        self._build_ui()
        self._fetch_tba_teams()                  # async on first load

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(16)

        # ── Header row ────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.addWidget(self._lbl("Scores", "h1"))
        hdr.addStretch()

        # Team selector
        hdr.addWidget(self._lbl("Team:", "kpi_lbl"))
        self._team_combo = QComboBox()
        self._team_combo.setMinimumWidth(140)
        self._team_combo.addItem(_ALL_TEAMS)
        self._team_combo.currentTextChanged.connect(self._on_team_changed)
        hdr.addWidget(self._team_combo)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("sec")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self.reload)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)

        root.addWidget(self._lbl(
            "Per-robot score attribution from the last pipeline run.", "muted"))

        # ── KPI cards row ─────────────────────────────────────────────────────
        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(12)
        root.addLayout(self._cards_row)
        self._card_widgets: list[QWidget] = []

        # ── Charts splitter ───────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._bar_canvas = ChartCanvas(height=3)
        splitter.addWidget(self._bar_canvas)

        self._history_canvas = ChartCanvas(height=3)
        splitter.addWidget(self._history_canvas)

        splitter.setSizes([400, 500])
        root.addWidget(splitter)

        # ── Per-match stats table ─────────────────────────────────────────────
        match_hdr = QHBoxLayout()
        match_hdr.addWidget(self._lbl("MATCH HISTORY", "kpi_lbl"))
        match_hdr.addStretch()
        root.addLayout(match_hdr)

        self._match_table = QTableWidget()
        self._match_table.setColumnCount(6)
        self._match_table.setHorizontalHeaderLabels(
            ["Match", "Team", "Score", "High", "Med", "Low"]
        )
        self._match_table.horizontalHeader().setStretchLastSection(True)
        self._match_table.setMaximumHeight(180)
        self._match_table.setSortingEnabled(True)
        root.addWidget(self._match_table)

        # ── Scoring events table ──────────────────────────────────────────────
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
        self._table.setMinimumHeight(200)
        self._table.setSortingEnabled(True)
        root.addWidget(self._table)

        self.reload()

    # ── TBA team fetching ─────────────────────────────────────────────────────

    def _fetch_tba_teams(self):
        """Fetch TBA event roster in background; update combo when done."""
        try:
            cfg = json.loads((_ROOT / "configs" / "tba_config.json").read_text())
            event_key = cfg.get("event_key", "")
            api_key   = cfg.get("api_key", "")
            if not event_key:
                return
        except Exception:
            return

        def _bg():
            teams = _load_tba_teams_sync(event_key, api_key)
            if teams:
                self._tba_teams = teams
                # Use a single-shot timer to safely update the combo from main thread
                QTimer.singleShot(0, self._refresh_combo_teams)

        threading.Thread(target=_bg, daemon=True).start()

    def _refresh_combo_teams(self):
        """Merge TBA teams + current-match teams into the selector (main thread)."""
        current = self._team_combo.currentText()
        self._team_combo.blockSignals(True)
        self._team_combo.clear()
        self._team_combo.addItem(_ALL_TEAMS)

        # All teams: TBA roster ∪ teams seen in current results
        seen: set[str] = set()
        combined: list[str] = []
        for t in self._tba_teams:
            if t not in seen:
                combined.append(t)
                seen.add(t)
        for t in getattr(self, "_current_match_teams", []):
            if t not in seen:
                combined.append(t)
                seen.add(t)

        for t in sorted(combined, key=lambda x: int(x) if x.isdigit() else 99999):
            self._team_combo.addItem(t)

        # Restore previous selection if still present
        idx = self._team_combo.findText(current)
        self._team_combo.setCurrentIndex(max(0, idx))
        self._team_combo.blockSignals(False)

    # ── Data ──────────────────────────────────────────────────────────────────

    def reload(self):
        scores, timeline = _load_results()
        self._scores   = scores
        self._timeline = timeline
        self._history  = _load_match_history()

        # Track which teams appear in the current match result
        self._current_match_teams = [
            t for t in scores
            if t not in ("UNATTRIBUTED", "REPLACED") and not t.startswith("UNKNOWN_")
        ]

        # Keep combo up to date with any new teams from this run
        self._refresh_combo_teams()

        self._render()

    def _render(self):
        """Re-draw all widgets based on current data + selection."""
        sel = self._team_combo.currentText()
        team_filter = None if sel == _ALL_TEAMS else sel

        self._populate_cards(self._scores, team_filter)

        # Bar chart: filter scores to selected team if needed
        bar_scores = self._scores
        if team_filter:
            bar_scores = {k: v for k, v in self._scores.items() if k == team_filter}
        score_bar_chart(self._bar_canvas, bar_scores)

        # Match history chart
        match_history_chart(self._history_canvas, self._history, team_filter)

        # Per-match stats table
        self._populate_match_table(self._history, team_filter)

        # Scoring events table
        self._populate_table(self._timeline,
                             detail=self._detail_btn.isChecked(),
                             team_filter=team_filter)

    def _on_team_changed(self, _text: str):
        self._render()

    def _on_detail_toggle(self, checked: bool):
        self._detail_btn.setText("Show Summary" if checked else "Show All Events")
        sel = self._team_combo.currentText()
        self._populate_table(
            getattr(self, "_timeline", []),
            detail=checked,
            team_filter=None if sel == _ALL_TEAMS else sel,
        )

    # ── KPI cards ─────────────────────────────────────────────────────────────

    def _populate_cards(self, scores: dict, team_filter: str | None):
        for w in self._card_widgets:
            self._cards_row.removeWidget(w)
            w.deleteLater()
        self._card_widgets.clear()

        teams = [t for t in scores if t not in ("UNATTRIBUTED", "REPLACED")]
        if team_filter:
            teams = [t for t in teams if t == team_filter]

        for i, team in enumerate(teams[:6]):
            card = _KpiCard(team, scores[team], TEAM_COLORS[i % len(TEAM_COLORS)])
            self._cards_row.addWidget(card)
            self._card_widgets.append(card)

        self._cards_row.addStretch()

        if not teams:
            # Check whether goal zones look like defaults (likely uncalibrated)
            _zones_ok = False
            try:
                _cfg = json.loads(
                    (_ROOT / "configs" / "field_config.json").read_text()
                )
                _sz = _cfg.get("scoring_zones", {})
                for _z in _sz.values():
                    if isinstance(_z, list) and len(_z) == 4 and _z[1] < 800:
                        _zones_ok = True
                        break
            except Exception:
                pass

            if team_filter:
                msg_text = (
                    f"Team {team_filter} has no scoring events in the current run.\n"
                    "Select 'All Teams' or run the pipeline on a match featuring this team."
                )
            elif not _zones_ok:
                msg_text = (
                    "Goal zones not calibrated.\n"
                    "Go to Analyze → click  Box Goals  to draw\n"
                    "the red and blue goal areas on a video frame,\n"
                    "then run the pipeline again."
                )
            else:
                msg_text = (
                    "No scoring events detected — pipeline ran but no balls\n"
                    "entered a goal zone. Check zone placement and ball detection."
                )

            msg = QLabel(msg_text)
            msg.setStyleSheet(
                f"color: {C['warning']}; font-size: 12px; "
                f"font-family: 'Fira Code', monospace; line-height: 1.6;"
            )
            msg.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._cards_row.addWidget(msg)
            self._card_widgets.append(msg)

    # ── Per-match stats table ─────────────────────────────────────────────────

    def _populate_match_table(self, history: list[dict], team_filter: str | None):
        """One row per (match, team) pair, filtered if a team is selected."""
        rows: list[tuple] = []
        for entry in history:
            label = entry.get("match_label", "?")
            for team, data in entry.get("teams", {}).items():
                if team in ("UNATTRIBUTED", "REPLACED"):
                    continue
                if team_filter and team != team_filter:
                    continue
                rows.append((
                    label,
                    team,
                    data.get("score", 0),
                    data.get("high_conf", 0),
                    data.get("med_conf", 0),
                    data.get("low_conf", 0),
                ))

        self._match_table.setRowCount(len(rows))
        self._match_table.setSortingEnabled(False)

        def cell(text: str, align=Qt.AlignmentFlag.AlignLeft) -> QTableWidgetItem:
            item = QTableWidgetItem(str(text))
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
            return item

        for r, (label, team, score, hi, med, lo) in enumerate(rows):
            self._match_table.setItem(r, 0, cell(label))
            self._match_table.setItem(r, 1, cell(team))
            self._match_table.setItem(r, 2, cell(str(score), Qt.AlignmentFlag.AlignCenter))
            self._match_table.setItem(r, 3, cell(str(hi),    Qt.AlignmentFlag.AlignCenter))
            self._match_table.setItem(r, 4, cell(str(med),   Qt.AlignmentFlag.AlignCenter))
            self._match_table.setItem(r, 5, cell(str(lo),    Qt.AlignmentFlag.AlignCenter))

        self._match_table.resizeColumnsToContents()
        self._match_table.setSortingEnabled(True)

        if not rows:
            self._match_table.setRowCount(1)
            empty = QTableWidgetItem("No match history — run the pipeline on more matches")
            empty.setForeground(
                __import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(C["text_muted"]))
            empty.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._match_table.setItem(0, 0, empty)
            self._match_table.setSpan(0, 0, 1, 6)

    # ── Scoring events table ──────────────────────────────────────────────────

    def _populate_table(self, timeline: list, detail: bool = False,
                        team_filter: str | None = None):
        valid = [e for e in timeline
                 if e.get("team_number") not in ("REPLACED", "UNATTRIBUTED")]

        if team_filter:
            valid = [e for e in valid if e.get("team_number") == team_filter]

        if not detail:
            by_team: dict[str, list] = {}
            for ev in valid:
                t = ev.get("team_number", "?")
                by_team.setdefault(t, []).append(ev)
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
