"""
src/gui/pages/alliance.py — Alliance builder + TBA pick list
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..theme import C, TEAM_COLORS

_ROOT = Path(__file__).parent.parent.parent.parent


def _load_pick_list() -> list:
    path = _ROOT / "data" / "exports" / "pick_list.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # May be a list or {"pick_list": [...]}
            if isinstance(data, list):
                return data
            return data.get("pick_list", [])
        except Exception:
            pass
    return []


def _load_tba_config() -> dict:
    path = _ROOT / "configs" / "tba_config.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


class _FetchWorker(QThread):
    done    = pyqtSignal(list, str)   # pick_list, status_msg
    error   = pyqtSignal(str)

    def __init__(self, event_key: str, our_team: str):
        super().__init__()
        self.event_key = event_key
        self.our_team  = our_team

    def run(self):
        import sys
        src = str(_ROOT / "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        try:
            from tba_client import get_event_team_stats, TBAAuthError, TBANotFoundError
            from alliance_builder import build_team_composite_scores, generate_pick_list

            stats = get_event_team_stats(self.event_key)
            composite = build_team_composite_scores(
                self.event_key, self.our_team,
                video_analysis_results={},
                driving_results={},
            )
            pick_list = generate_pick_list(
                self.our_team, self.event_key, composite, top_n=20
            )
            self.done.emit(pick_list, f"Loaded {len(pick_list)} teams from TBA")
        except Exception as exc:
            self.error.emit(str(exc))


class AlliancePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _FetchWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        root.addWidget(self._lbl("Alliance Builder", "h1"))
        root.addWidget(self._lbl(
            "Pick list ranked by composite score (OPR + EPA + video data).", "muted"))

        # Config card
        cfg_frame = QFrame()
        cfg_frame.setObjectName("card")
        cfg_lay = QHBoxLayout(cfg_frame)
        cfg_lay.setContentsMargins(16, 14, 16, 14)
        cfg_lay.setSpacing(12)

        tba_cfg = _load_tba_config()

        cfg_lay.addWidget(self._lbl("Event key:"))
        self._event_edit = QLineEdit()
        self._event_edit.setText(tba_cfg.get("event_key", ""))
        self._event_edit.setPlaceholderText("e.g. 2026txhou")
        self._event_edit.setFixedWidth(130)
        cfg_lay.addWidget(self._event_edit)

        cfg_lay.addWidget(self._lbl("Our team:"))
        self._team_edit = QLineEdit()
        self._team_edit.setText(str(tba_cfg.get("our_team_number", "")))
        self._team_edit.setPlaceholderText("e.g. 1234")
        self._team_edit.setFixedWidth(80)
        cfg_lay.addWidget(self._team_edit)

        cfg_lay.addStretch()

        self._fetch_btn = QPushButton("Fetch from TBA")
        self._fetch_btn.clicked.connect(self._fetch)
        cfg_lay.addWidget(self._fetch_btn)

        self._local_btn = QPushButton("Load Local")
        self._local_btn.setObjectName("sec")
        self._local_btn.clicked.connect(self._load_local)
        cfg_lay.addWidget(self._local_btn)

        self._status_lbl = self._lbl("", "muted")
        cfg_lay.addWidget(self._status_lbl)

        root.addWidget(cfg_frame)

        # Top 3 recommendation card
        rec = QFrame()
        rec.setObjectName("card_hi")
        rec_lay = QHBoxLayout(rec)
        rec_lay.setContentsMargins(20, 16, 20, 16)
        rec_lay.setSpacing(24)

        self._pick_labels: list[QWidget] = []
        for slot in ("Captain", "Pick 1", "Pick 2"):
            box = QVBoxLayout()
            slot_lbl = QLabel(slot.upper())
            slot_lbl.setObjectName("kpi_lbl")
            box.addWidget(slot_lbl)
            pick_lbl = QLabel("—")
            pick_lbl.setObjectName("kpi_num")
            pick_lbl.setStyleSheet(f"font-size: 20px; color: {C['accent_hi']};")
            box.addWidget(pick_lbl)
            score_lbl = QLabel("")
            score_lbl.setObjectName("muted")
            box.addWidget(score_lbl)
            rec_lay.addLayout(box)
            self._pick_labels.append((pick_lbl, score_lbl))

        rec_lay.addStretch()

        self._synergy_lbl = QLabel("")
        self._synergy_lbl.setObjectName("muted")
        self._synergy_lbl.setWordWrap(True)
        rec_lay.addWidget(self._synergy_lbl)

        root.addWidget(rec)

        # Pick list table
        root.addWidget(self._lbl("RANKED PICK LIST", "kpi_lbl"))
        self._table = QTableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["Rank", "Team", "Composite", "Style", "OPR", "Video Score", "Warnings"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSortingEnabled(True)
        self._table.setMinimumHeight(300)
        root.addWidget(self._table)

        self._load_local()

    # ── actions ───────────────────────────────────────────────────────────────

    def _fetch(self):
        event_key = self._event_edit.text().strip()
        our_team  = self._team_edit.text().strip()
        if not event_key:
            self._status_lbl.setText("Enter an event key first.")
            return

        self._fetch_btn.setEnabled(False)
        self._status_lbl.setText("Fetching…")

        self._worker = _FetchWorker(event_key, our_team)
        self._worker.done.connect(self._on_fetched)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_fetched(self, pick_list: list, msg: str):
        self._fetch_btn.setEnabled(True)
        self._status_lbl.setText(msg)
        self._populate(pick_list)

    def _on_error(self, msg: str):
        self._fetch_btn.setEnabled(True)
        self._status_lbl.setText(f"Error: {msg}")

    def _load_local(self):
        pick_list = _load_pick_list()
        if pick_list:
            self._status_lbl.setText(f"Loaded {len(pick_list)} teams from local file")
            self._populate(pick_list)
        else:
            self._status_lbl.setText("No local data — run pipeline or fetch from TBA")

    # ── population ────────────────────────────────────────────────────────────

    def _populate(self, pick_list: list):
        # Top 3 picks
        for i, (pick_lbl, score_lbl) in enumerate(self._pick_labels):
            if i < len(pick_list):
                entry = pick_list[i]
                pick_lbl.setText(str(entry.get("team_number", "—")))
                score_lbl.setText(f"{entry.get('composite_score', 0):.3f}")
            else:
                pick_lbl.setText("—")
                score_lbl.setText("")

        # Table
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(pick_list))

        for row, entry in enumerate(pick_list):
            def cell(text, align=Qt.AlignmentFlag.AlignLeft):
                item = QTableWidgetItem(str(text))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                item.setTextAlignment(align | Qt.AlignmentFlag.AlignVCenter)
                return item

            self._table.setItem(row, 0, cell(
                entry.get("rank", row + 1), Qt.AlignmentFlag.AlignCenter))
            self._table.setItem(row, 1, cell(entry.get("team_number", "?")))

            score = entry.get("composite_score", 0)
            score_item = cell(f"{score:.3f}", Qt.AlignmentFlag.AlignRight)
            col = (C["success"] if score >= 0.7 else
                   C["warning"] if score >= 0.4 else C["danger"])
            score_item.setForeground(QColor(col))
            self._table.setItem(row, 2, score_item)

            self._table.setItem(row, 3, cell(entry.get("driving_style", "—")))
            self._table.setItem(row, 4, cell(
                f"{entry.get('tba_opr', 0):.1f}" if entry.get("tba_opr") else "—",
                Qt.AlignmentFlag.AlignRight))
            self._table.setItem(row, 5, cell(
                f"{entry.get('video_score_rate', 0):.2f}"
                if entry.get("video_score_rate") is not None else "—",
                Qt.AlignmentFlag.AlignRight))

            warnings = entry.get("warnings", [])
            w_item = cell("; ".join(warnings) if warnings else "")
            if warnings:
                w_item.setForeground(QColor(C["warning"]))
            self._table.setItem(row, 6, w_item)

        self._table.resizeColumnsToContents()
        self._table.setSortingEnabled(True)

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
