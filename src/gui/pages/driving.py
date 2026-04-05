"""
src/gui/pages/driving.py — Driving style analysis cards
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from ..chart import ChartCanvas, style_radar_chart
from ..theme import C, STYLE_COLOR, TEAM_COLORS

_ROOT = Path(__file__).parent.parent.parent.parent


def _load_driving() -> dict:
    path = _ROOT / "data" / "exports" / "driving_report.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


class _StyleBadge(QLabel):
    def __init__(self, style: str, parent=None):
        super().__init__(style, parent)
        self.setObjectName("badge")
        col = STYLE_COLOR.get(style, C["accent"])
        self.setStyleSheet(
            f"background-color: {col}22; color: {col}; "
            f"border: 1px solid {col}55; border-radius: 4px; "
            f"padding: 2px 8px; font-size: 11px; font-weight: 700;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)


class _MetricRow(QWidget):
    def __init__(self, label: str, value: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lbl = QLabel(label)
        lbl.setObjectName("muted")
        lbl.setFixedWidth(200)
        lay.addWidget(lbl)

        val = QLabel(value)
        val.setStyleSheet(f"color: {C['text']}; font-family: 'Fira Code', monospace;")
        lay.addWidget(val)
        lay.addStretch()


class _RobotCard(QFrame):
    def __init__(self, team: str, data: dict, color: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        # Team + style badge
        hdr = QHBoxLayout()
        team_lbl = QLabel(f"Team {team}")
        team_lbl.setStyleSheet(
            f"color: {color}; font-size: 15px; font-weight: 700;"
        )
        hdr.addWidget(team_lbl)
        hdr.addStretch()

        primary = data.get("style", data.get("primary_style", "—"))
        hdr.addWidget(_StyleBadge(primary))
        lay.addLayout(hdr)

        # Secondary style
        secondary = data.get("secondary", data.get("secondary_style"))
        if secondary:
            sec_row = QHBoxLayout()
            sec_row.addWidget(QLabel("Secondary:"))
            sec_row.addWidget(_StyleBadge(secondary))
            sec_row.addStretch()
            lay.addLayout(sec_row)

        # Confidence bar
        conf = data.get("confidence", 0.0)
        conf_lbl = QLabel(f"Confidence  {conf:.0%}")
        conf_lbl.setObjectName("muted")
        lay.addWidget(conf_lbl)

        bar_bg = QFrame()
        bar_bg.setFixedHeight(4)
        bar_bg.setStyleSheet(
            f"background-color: {C['bg_elevated']}; border-radius: 2px;"
        )
        bar_fill = QFrame(bar_bg)
        bar_fill.setFixedHeight(4)
        bar_fill.setFixedWidth(max(4, int(conf * 200)))
        col = (C["success"] if conf >= 0.75 else
               C["warning"] if conf >= 0.50 else C["danger"])
        bar_fill.setStyleSheet(f"background-color: {col}; border-radius: 2px;")
        lay.addWidget(bar_bg)

        # Key evidence
        evidence = data.get("key_evidence", [])
        if evidence:
            ev_lbl = QLabel("Evidence")
            ev_lbl.setObjectName("kpi_lbl")
            lay.addWidget(ev_lbl)
            for e in evidence[:3]:
                e_row = QLabel(f"· {e}")
                e_row.setObjectName("muted")
                e_row.setWordWrap(True)
                lay.addWidget(e_row)

        # Key metrics
        metrics = data.get("metrics", {})
        if metrics:
            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"color: {C['border']};")
            lay.addWidget(sep)

            grid = QGridLayout()
            grid.setSpacing(4)
            items = [
                ("Avg velocity",   f"{metrics.get('avg_velocity_px_per_frame', 0):.1f} px/fr"),
                ("Collisions",     str(metrics.get("collision_count", 0))),
                ("Shadowing",      str(metrics.get("shadowing_events", 0))),
                ("Under pressure", f"{metrics.get('scoring_under_pressure_rate', 0):.0%}"),
            ]
            for r, (lbl_text, val_text) in enumerate(items):
                lbl = QLabel(lbl_text)
                lbl.setObjectName("muted")
                val = QLabel(val_text)
                val.setStyleSheet(
                    f"color: {C['text']}; font-family: 'Fira Code', monospace;"
                )
                grid.addWidget(lbl, r, 0)
                grid.addWidget(val, r, 1)
            lay.addLayout(grid)


class DrivingPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        hdr = QHBoxLayout()
        hdr.addWidget(self._lbl("Driving Analysis", "h1"))
        hdr.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("sec")
        refresh_btn.setFixedWidth(90)
        refresh_btn.clicked.connect(self.reload)
        hdr.addWidget(refresh_btn)
        root.addLayout(hdr)

        root.addWidget(self._lbl(
            "Driving style classification based on robot movement patterns.", "muted"))

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: cards scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea {{ border: none; background: transparent; }}")

        self._cards_widget = QWidget()
        self._cards_widget.setStyleSheet("background: transparent;")
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setSpacing(12)
        self._cards_layout.addStretch()
        scroll.setWidget(self._cards_widget)
        splitter.addWidget(scroll)

        # Right: style chart for selected robot
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(self._lbl("STYLE SCORES", "kpi_lbl"))
        self._chart = ChartCanvas(height=4)
        right_lay.addWidget(self._chart)
        right_lay.addStretch()
        splitter.addWidget(right)

        splitter.setSizes([480, 340])
        root.addWidget(splitter)

        self._cards: list[_RobotCard] = []
        self.reload()

    def reload(self):
        data = _load_driving()

        # Clear
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        if not data:
            ph = QLabel("No driving data — run the pipeline first.")
            ph.setObjectName("muted")
            self._cards_layout.insertWidget(0, ph)
            self._cards.append(ph)
            return

        teams = list(data.keys())
        for i, team in enumerate(teams):
            card = _RobotCard(team, data[team],
                              TEAM_COLORS[i % len(TEAM_COLORS)])
            self._cards_layout.insertWidget(i, card)
            self._cards.append(card)

        # Draw chart for first robot
        if teams:
            first = data[teams[0]]
            style_scores = first.get("style_scores", {})
            if style_scores:
                style_radar_chart(self._chart, style_scores, teams[0])

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
