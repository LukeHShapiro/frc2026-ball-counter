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

from PyQt6.QtWidgets import QComboBox, QTabWidget  # extra widgets for heatmap tab

from ..chart import ChartCanvas, style_radar_chart, robot_heatmap_chart
from ..theme import C, STYLE_COLOR, TEAM_COLORS

_ROOT = Path(__file__).parent.parent.parent.parent
_POSITIONS_PATH = _ROOT / "data" / "exports" / "robot_positions.json"


def _load_positions() -> dict:
    if _POSITIONS_PATH.exists():
        try:
            return json.loads(_POSITIONS_PATH.read_text())
        except Exception:
            pass
    return {}


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

        # Right: tabbed panel — Style Scores | Field Heatmap
        right_tabs = QTabWidget()
        right_tabs.setStyleSheet(
            f"QTabBar::tab {{ color: {C['text_muted']}; padding: 6px 14px; }}"
            f"QTabBar::tab:selected {{ color: {C['accent']}; border-bottom: 2px solid {C['accent']}; }}"
        )

        # Tab 1: Style radar chart
        style_tab = QWidget()
        style_lay = QVBoxLayout(style_tab)
        style_lay.setContentsMargins(4, 8, 4, 4)
        style_lay.addWidget(self._lbl("STYLE SCORES", "kpi_lbl"))
        self._chart = ChartCanvas(height=4)
        style_lay.addWidget(self._chart)
        style_lay.addStretch()
        right_tabs.addTab(style_tab, "Style Scores")

        # Tab 2: Field heatmap
        heat_tab = QWidget()
        heat_lay = QVBoxLayout(heat_tab)
        heat_lay.setContentsMargins(4, 8, 4, 4)

        heat_ctrl = QHBoxLayout()
        heat_ctrl.addWidget(self._lbl("Team:", "kpi_lbl"))
        self._heatmap_combo = QComboBox()
        self._heatmap_combo.setMinimumWidth(130)
        self._heatmap_combo.addItem("All Teams")
        self._heatmap_combo.currentTextChanged.connect(self._update_heatmap)
        heat_ctrl.addWidget(self._heatmap_combo)
        heat_ctrl.addStretch()
        heat_lay.addLayout(heat_ctrl)

        self._heatmap_canvas = ChartCanvas(height=4)
        heat_lay.addWidget(self._heatmap_canvas)
        right_tabs.addTab(heat_tab, "Field Heatmap")

        splitter.addWidget(right_tabs)
        splitter.setSizes([480, 380])
        root.addWidget(splitter)

        self._cards: list[_RobotCard] = []
        self.reload()

    def reload(self):
        data      = _load_driving()
        positions = _load_positions()

        # Clear robot cards
        for card in self._cards:
            self._cards_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        if not data:
            ph = QLabel("No driving data — run the pipeline first.")
            ph.setObjectName("muted")
            self._cards_layout.insertWidget(0, ph)
            self._cards.append(ph)
        else:
            teams = list(data.keys())
            for i, team in enumerate(teams):
                card = _RobotCard(team, data[team],
                                  TEAM_COLORS[i % len(TEAM_COLORS)])
                self._cards_layout.insertWidget(i, card)
                self._cards.append(card)

            # Style chart for first robot
            first = data[teams[0]]
            style_scores = first.get("style_scores", {})
            if style_scores:
                style_radar_chart(self._chart, style_scores, teams[0])

        # Refresh heatmap team selector
        self._positions = positions
        current = self._heatmap_combo.currentText()
        self._heatmap_combo.blockSignals(True)
        self._heatmap_combo.clear()
        self._heatmap_combo.addItem("All Teams")
        for t in sorted(positions.keys(), key=lambda x: int(x) if x.isdigit() else 99999):
            self._heatmap_combo.addItem(t)
        idx = self._heatmap_combo.findText(current)
        self._heatmap_combo.setCurrentIndex(max(0, idx))
        self._heatmap_combo.blockSignals(False)

        self._update_heatmap(self._heatmap_combo.currentText())

    def _update_heatmap(self, team: str):
        positions = getattr(self, "_positions", {})
        robot_heatmap_chart(self._heatmap_canvas, positions,
                            selected_team=team if team != "All Teams" else None)

    @staticmethod
    def _lbl(text: str, obj: str = "") -> QLabel:
        lbl = QLabel(text)
        if obj:
            lbl.setObjectName(obj)
        return lbl
