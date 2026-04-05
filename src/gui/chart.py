"""
src/gui/chart.py — Reusable embedded matplotlib chart widget

All charts share the dark theme from theme.MPLSTYLE.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QSizePolicy, QWidget, QVBoxLayout

from .theme import MPLSTYLE, TEAM_COLORS, C


def _apply_style(ax):
    """Apply dark theme to a single axes object."""
    ax.set_facecolor(C["bg_base"])
    ax.tick_params(colors=C["text_muted"], labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["border"])
    ax.grid(True, color=C["border"], alpha=0.5, linewidth=0.5)


class ChartCanvas(FigureCanvasQTAgg):
    """A matplotlib figure embedded as a Qt widget."""

    def __init__(self, width: int = 6, height: int = 3, dpi: int = 96):
        plt.rcParams.update(MPLSTYLE)
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        fig.patch.set_facecolor(C["bg_deep"])
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._fig = fig

    def clear(self):
        self._fig.clf()
        self.draw()

    def add_subplot(self, *args, **kwargs):
        ax = self._fig.add_subplot(*args, **kwargs)
        _apply_style(ax)
        return ax

    def refresh(self):
        self._fig.tight_layout(pad=1.5)
        self.draw()


# ── Pre-built chart helpers ────────────────────────────────────────────────────

def score_bar_chart(canvas: ChartCanvas, scores: dict[str, dict]) -> None:
    """
    Horizontal bar chart: team → total score.
    scores = {team_number: {score, high_conf, med_conf, low_conf}}
    """
    canvas._fig.clf()
    if not scores:
        ax = canvas.add_subplot(111)
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color=C["text_muted"])
        canvas.refresh()
        return

    teams = [t for t in scores if t not in ("UNATTRIBUTED", "REPLACED")]
    values = [scores[t]["score"] for t in teams]
    colors = [TEAM_COLORS[i % len(TEAM_COLORS)] for i in range(len(teams))]

    ax = canvas.add_subplot(111)
    bars = ax.barh(teams, values, color=colors, height=0.55)
    ax.set_xlabel("Total Points", color=C["text_muted"])
    ax.set_title("Score by Team", color=C["text"], fontsize=12, pad=10)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", color=C["text_muted"], fontsize=10)
    canvas.refresh()


def score_timeline_chart(canvas: ChartCanvas, timeline: list[dict],
                          source_fps: float = 30.0) -> None:
    """
    Line chart: per-robot cumulative score over match time.
    timeline = [{frame_id, team_number, confidence}]
    """
    canvas._fig.clf()
    if not timeline:
        ax = canvas.add_subplot(111)
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color=C["text_muted"])
        canvas.refresh()
        return

    # Build cumulative series per team
    teams = sorted({e["team_number"] for e in timeline
                    if e["team_number"] not in ("UNATTRIBUTED", "REPLACED")})
    team_events: dict[str, list[tuple[float, int]]] = {t: [] for t in teams}
    for ev in sorted(timeline, key=lambda e: e["frame_id"]):
        t = ev["team_number"]
        if t in team_events:
            team_events[t].append(ev["frame_id"] / source_fps)

    ax = canvas.add_subplot(111)
    for i, team in enumerate(teams):
        times = sorted(team_events[team])
        if not times:
            continue
        xs = [0.0] + times + [times[-1]]
        ys = list(range(len(xs)))
        ys[-1] = ys[-2]
        ax.step(xs, ys, where="post", color=TEAM_COLORS[i % len(TEAM_COLORS)],
                label=f"Team {team}", linewidth=2)

    ax.set_xlabel("Match Time (s)", color=C["text_muted"])
    ax.set_ylabel("Cumulative Score", color=C["text_muted"])
    ax.set_title("Score Timeline", color=C["text"], fontsize=12, pad=10)
    ax.legend(loc="upper left", framealpha=0.7)
    canvas.refresh()


def style_radar_chart(canvas: ChartCanvas, style_scores: dict[str, float],
                       team: str) -> None:
    """
    Simple horizontal bar chart for driving style scores (radar alternative).
    style_scores = {"SMOOTH": 0.7, "RECKLESS": 0.2, ...}
    """
    from .theme import STYLE_COLOR

    canvas._fig.clf()
    ax = canvas.add_subplot(111)

    styles = list(style_scores.keys())
    values = [style_scores[s] for s in styles]
    colors = [STYLE_COLOR.get(s, C["accent"]) for s in styles]

    bars = ax.barh(styles, values, color=colors, height=0.45)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score", color=C["text_muted"])
    ax.set_title(f"Team {team} — Style Scores", color=C["text"],
                 fontsize=11, pad=8)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.03, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", color=C["text_muted"], fontsize=9)
    canvas.refresh()
