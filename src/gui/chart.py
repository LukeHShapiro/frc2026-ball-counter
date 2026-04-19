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


def match_history_chart(
    canvas: ChartCanvas,
    history: list[dict],
    selected_team: str | None = None,
) -> None:
    """
    Line chart showing score per analyzed match for one or all teams.

    history = [{match_label, teams: {team_number: {score, ...}}}]
    If selected_team is None or "All Teams", plot all teams.
    """
    canvas._fig.clf()
    ax = canvas.add_subplot(111)

    if not history:
        ax.text(0.5, 0.5, "No match history yet.\nRun the pipeline on multiple matches.",
                transform=ax.transAxes, ha="center", va="center", color=C["text_muted"],
                fontsize=10)
        canvas.refresh()
        return

    labels = [m.get("match_label", f"Match {i+1}") for i, m in enumerate(history)]
    x = list(range(len(labels)))

    # Collect all teams across history
    all_teams: list[str] = []
    seen: set[str] = set()
    for m in history:
        for t in m.get("teams", {}):
            if t not in seen and t not in ("UNATTRIBUTED", "REPLACED"):
                all_teams.append(t)
                seen.add(t)

    teams_to_plot = [selected_team] if (selected_team and selected_team != "All Teams") else all_teams

    for i, team in enumerate(teams_to_plot):
        ys = [m.get("teams", {}).get(team, {}).get("score", 0) for m in history]
        color = TEAM_COLORS[i % len(TEAM_COLORS)]
        ax.plot(x, ys, marker="o", linewidth=2, color=color,
                label=f"Team {team}", markersize=5)
        for xi, yi in zip(x, ys):
            if yi:
                ax.annotate(str(yi), (xi, yi), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8,
                            color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score", color=C["text_muted"])
    ax.set_title("Score per Match", color=C["text"], fontsize=12, pad=10)
    if len(teams_to_plot) > 1:
        ax.legend(loc="upper left", framealpha=0.7, fontsize=8)
    canvas.refresh()


def robot_heatmap_chart(
    canvas: ChartCanvas,
    positions: dict,
    selected_team: str | None = None,
    field_w: int = 1920,
    field_h: int = 1080,
) -> None:
    """
    2D density heat map of robot centroid positions over the match.

    Single team → gaussian heat map in that team's colour.
    All teams   → light scatter plot with one colour per team.

    positions = {team_number: [[cx, cy, frame_id], ...]}
    """
    import numpy as np

    canvas._fig.clf()
    ax = canvas.add_subplot(111)

    if not positions:
        ax.text(0.5, 0.5, "No position data.\nRun the pipeline first.",
                transform=ax.transAxes, ha="center", va="center",
                color=C["text_muted"], fontsize=10)
        canvas.refresh()
        return

    teams = list(positions.keys())
    teams_to_plot = (
        [selected_team]
        if (selected_team and selected_team != "All Teams" and selected_team in positions)
        else teams
    )

    ax.set_xlim(0, field_w)
    ax.set_ylim(field_h, 0)   # image-coord y (0 at top)
    ax.set_aspect("auto")

    # Field outline + centre line
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), field_w, field_h,
                            fill=False, edgecolor=C["border"], linewidth=1.5))
    ax.axvline(field_w / 2, color=C["border"], linewidth=1, alpha=0.5, linestyle="--")

    # Alliance labels
    ax.text(field_w * 0.15, field_h * 0.97, "RED SIDE",
            ha="center", color="#ef4444", fontsize=7, alpha=0.55)
    ax.text(field_w * 0.85, field_h * 0.97, "BLUE SIDE",
            ha="center", color="#3b82f6", fontsize=7, alpha=0.55)

    if len(teams_to_plot) == 1:
        team = teams_to_plot[0]
        pts  = positions[team]
        xs   = np.array([p[0] for p in pts], dtype=float)
        ys   = np.array([p[1] for p in pts], dtype=float)

        h2d, xedges, yedges = np.histogram2d(
            xs, ys, bins=[80, 45],
            range=[[0, field_w], [0, field_h]])

        try:
            from scipy.ndimage import gaussian_filter
            h2d = gaussian_filter(h2d.T, sigma=2.5)
        except ImportError:
            h2d = h2d.T

        import matplotlib.colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap
        color = TEAM_COLORS[teams.index(team) % len(TEAM_COLORS)]
        try:
            rgb  = mcolors.to_rgb(color)
            cmap = LinearSegmentedColormap.from_list(
                "t", [(0, 0, 0, 0), (*rgb, 1.0)], N=256)
        except Exception:
            cmap = "hot"

        ax.imshow(h2d, origin="upper",
                  extent=[0, field_w, field_h, 0],
                  cmap=cmap, alpha=0.85, aspect="auto")
        ax.set_title(f"Team {team} — Movement Density",
                     color=C["text"], fontsize=11, pad=8)
    else:
        for i, team in enumerate(teams_to_plot):
            pts = positions.get(team, [])
            if not pts:
                continue
            step = max(1, len(pts) // 500)
            xs = [p[0] for p in pts[::step]]
            ys = [p[1] for p in pts[::step]]
            ax.scatter(xs, ys, c=TEAM_COLORS[i % len(TEAM_COLORS)],
                       s=3, alpha=0.22, label=f"Team {team}")
        ax.legend(loc="upper right", framealpha=0.6, fontsize=8, markerscale=4)
        ax.set_title("All Robots — Field Coverage",
                     color=C["text"], fontsize=11, pad=8)

    ax.set_xlabel("Field X (px)", color=C["text_muted"], fontsize=8)
    ax.set_ylabel("Field Y (px)", color=C["text_muted"], fontsize=8)
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
