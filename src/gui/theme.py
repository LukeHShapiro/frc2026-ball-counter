"""
src/gui/theme.py — Design tokens and QSS stylesheet

Design system: Data-Dense Dashboard + Dark Mode (OLED)
Typography:    Fira Code (data/mono) + Segoe UI (body)
Accent:        Indigo #6366f1
"""

# ── Color tokens ──────────────────────────────────────────────────────────────

C = {
    "bg_deep":     "#0a0a0c",
    "bg_base":     "#111113",
    "bg_panel":    "#16161a",   # sidebar / panel surfaces
    "bg_elevated": "#1a1a1d",
    "border":      "#1f1f23",
    "border_hi":   "#2a2a30",
    "accent":      "#6366f1",
    "accent_hi":   "#818cf8",
    "accent_dim":  "#1e1b4b",
    "text":        "#f1f5f9",
    "text_muted":  "#94a3b8",
    "text_dim":    "#64748b",
    "success":     "#22c55e",
    "warning":     "#f59e0b",
    "danger":      "#ef4444",
    "red":         "#dc2626",
    "blue":        "#2563eb",
    "purple":      "#a855f7",
}

# Per-robot chart colours (6 slots)
TEAM_COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6"]

# Driving style badge colours
STYLE_COLOR = {
    "SMOOTH":        "#22c55e",
    "RECKLESS":      "#ef4444",
    "DEFENSIVE":     "#f59e0b",
    "DEFENCE_PROOF": "#6366f1",
}

# Flag badge colours
FLAG_COLOR = {
    "AMBIGUOUS-MANUAL-REVIEW":    "#f59e0b",
    "INFERRED-LOW-CONF":          "#ef4444",
    "OPR-WEIGHTED":               "#a855f7",
    "TEAM-NUMBER-UNCONFIRMED":    "#f97316",
    "exit_count_confirmed":       "#22c55e",
    "SUPERSEDED-BY-OPR-WEIGHTED": "#64748b",
}

# ── Matplotlib style dict (apply with rcParams.update) ────────────────────────

MPLSTYLE = {
    "figure.facecolor": C["bg_deep"],
    "axes.facecolor":   C["bg_base"],
    "axes.edgecolor":   C["border"],
    "axes.labelcolor":  C["text_muted"],
    "axes.grid":        True,
    "grid.color":       C["border"],
    "grid.alpha":       0.6,
    "grid.linewidth":   0.5,
    "text.color":       C["text"],
    "xtick.color":      C["text_muted"],
    "ytick.color":      C["text_muted"],
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.facecolor": C["bg_elevated"],
    "legend.edgecolor": C["border"],
    "legend.fontsize":  10,
    "font.family":      "monospace",
    "lines.linewidth":  2.2,
    "patch.linewidth":  0,
}

# ── QSS stylesheet ────────────────────────────────────────────────────────────

QSS = f"""
/* Base */
QMainWindow, QWidget {{
    background-color: {C['bg_deep']};
    color: {C['text']};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
    border: none;
    outline: none;
}}

/* Sidebar */
#sidebar {{
    background-color: {C['bg_base']};
    border-right: 1px solid {C['border']};
    min-width: 192px;
    max-width: 192px;
}}
#app_title {{
    color: {C['text']};
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.3px;
    padding: 24px 20px 8px 20px;
}}
#app_sub {{
    color: {C['text_dim']};
    font-size: 11px;
    padding: 0 20px 20px 20px;
}}
#nav_section {{
    color: {C['text_dim']};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 16px 20px 6px 20px;
    text-transform: uppercase;
}}
QPushButton#nav_btn {{
    background: transparent;
    color: {C['text_muted']};
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0;
    text-align: left;
    padding: 9px 20px;
    font-size: 13px;
    font-weight: 400;
}}
QPushButton#nav_btn:hover {{
    background-color: {C['bg_elevated']};
    color: {C['text']};
}}
QPushButton#nav_btn[active=true] {{
    background-color: {C['accent_dim']};
    color: {C['accent_hi']};
    border-left: 2px solid {C['accent']};
    font-weight: 600;
}}

/* Cards */
QFrame#card {{
    background-color: {C['bg_base']};
    border: 1px solid {C['border']};
    border-radius: 8px;
}}
QFrame#card_hi {{
    background-color: {C['bg_elevated']};
    border: 1px solid {C['border_hi']};
    border-radius: 8px;
}}

/* Primary button */
QPushButton {{
    background-color: {C['accent']};
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    min-height: 34px;
}}
QPushButton:hover  {{ background-color: {C['accent_hi']}; }}
QPushButton:pressed {{ background-color: {C['accent_dim']}; }}
QPushButton:disabled {{
    background-color: {C['bg_elevated']};
    color: {C['text_dim']};
}}

/* Secondary button */
QPushButton#sec {{
    background-color: {C['bg_elevated']};
    color: {C['text']};
    border: 1px solid {C['border']};
}}
QPushButton#sec:hover {{
    background-color: {C['bg_base']};
    border-color: {C['accent']};
    color: {C['accent_hi']};
}}

/* Danger button */
QPushButton#danger {{
    background-color: transparent;
    color: {C['danger']};
    border: 1px solid {C['danger']};
}}
QPushButton#danger:hover {{ background-color: rgba(239,68,68,0.12); }}

/* Inputs */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {C['bg_elevated']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
    selection-background-color: {C['accent_dim']};
    min-height: 32px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {C['accent']};
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background-color: {C['bg_elevated']};
    color: {C['text']};
    border: 1px solid {C['border']};
    selection-background-color: {C['accent_dim']};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background: {C['bg_elevated']};
    border: none;
    width: 18px;
}}

/* Progress bar */
QProgressBar {{
    background-color: {C['bg_elevated']};
    border: none;
    border-radius: 3px;
    max-height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {C['accent']};
    border-radius: 3px;
}}

/* Tables */
QTableWidget {{
    background-color: {C['bg_base']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    gridline-color: {C['border']};
    selection-background-color: {C['accent_dim']};
    outline: none;
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 7px 12px;
    border-bottom: 1px solid {C['border']};
}}
QTableWidget::item:selected {{
    background-color: {C['accent_dim']};
    color: {C['accent_hi']};
}}
QHeaderView::section {{
    background-color: {C['bg_elevated']};
    color: {C['text_muted']};
    border: none;
    border-bottom: 1px solid {C['border']};
    border-right: 1px solid {C['border']};
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 600;
}}
QHeaderView {{ background-color: transparent; }}

/* Text areas */
QTextEdit, QPlainTextEdit {{
    background-color: {C['bg_elevated']};
    color: {C['text_muted']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    font-family: "Fira Code", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
    selection-background-color: {C['accent_dim']};
}}

/* Scrollbars */
QScrollBar:vertical {{
    background: transparent;
    width: 5px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C['border_hi']};
    border-radius: 2px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {C['text_dim']}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 5px;
}}
QScrollBar::handle:horizontal {{
    background: {C['border_hi']};
    border-radius: 2px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* Labels */
QLabel#h1 {{ font-size: 20px; font-weight: 700; color: {C['text']}; }}
QLabel#h2 {{ font-size: 15px; font-weight: 700; color: {C['text']}; }}
QLabel#muted {{ color: {C['text_muted']}; font-size: 12px; }}
QLabel#kpi_num {{
    font-size: 30px;
    font-weight: 700;
    font-family: "Fira Code", "Consolas", monospace;
    color: {C['text']};
}}
QLabel#kpi_lbl {{
    font-size: 10px;
    font-weight: 600;
    color: {C['text_dim']};
    letter-spacing: 0.8px;
}}
QLabel#badge {{
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.3px;
}}
QLabel#status_dot {{
    border-radius: 4px;
    min-width: 8px;
    max-width: 8px;
    min-height: 8px;
    max-height: 8px;
}}

/* Separators */
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {C['border']}; }}

/* Status bar */
QStatusBar {{
    background-color: {C['bg_base']};
    color: {C['text_muted']};
    border-top: 1px solid {C['border']};
    font-size: 12px;
}}

/* Tooltips */
QToolTip {{
    background-color: {C['bg_elevated']};
    color: {C['text']};
    border: 1px solid {C['border']};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}

/* Splitter */
QSplitter::handle {{ background-color: {C['border']}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical  {{ height: 1px; }}
"""
