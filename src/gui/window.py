"""
src/gui/window.py — Main application window with sidebar navigation
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QByteArray, QSize
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton,
    QSizePolicy, QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

# OmniScouter mark — extracted from brand identity SVG (compact icon variant)
from .theme import C, QSS

# OmniScouter mark — compact icon variant from brand identity SVG
_LOGO_SVG = b"""<svg viewBox="0 0 240 240" width="42" height="42" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <filter id="og" x="-30%" y="-30%" width="160%" height="160%">
      <feGaussianBlur stdDeviation="2.5" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <filter id="tg" x="-40%" y="-40%" width="180%" height="180%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="240" height="240" rx="54" fill="#0f1410"/>
  <polygon points="120,34 188,74 188,166 120,206 52,166 52,74"
    fill="none" stroke="#f97316" stroke-width="3" filter="url(#og)"/>
  <circle cx="120" cy="34"  r="5" fill="#f97316" filter="url(#og)"/>
  <circle cx="188" cy="74"  r="5" fill="#f97316" filter="url(#og)"/>
  <circle cx="188" cy="166" r="5" fill="#f97316" filter="url(#og)"/>
  <circle cx="120" cy="206" r="5" fill="#f97316" filter="url(#og)"/>
  <circle cx="52"  cy="166" r="5" fill="#f97316" filter="url(#og)"/>
  <circle cx="52"  cy="74"  r="5" fill="#f97316" filter="url(#og)"/>
  <ellipse cx="120" cy="120" rx="42" ry="26"
    fill="none" stroke="#2dd4bf" stroke-width="2.5" filter="url(#tg)"/>
  <circle cx="120" cy="120" r="18"
    fill="none" stroke="#2dd4bf" stroke-width="2" stroke-opacity="0.5"/>
  <circle cx="120" cy="120" r="7" fill="#2dd4bf" filter="url(#tg)"/>
  <circle cx="120" cy="120" r="3" fill="#ffffff" opacity="0.9"/>
</svg>"""
from .pages.analyze  import AnalyzePage
from .pages.scores   import ScoresPage
from .pages.driving  import DrivingPage
from .pages.alliance import AlliancePage
from .pages.review   import ReviewPage
from .pages.settings import SettingsPage


_NAV_ITEMS = [
    ("Analyze",  "⬡"),
    ("Scores",   "◈"),
    ("Driving",  "◉"),
    ("Alliance", "⬟"),
    ("Review",   "⚑"),
    ("Settings", "⚙"),
]


class _NavButton(QPushButton):
    def __init__(self, icon_char: str, label: str, parent=None):
        super().__init__(parent)
        self._icon_char = icon_char
        self._label     = label
        self.setCheckable(True)
        self.setFlat(True)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._refresh_style(False)

    def _refresh_style(self, active: bool):
        accent   = C["accent"]
        text     = C["text"]
        muted    = C["text_muted"]
        bg_hover = C["bg_elevated"]

        if active:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: {accent}18;
                    border-left: 3px solid {accent};
                    border-right: none;
                    border-top: none;
                    border-bottom: none;
                    border-radius: 0;
                    color: {accent};
                    font-size: 12px;
                    font-weight: 700;
                    text-align: left;
                    padding-left: 20px;
                    letter-spacing: 0.5px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QPushButton {{
                    background: transparent;
                    border-left: 3px solid transparent;
                    border-right: none;
                    border-top: none;
                    border-bottom: none;
                    border-radius: 0;
                    color: {muted};
                    font-size: 12px;
                    font-weight: 400;
                    text-align: left;
                    padding-left: 20px;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover {{
                    background: {bg_hover};
                    color: {text};
                }}
            """)

    def set_active(self, active: bool):
        self._refresh_style(active)

    # Override text to include icon prefix
    def setText(self, text: str):
        super().setText(f"  {self._icon_char}  {text}")


class _Sidebar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(192)
        self.setObjectName("sidebar")
        self.setStyleSheet(f"""
            #sidebar {{
                background-color: {C['bg_panel']};
                border-right: 1px solid {C['border']};
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Logo / app title
        logo_frame = QFrame()
        logo_frame.setFixedHeight(68)
        logo_frame.setStyleSheet(
            f"background: {C['bg_panel']}; "
            f"border-bottom: 1px solid {C['border']};"
        )
        logo_lay = QHBoxLayout(logo_frame)
        logo_lay.setContentsMargins(14, 0, 14, 0)
        logo_lay.setSpacing(10)

        # SVG mark
        svg_mark = QSvgWidget()
        svg_mark.load(QByteArray(_LOGO_SVG))
        svg_mark.setFixedSize(42, 42)
        svg_mark.setStyleSheet("background: transparent;")
        logo_lay.addWidget(svg_mark, 0, Qt.AlignmentFlag.AlignVCenter)

        # Wordmark stack
        text_widget = QWidget()
        text_widget.setStyleSheet("background: transparent;")
        text_lay = QVBoxLayout(text_widget)
        text_lay.setContentsMargins(0, 0, 0, 0)
        text_lay.setSpacing(1)

        title = QLabel("Omni<span style='color:#2dd4bf;'>Scouter</span>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setStyleSheet(
            f"color: {C['accent']}; font-size: 13px; font-weight: 700; "
            f"font-family: Georgia, serif; letter-spacing: 1px; background: transparent;"
        )
        sub = QLabel("FRC Intelligence")
        sub.setStyleSheet(
            f"color: {C['text_dim']}; font-size: 9px; letter-spacing: 0.8px; "
            f"font-family: 'Courier New', monospace; background: transparent;"
        )
        text_lay.addWidget(title)
        text_lay.addWidget(sub)
        logo_lay.addWidget(text_widget, 1, Qt.AlignmentFlag.AlignVCenter)

        root.addWidget(logo_frame)

        root.addSpacing(8)

        # Nav section label
        nav_lbl = QLabel("NAVIGATION")
        nav_lbl.setStyleSheet(
            f"color: {C['text_dim']}; font-size: 9px; letter-spacing: 1.5px; "
            f"padding: 12px 20px 4px 20px;"
        )
        root.addWidget(nav_lbl)

        # Nav buttons
        self._buttons: list[_NavButton] = []
        for label, icon in _NAV_ITEMS:
            btn = _NavButton(icon, label)
            btn.setText(label)
            root.addWidget(btn)
            self._buttons.append(btn)

        root.addStretch()

        # Version footer
        ver = QLabel("v0.1.0-alpha")
        ver.setStyleSheet(
            f"color: {C['text_dim']}; font-size: 9px; "
            f"padding: 12px 20px; font-family: 'Fira Code', monospace;"
        )
        root.addWidget(ver)

    @property
    def buttons(self) -> list[_NavButton]:
        return self._buttons


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniScouter")
        self.setMinimumSize(1100, 720)
        self.resize(1280, 800)
        self._build_ui()

    def _build_ui(self):
        # Apply global stylesheet
        self.setStyleSheet(QSS)

        central = QWidget()
        self.setCentralWidget(central)
        main_lay = QHBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)
        main_lay.setSpacing(0)

        # Sidebar
        self._sidebar = _Sidebar()
        main_lay.addWidget(self._sidebar)

        # Page stack
        self._stack = QStackedWidget()
        self._stack.setStyleSheet(
            f"background-color: {C['bg_base']};"
        )
        main_lay.addWidget(self._stack)

        # Pages
        self._pages = [
            AnalyzePage(),
            ScoresPage(),
            DrivingPage(),
            AlliancePage(),
            ReviewPage(),
            SettingsPage(),
        ]
        for page in self._pages:
            self._stack.addWidget(page)

        # Wire nav buttons
        for i, btn in enumerate(self._sidebar.buttons):
            btn.clicked.connect(lambda checked, idx=i: self._nav_to(idx))

        # Status bar
        sb = QStatusBar()
        sb.setStyleSheet(
            f"background: {C['bg_panel']}; color: {C['text_muted']}; "
            f"border-top: 1px solid {C['border']}; font-size: 11px;"
        )
        sb.showMessage("Ready  •  No video loaded")
        self.setStatusBar(sb)
        self._status_bar = sb

        # Reload data pages when pipeline finishes
        analyze_page: AnalyzePage = self._pages[0]  # type: ignore[assignment]
        scores_page:  ScoresPage  = self._pages[1]  # type: ignore[assignment]
        driving_page: DrivingPage = self._pages[2]  # type: ignore[assignment]
        review_page:  ReviewPage  = self._pages[4]  # type: ignore[assignment]
        analyze_page.pipeline_finished.connect(lambda _ok: scores_page.reload())
        analyze_page.pipeline_finished.connect(lambda _ok: driving_page.reload())
        analyze_page.pipeline_finished.connect(lambda _ok: review_page.reload())

        # Activate first page
        self._nav_to(0)

    def _nav_to(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._sidebar.buttons):
            btn.set_active(i == index)
        labels = [label for label, _ in _NAV_ITEMS]
        self._status_bar.showMessage(f"  {labels[index]}")
