"""
src/gui/pages/analyze.py — Video upload + pipeline runner page

Layout (two-column):
  Left  — video config, run controls, progress, robot assignment
  Right — live preview (while running) + pipeline log + input
"""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QElapsedTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QSizePolicy,
    QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
    QCheckBox, QLineEdit, QComboBox,
)

from ..theme import C
from ..worker import PipelineWorker
from ..robot_labeler import RobotLabelerDialog
from ..zone_calibrator import ZoneCalibratorDialog

_ROOT = Path(__file__).parent.parent.parent.parent


def _lbl(text: str, obj: str = "") -> QLabel:
    w = QLabel(text)
    if obj:
        w.setObjectName(obj)
    return w


def _card() -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)
    return f, lay


class AnalyzePage(QWidget):
    pipeline_finished = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: PipelineWorker | None = None
        self._assign_rows: list[dict] = []
        self._elapsed = QElapsedTimer()
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_timer)
        self._build_ui()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {C['border']}; }}"
        )
        root.addWidget(splitter)

        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([380, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left(self) -> QWidget:
        pane = QWidget()
        pane.setMinimumWidth(340)
        pane.setMaximumWidth(460)
        pane.setObjectName("left_pane")
        pane.setStyleSheet(
            f"#left_pane {{ background: {C['bg_panel']}; "
            f"border-right: 1px solid {C['border']}; }}"
        )

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(20, 24, 20, 24)
        lay.setSpacing(16)

        # Heading
        lay.addWidget(_lbl("Analyze Match", "h1"))
        lay.addWidget(_lbl("Configure and run the video analysis pipeline.", "muted"))

        # ── Video file / batch queue ───────────────────────────────────────────
        card, cl = _card()
        cl.addWidget(_lbl("VIDEO FILE", "kpi_lbl"))

        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("No file selected…")
        self._path_edit.setReadOnly(True)
        file_row.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("sec")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(browse_btn)
        cl.addLayout(file_row)

        # Batch queue label
        self._queue_lbl = _lbl("", "muted")
        self._queue_lbl.setStyleSheet(
            f"color: {C['accent']}; font-size: 10px; font-family: 'Fira Code', monospace;"
        )
        self._queue_lbl.hide()
        cl.addWidget(self._queue_lbl)

        # Batch mode controls
        batch_row = QHBoxLayout()
        self._batch_btn = QPushButton("+ Add to Batch")
        self._batch_btn.setObjectName("sec")
        self._batch_btn.setFixedHeight(28)
        self._batch_btn.setEnabled(False)
        self._batch_btn.setToolTip("Add more videos to process sequentially")
        self._batch_btn.clicked.connect(self._add_to_batch)
        batch_row.addWidget(self._batch_btn)

        self._clear_batch_btn = QPushButton("Clear Batch")
        self._clear_batch_btn.setObjectName("sec")
        self._clear_batch_btn.setFixedHeight(28)
        self._clear_batch_btn.setEnabled(False)
        self._clear_batch_btn.clicked.connect(self._clear_batch)
        batch_row.addWidget(self._clear_batch_btn)
        batch_row.addStretch()
        cl.addLayout(batch_row)

        self._batch_queue: list[str] = []   # extra videos (beyond the primary)

        # Options
        opts = QGridLayout()
        opts.setSpacing(8)
        opts.setColumnStretch(1, 1)

        opts.addWidget(_lbl("Sample every N frames:"), 0, 0)
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(1, 30)
        self._sample_spin.setValue(5)
        self._sample_spin.setFixedWidth(64)
        self._sample_spin.setToolTip(
            "5 = ~6 fps coverage, fast (recommended for bulk processing)\n"
            "3 = ~10 fps, more precise\n"
            "1 = every frame (slowest)"
        )
        opts.addWidget(self._sample_spin, 0, 1, Qt.AlignmentFlag.AlignLeft)

        self._skip_detect_cb = QCheckBox("Skip detection (use cached)")
        self._skip_detect_cb.setToolTip("Reuse data/detections.json from a previous run")
        opts.addWidget(self._skip_detect_cb, 1, 0, 1, 2)

        cl.addLayout(opts)
        lay.addWidget(card)

        # ── Run controls ──────────────────────────────────────────────────────
        card2, cl2 = _card()

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Pipeline")
        self._run_btn.setFixedHeight(36)
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run)
        btn_row.addWidget(self._run_btn, 2)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("danger")
        self._stop_btn.setFixedHeight(36)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self._stop_btn, 1)

        self._label_btn = QPushButton("Label Robots")
        self._label_btn.setObjectName("sec")
        self._label_btn.setFixedHeight(36)
        self._label_btn.setEnabled(False)
        self._label_btn.setToolTip("Open visual labeler — draw boxes and assign team numbers")
        self._label_btn.clicked.connect(self._open_labeler)
        btn_row.addWidget(self._label_btn, 1)

        self._zone_btn = QPushButton("Box Goals")
        self._zone_btn.setObjectName("sec")
        self._zone_btn.setFixedHeight(36)
        self._zone_btn.setEnabled(False)
        self._zone_btn.setToolTip("Draw red/blue goal zones on a video frame to set scoring area coordinates")
        self._zone_btn.clicked.connect(self._open_zone_calibrator)
        btn_row.addWidget(self._zone_btn, 1)
        cl2.addLayout(btn_row)

        # Status row
        status_row = QHBoxLayout()
        self._status_dot = QLabel()
        self._status_dot.setFixedSize(10, 10)
        self._set_dot("idle")
        status_row.addWidget(self._status_dot)
        self._status_lbl = _lbl("Idle", "muted")
        status_row.addWidget(self._status_lbl)
        status_row.addStretch()
        self._timer_lbl = QLabel("—")
        self._timer_lbl.setStyleSheet(
            f"color: {C['text_muted']}; font-family: 'Fira Code', monospace; font-size: 12px;"
        )
        self._timer_lbl.setToolTip("Elapsed pipeline time")
        status_row.addWidget(self._timer_lbl)
        cl2.addLayout(status_row)

        # Progress
        phase_row = QHBoxLayout()
        self._phase_lbl = _lbl("—", "muted")
        phase_row.addWidget(self._phase_lbl)
        phase_row.addStretch()
        self._pct_lbl = _lbl("0%", "muted")
        phase_row.addWidget(self._pct_lbl)
        cl2.addLayout(phase_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        cl2.addWidget(self._progress)

        lay.addWidget(card2)

        # ── Robot assignment ──────────────────────────────────────────────────
        assign_card, acl = _card()

        ahdr = QHBoxLayout()
        ahdr.addWidget(_lbl("ROBOT ASSIGNMENT", "kpi_lbl"))
        ahdr.addStretch()
        rel_btn = QPushButton("Reload")
        rel_btn.setObjectName("sec")
        rel_btn.setFixedWidth(64)
        rel_btn.clicked.connect(self._load_identity)
        ahdr.addWidget(rel_btn)
        sav_btn = QPushButton("Save")
        sav_btn.setFixedWidth(56)
        sav_btn.clicked.connect(self._save_identity)
        ahdr.addWidget(sav_btn)
        acl.addLayout(ahdr)

        acl.addWidget(_lbl(
            "After the pipeline runs, assign team numbers here and press Save.", "muted"))

        self._assign_grid = QGridLayout()
        self._assign_grid.setSpacing(6)
        for col, text in enumerate(("Track", "Alliance", "Team #", "Override")):
            h = QLabel(text)
            h.setObjectName("kpi_lbl")
            self._assign_grid.addWidget(h, 0, col)
        acl.addLayout(self._assign_grid)

        lay.addWidget(assign_card)
        lay.addStretch()

        outer = QVBoxLayout(pane)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(inner)
        return pane

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self) -> QWidget:
        pane = QWidget()
        pane.setStyleSheet(f"background: {C['bg_base']};")
        lay = QVBoxLayout(pane)
        lay.setContentsMargins(20, 24, 20, 16)
        lay.setSpacing(12)

        # Live preview (always visible)
        self._preview_card = QFrame()
        self._preview_card.setObjectName("card")
        prev_lay = QVBoxLayout(self._preview_card)
        prev_lay.setContentsMargins(12, 10, 12, 10)
        prev_lay.setSpacing(6)
        prev_lay.addWidget(_lbl("LIVE DETECTION PREVIEW", "kpi_lbl"))
        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setMinimumHeight(260)
        self._preview_lbl.setStyleSheet(
            f"background: {C['bg_deep']}; border-radius: 4px; color: {C['text_muted']};"
        )
        self._preview_lbl.setText("No pipeline running — frames will appear here during analysis.")
        prev_lay.addWidget(self._preview_lbl)
        lay.addWidget(self._preview_card)

        # Log header
        log_hdr = QHBoxLayout()
        log_hdr.addWidget(_lbl("PIPELINE LOG", "kpi_lbl"))
        log_hdr.addStretch()
        clr_btn = QPushButton("Clear")
        clr_btn.setObjectName("sec")
        clr_btn.setFixedWidth(56)
        clr_btn.clicked.connect(lambda: self._log.clear())
        log_hdr.addWidget(clr_btn)
        lay.addLayout(log_hdr)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._log.setStyleSheet(
            f"QTextEdit {{ background: {C['bg_deep']}; color: {C['text_muted']}; "
            f"font-family: 'Fira Code', 'Consolas', monospace; font-size: 11px; "
            f"border: 1px solid {C['border']}; border-radius: 6px; padding: 8px; }}"
        )
        lay.addWidget(self._log)

        # Input row
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText(
            "Pipeline input — type 'yes' or 'correct [track] to [team]' and press Send"
        )
        self._input_edit.setEnabled(False)
        self._input_edit.returnPressed.connect(self._send_input)
        input_row.addWidget(self._input_edit)

        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(64)
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_input)
        input_row.addWidget(self._send_btn)
        lay.addLayout(input_row)

        return pane

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_dot(self, state: str):
        colors = {
            "idle":    C["text_dim"],
            "running": C["warning"],
            "ok":      C["success"],
            "error":   C["danger"],
        }
        col = colors.get(state, C["text_dim"])
        self._status_dot.setStyleSheet(
            f"background-color: {col}; border-radius: 5px;"
        )

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select match video",
            str(Path.home()),
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
        )
        if path:
            self._path_edit.setText(path)
            self._run_btn.setEnabled(True)
            self._label_btn.setEnabled(True)
            self._zone_btn.setEnabled(True)
            self._batch_btn.setEnabled(True)

    # ── Batch queue ───────────────────────────────────────────────────────────

    def _add_to_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add videos to batch queue",
            str(Path.home()),
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
        )
        for p in paths:
            if p and p not in self._batch_queue and p != self._path_edit.text():
                self._batch_queue.append(p)
        self._update_batch_label()
        self._clear_batch_btn.setEnabled(bool(self._batch_queue))

    def _clear_batch(self):
        self._batch_queue.clear()
        self._update_batch_label()
        self._clear_batch_btn.setEnabled(False)

    def _update_batch_label(self):
        n = len(self._batch_queue)
        if n:
            self._queue_lbl.setText(f"+ {n} more video{'s' if n > 1 else ''} queued")
            self._queue_lbl.show()
        else:
            self._queue_lbl.hide()

    # ── Run ───────────────────────────────────────────────────────────────────

    def _run(self):
        path = self._path_edit.text().strip()
        has_video = bool(path and Path(path).exists())
        self._label_btn.setEnabled(has_video)
        self._zone_btn.setEnabled(has_video)
        self._batch_btn.setEnabled(has_video)
        if not path or not Path(path).exists():
            self._log_append("[ERROR] File not found.")
            return

        # Build full queue: primary + batch
        self._run_queue = [path] + [p for p in self._batch_queue if Path(p).exists()]
        self._run_queue_total = len(self._run_queue)

        self._log.clear()
        self._progress.setValue(0)
        self._set_dot("running")
        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._elapsed.start()
        self._tick.start()
        self._timer_lbl.setText("0:00")

        self._start_next_in_queue()

    def _start_next_in_queue(self):
        if not self._run_queue:
            return
        path = self._run_queue[0]
        done = self._run_queue_total - len(self._run_queue)
        total = self._run_queue_total
        prefix = f"[{done+1}/{total}] " if total > 1 else ""
        self._phase_lbl.setText(f"{prefix}Starting…")
        self._pct_lbl.setText("0%")
        self._status_lbl.setText(f"Running ({done+1}/{total})" if total > 1 else "Running")
        if total > 1:
            self._log_append(f"\n{'='*50}\n{prefix}{Path(path).name}\n{'='*50}")

        self._worker = PipelineWorker(
            video_path   = path,
            sample_every = self._sample_spin.value(),
            skip_ingest  = True,   # ingest disabled — process_video reads video directly
            skip_detect  = self._skip_detect_cb.isChecked(),
        )
        self._worker.log_line.connect(self._log_append)
        self._worker.phase_update.connect(self._on_phase)
        self._worker.needs_input.connect(self._on_needs_input)
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.finished.connect(self._on_finished)
        self._preview_lbl.setText("Waiting for frames…")
        self._worker.start()

    def _stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log_append("[Stopped by user]")
            self._on_finished(False, "Stopped by user.")

    def _log_append(self, text: str):
        self._log.append(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_phase(self, label: str, pct: int):
        self._phase_lbl.setText(label)
        self._pct_lbl.setText(f"{pct}%")
        self._progress.setValue(pct)

    def _open_labeler(self):
        path = self._path_edit.text().strip()
        if not path or not Path(path).exists():
            self._log_append("[ERROR] Select a valid video file first.")
            return
        dlg = RobotLabelerDialog(path, parent=self)
        if dlg.exec() == RobotLabelerDialog.DialogCode.Accepted:
            self._log_append("[Robot Labeler] Labels saved to configs/match_identity.json")
            self._load_identity()

    def _open_zone_calibrator(self):
        path = self._path_edit.text().strip()
        if not path or not Path(path).exists():
            self._log_append("[ERROR] Select a valid video file first.")
            return
        cfg_path = _ROOT / "configs" / "field_config.json"
        dlg = ZoneCalibratorDialog(path, cfg_path, parent=self)
        if dlg.exec() == ZoneCalibratorDialog.DialogCode.Accepted:
            self._log_append("[Zone Calibrator] Scoring zones saved to configs/field_config.json")

    def _on_frame_ready(self, jpeg_bytes: bytes):
        try:
            px = QPixmap()
            px.loadFromData(jpeg_bytes, "JPEG")
            if not px.isNull():
                scaled = px.scaledToWidth(
                    min(self._preview_lbl.width() or 760, 760),
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._preview_lbl.setPixmap(scaled)
        except Exception:
            pass

    def _on_needs_input(self, prompt: str):
        self._input_edit.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input_edit.setFocus()
        self._input_edit.setStyleSheet(
            f"border: 1px solid {C['accent']}; border-radius: 6px;"
        )
        self._log_append("  >> Pipeline waiting for input — type below and press Send")

    def _send_input(self):
        text = self._input_edit.text().strip()
        if not text or not self._worker:
            return
        self._log_append(f"  > {text}")
        self._worker.send_input(text)
        self._input_edit.clear()
        self._input_edit.setEnabled(False)
        self._send_btn.setEnabled(False)
        self._input_edit.setStyleSheet("")

    def _update_timer(self):
        ms = self._elapsed.elapsed()
        s  = ms // 1000
        self._timer_lbl.setText(f"{s // 60}:{s % 60:02d}")

    def _on_finished(self, success: bool, msg: str):
        self._input_edit.setEnabled(False)
        self._send_btn.setEnabled(False)
        self._input_edit.setStyleSheet("")
        self._log_append(f"\n{'✓' if success else '✗'} {msg}")
        self._load_identity()

        # Pop completed video from queue
        if hasattr(self, "_run_queue") and self._run_queue:
            self._run_queue.pop(0)

        remaining = len(getattr(self, "_run_queue", []))
        total     = getattr(self, "_run_queue_total", 1)

        if success and remaining > 0:
            # More videos to process — start next immediately
            next_path = self._run_queue[0]
            self._log_append(
                f"\n[Batch] {total - remaining}/{total} done — "
                f"starting {Path(next_path).name} …"
            )
            self._start_next_in_queue()
            return   # don't stop timer or re-enable run button yet

        # All done (or stopped/errored)
        self._tick.stop()
        self._update_timer()
        self._stop_btn.setEnabled(False)
        self._run_btn.setEnabled(True)

        if success:
            self._set_dot("ok")
            done_label = f"Complete ({total} videos)" if total > 1 else "Complete"
            self._status_lbl.setText(done_label)
            self._phase_lbl.setText("All done" if total > 1 else "Pipeline complete")
            self._preview_lbl.setText("Pipeline complete — run again to see new frames.")
            self._progress.setValue(100)
            self._pct_lbl.setText("100%")
        else:
            self._set_dot("error")
            self._status_lbl.setText("Error")
            self._preview_lbl.setText("Pipeline stopped.")

        self.pipeline_finished.emit(success)

    # ── Robot assignment panel ────────────────────────────────────────────────

    def _load_identity(self):
        path = _ROOT / "configs" / "match_identity.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except Exception:
            return

        robots = data.get("robots", [])

        for row_info in self._assign_rows:
            for w in row_info["widgets"]:
                w.setParent(None)
        self._assign_rows.clear()

        for i, robot in enumerate(robots, start=1):
            tid      = robot.get("track_id", i)
            alliance = robot.get("alliance", "unknown")
            team_num = robot.get("team_number", f"UNKNOWN_{tid}")
            conf     = robot.get("confidence", 0.0)

            tid_lbl = QLabel(str(tid))
            tid_lbl.setStyleSheet(
                f"color: {C['text_muted']}; font-size: 12px; "
                f"font-family: 'Fira Code', monospace;"
            )
            self._assign_grid.addWidget(tid_lbl, i, 0)

            col = C["red"] if alliance == "red" else (
                  C["blue"] if alliance == "blue" else C["text_dim"])
            al_lbl = QLabel(alliance.upper())
            al_lbl.setStyleSheet(
                f"color: white; background: {col}; border-radius: 3px; "
                f"padding: 1px 6px; font-size: 10px; font-weight: 700;"
            )
            al_lbl.setFixedWidth(56)
            self._assign_grid.addWidget(al_lbl, i, 1)

            team_edit = QLineEdit(team_num if not team_num.startswith("UNKNOWN") else "")
            team_edit.setPlaceholderText("e.g. 1234")
            team_edit.setFixedWidth(90)
            if conf >= 0.70:
                team_edit.setStyleSheet(f"color: {C['success']};")
            elif team_num.startswith("UNKNOWN"):
                team_edit.setStyleSheet(f"color: {C['danger']};")
            self._assign_grid.addWidget(team_edit, i, 2)

            override = QComboBox()
            override.addItems(["auto", "red", "blue"])
            override.setCurrentText(alliance if alliance in ("red", "blue") else "auto")
            override.setFixedWidth(72)
            self._assign_grid.addWidget(override, i, 3)

            self._assign_rows.append({
                "track_id":      tid,
                "team_edit":     team_edit,
                "override":      override,
                "widgets":       [tid_lbl, al_lbl, team_edit, override],
                "orig_alliance": alliance,
            })

    def _save_identity(self):
        path = _ROOT / "configs" / "match_identity.json"
        if not path.exists():
            self._log_append("[Robot Assignment] No identity file yet — run the pipeline first.")
            return
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            self._log_append(f"[Robot Assignment] Read error: {e}")
            return

        robots = {r["track_id"]: r for r in data.get("robots", [])}
        for row in self._assign_rows:
            tid     = row["track_id"]
            team    = row["team_edit"].text().strip() or f"UNKNOWN_{tid}"
            al_pick = row["override"].currentText()
            alliance = (row["orig_alliance"] if al_pick == "auto" else al_pick)
            if tid in robots:
                robots[tid]["team_number"]    = team
                robots[tid]["alliance"]       = alliance
                robots[tid]["user_corrected"] = True

        data["robots"]         = list(robots.values())
        data["user_confirmed"] = True
        path.write_text(json.dumps(data, indent=2))
        self._log_append(
            f"[Robot Assignment] Saved {len(self._assign_rows)} robots → configs/match_identity.json"
        )
