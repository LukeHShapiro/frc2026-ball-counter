"""
src/gui/worker.py — Pipeline QThread
"""

from __future__ import annotations

import io
import queue
import sys
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal


# ── stdout capture ─────────────────────────────────────────────────────────────

class _Capture(io.TextIOBase):
    """Redirect writes to a callable (e.g. self._emit_line)."""
    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def write(self, text: str) -> int:
        if text and text.strip():
            self._callback(text.rstrip())
        return len(text)

    def flush(self):
        pass


# ── stdin pipe (GUI → pipeline thread) ────────────────────────────────────────

class _StdinPipe(io.TextIOBase):
    """
    Fake stdin backed by a Queue.
    The GUI pushes lines via .send(text); the pipeline reads via input() / readline().
    """
    def __init__(self):
        super().__init__()
        self._q: queue.Queue[str] = queue.Queue()

    def send(self, text: str):
        """Called from the GUI thread to deliver a line."""
        self._q.put(text)

    def readline(self) -> str:
        """Blocks until the GUI sends a line. Returns text + newline."""
        return self._q.get() + "\n"

    def read(self, n=-1) -> str:
        return self.readline()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True


# ── worker ────────────────────────────────────────────────────────────────────

class PipelineWorker(QThread):
    """
    Signals
    -------
    log_line(str)          - one line of stdout from the pipeline
    phase_update(str, int) - current phase label + 0-100 progress pct
    needs_input(str)       - pipeline is waiting for user input (prompt text)
    finished(bool, str)    - success flag + summary message
    """
    log_line     = pyqtSignal(str)
    phase_update = pyqtSignal(str, int)
    needs_input  = pyqtSignal(str)   # emitted when pipeline calls input()
    frame_ready  = pyqtSignal(bytes) # JPEG bytes of latest annotated frame (#12)
    finished     = pyqtSignal(bool, str)

    _PHASE_MAP = {
        "PHASE 2":  10,
        "PHASE 5":  20,
        "PHASE 6":  35,
        "PHASE 7":  50,
        "PHASE 8":  60,
        "PHASE 9":  70,
        "PHASE 10": 80,
        "PHASE 11": 85,
        "PHASE 12": 88,
        "PHASE 13": 92,
        "PHASE 14": 96,
    }

    def __init__(
        self,
        video_path:   str,
        sample_every: int  = 5,
        skip_ingest:  bool = False,
        skip_detect:  bool = False,
    ):
        super().__init__()
        self.video_path   = video_path
        self.sample_every = sample_every
        self.skip_ingest  = skip_ingest
        self.skip_detect  = skip_detect
        self.stdin_pipe   = _StdinPipe()

    def send_input(self, text: str):
        """Called from the GUI to answer a pipeline prompt."""
        self.stdin_pipe.send(text)

    # ── intercept log lines to emit phase updates ────────────────────────────

    def _emit_line(self, line: str):
        self.log_line.emit(line)
        # Detect pipeline waiting for input
        if "Enter 'yes'" in line or line.strip().endswith(">"):
            self.needs_input.emit(line)
            return
        for key, pct in self._PHASE_MAP.items():
            if key in line:
                label = line.strip().lstrip("=- ").strip()
                self.phase_update.emit(label, pct)
                return
        for keyword, (label, pct) in {
            "[Detect]":      ("Detection",          25),
            "[Track]":       ("Tracking",            38),
            "[Possession]":  ("Possession engine",   45),
            "[Trajectory]":  ("Trajectory analysis", 55),
            "[Attribution]": ("Attribution",         72),
            "[Statbotics]":  ("Fetching OPR",        68),
            "[Calibrate]":   ("Robot calibration",   42),
            "[Alliance]":    ("Alliance detection",  44),
        }.items():
            if keyword in line:
                self.phase_update.emit(label, pct)
                return

    # ── main thread entry ────────────────────────────────────────────────────

    def run(self):
        project_root = str(Path(__file__).parent.parent.parent)
        src_root     = str(Path(__file__).parent.parent)
        for p in (project_root, src_root):
            if p not in sys.path:
                sys.path.insert(0, p)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdin  = sys.stdin
        capture    = _Capture(self._emit_line)
        sys.stdout = capture
        sys.stderr = capture
        sys.stdin  = self.stdin_pipe   # replace stdin with our pipe

        try:
            # Register real-time frame preview callback (#12)
            try:
                import detect as _detect
                _detect.set_frame_callback(self.frame_ready.emit)
            except Exception:
                pass

            from main import run_pipeline  # type: ignore

            run_pipeline(
                video_path   = self.video_path,
                sample_every = self.sample_every,
                skip_ingest  = self.skip_ingest,
                skip_detect  = self.skip_detect,
                no_ui        = True,
            )
            self.phase_update.emit("Complete", 100)
            self.finished.emit(True, "Pipeline finished successfully.")

        except Exception:
            tb = traceback.format_exc()
            for line in tb.splitlines():
                self.log_line.emit(line)
            self.finished.emit(False, tb.splitlines()[-1])

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin  = old_stdin
            # Clear frame callback so it doesn't hold a reference to this worker
            try:
                import detect as _detect
                _detect.clear_frame_callback()
            except Exception:
                pass
