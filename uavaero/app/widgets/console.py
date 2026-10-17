"""The console: what the solvers are doing, and what they said when they stopped.

Two problems this fixes, both of which made the app feel broken when it was merely slow.

**You could not tell it was running.** A VSPAERO solve is tens of seconds of silence behind a
window that does not repaint. The output was already being tailed line by line — `run_isolated`
watches the solver's log file precisely so that a long wait is visible — and it was being thrown
into `statusBar().showMessage(line[:160])`, where each line replaced the last and all of it was
gone a moment later.

**You could not see why a solve failed.** The worker captures forty lines of solver output before
it dies; the error dialog showed three of them, truncated. The interesting part of a VSPAERO
failure is usually further up than that.

So: everything goes here, timestamped, kept, colour-coded by level, and the run directory — where
`job.json`, `vspaero.log` and the `.vsp3` actually live — is one click away.

**Auto-scroll pauses when you scroll up.** Without that, reading anything during a live run is
impossible: the view yanks itself back to the bottom on every line. Scroll back to the bottom and
it resumes.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import PALETTE, mono_font, ui_font

#: How many lines to keep. A thick-panel sweep at HIGH fidelity emits a few thousand; keeping ten
#: thousand means a whole session of solves stays readable without the widget becoming a memory
#: leak that only shows up after an afternoon.
MAX_LINES = 10_000

INFO = "info"
DETAIL = "detail"
GOOD = "good"
WARN = "warn"
ERROR = "error"

_COLOURS = {
    INFO: "ink",
    DETAIL: "ink_3",
    GOOD: "good",
    WARN: "warn",
    ERROR: "warn",
}


class Console(QWidget):
    """A bounded, timestamped log of everything the solvers say."""

    #: Emitted when the user asks to open the directory a run left its artefacts in.
    open_folder_requested = Signal(object)

    #: Internal: carries one line to the GUI thread. `append` emits this rather than touching the
    #: widget, so a caller on a solve thread is marshalled instead of corrupting the document.
    _line_ready = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._run_directory: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_bar())

        # Connected to a bound method, which is what gives Qt a receiver whose thread it can
        # compare — the connection is then queued from a worker and direct from the GUI thread.
        # A lambda here would defeat the whole point; see CLAUDE.md rule 6.
        self._line_ready.connect(self._append_on_gui_thread)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setFont(mono_font(8))
        self.view.setMaximumBlockCount(MAX_LINES)
        self.view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.view.setStyleSheet(
            f"QPlainTextEdit {{ background: {PALETTE['inset']}; color: {PALETTE['ink_2']};"
            f" border: none; padding: 6px 10px; }}"
        )
        root.addWidget(self.view, 1)

    def _build_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("ChromeBar")
        bar.setFixedHeight(32)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(12)

        title = QLabel("CONSOLE")
        font = ui_font(8)
        font.setBold(True)
        title.setFont(font)
        title.setStyleSheet(f"color: {PALETTE['ink_3']};")
        layout.addWidget(title)

        self._status = QLabel("idle")
        self._status.setFont(mono_font(8))
        self._status.setStyleSheet(f"color: {PALETTE['ink_3']};")
        layout.addWidget(self._status)
        layout.addStretch(1)

        for text, slot in (
            ("Open run folder", self._open_folder),
            ("Copy", self._copy),
            ("Clear", self.clear),
        ):
            button = QPushButton(text)
            button.setProperty("chrome", "true")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return bar

    # ── writing ────────────────────────────────────────────────────────────────────────

    def append(self, text: str, level: str = INFO) -> None:
        """One line, timestamped and coloured. **Safe to call from any thread.**

        The solve runs on a QThread and its output arrives from there, so this cannot touch the
        widget directly: Qt widgets are not thread-safe, and a `QPlainTextEdit` written from a
        worker while the GUI thread lays it out corrupts the document — the window stops repainting
        and the process dies natively with no traceback. That is not hypothetical, it is what this
        console did on its first real sweep. So every line goes through a signal, and the widget is
        only ever touched by `_append_on_gui_thread`.
        """
        if not text:
            return
        self._line_ready.emit(text, level)

    def _append_on_gui_thread(self, text: str, level: str) -> None:
        """The only place the view is written. GUI thread only — see `append`.

        Auto-scroll only if the view was already at the bottom. Otherwise reading back through a
        live run is impossible — every arriving line drags the viewport away from what you are
        looking at.
        """
        bar = self.view.verticalScrollBar()
        was_at_bottom = bar.value() >= bar.maximum() - 4

        colour = PALETTE.get(_COLOURS.get(level, "ink"), PALETTE["ink"])
        stamp = datetime.now().strftime("%H:%M:%S")
        escaped = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        self.view.appendHtml(
            f'<span style="color:{PALETTE["ink_3"]}">{stamp}</span> '
            f'<span style="color:{colour}">{escaped}</span>'
        )
        if was_at_bottom:
            self.view.moveCursor(QTextCursor.MoveOperation.End)

    def rule(self, title: str) -> None:
        """A separator, so one run is visually distinct from the next."""
        self.append("─" * 4 + f" {title} " + "─" * max(4, 72 - len(title)), DETAIL)

    def block(self, text: str, level: str = ERROR, limit: int = 60) -> None:
        """A captured multi-line blob — a traceback, or the tail of a solver log.

        Trimmed from the *front* when it is too long. The end of a traceback names the failure; the
        start names whichever frame happened to be outermost.
        """
        lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
        if not lines:
            return
        if len(lines) > limit:
            self.append(f"  … {len(lines) - limit} earlier line(s) omitted", DETAIL)
            lines = lines[-limit:]
        for line in lines:
            self.append(f"  {line}", level)

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_run_directory(self, directory: Path | None) -> None:
        """Where this run left its artefacts. The `job.json`, the solver log and the `.vsp3` are
        the actual evidence when something goes wrong; the console is a summary of them."""
        self._run_directory = Path(directory) if directory else None

    def clear(self) -> None:
        self.view.clear()

    # ── actions ────────────────────────────────────────────────────────────────────────

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.view.toPlainText())
        self.append("console copied to the clipboard", DETAIL)

    def _open_folder(self) -> None:
        if self._run_directory is None or not self._run_directory.exists():
            self.append("no run directory yet — solve something first", WARN)
            return
        self.open_folder_requested.emit(self._run_directory)


class ConsoleHandler(logging.Handler):
    """Routes the standard logging module into the console.

    So that pure-Python code under `uavaero/` can report without importing Qt: it calls
    `logger.warning(...)` like any library, and the GUI decides where that goes.

    Records arrive on whichever thread emitted them, which is why `Console.append` has to be
    thread-safe rather than this handler being careful. An earlier version of this docstring
    claimed the marshalling happened here; it did not happen anywhere, and the identical defect on
    the solver's output signal is what froze the app.
    """

    LEVELS = {
        logging.DEBUG: DETAIL,
        logging.INFO: INFO,
        logging.WARNING: WARN,
        logging.ERROR: ERROR,
        logging.CRITICAL: ERROR,
    }

    def __init__(self, emit_line) -> None:
        super().__init__()
        self._emit_line = emit_line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._emit_line(record.getMessage(), self.LEVELS.get(record.levelno, INFO))
        except Exception:  # noqa: BLE001 - a logging handler must never raise into its caller
            pass
