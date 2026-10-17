"""Dynamic modes: how the aircraft behaves when something disturbs it.

The mode table leads, because the five names are the vocabulary — short period, phugoid, Dutch
roll, roll subsidence, spiral — and each one is a distinct thing the aircraft does. A damping ratio
without the name attached is a number; with the name it is "it porpoises for eight seconds after a
gust".

Each mode carries its verdict against the handling-qualities floor for *that* mode, because the
floors differ by an order of magnitude: 0.3 is comfortable for a short period and unheard of for a
phugoid. A single "is it damped" threshold would flag every aircraft ever built.

**A divergent spiral is not an alarm.** Most aeroplanes have one; it is flown without noticing
because the doubling time is tens of seconds. The screen says so rather than colouring it red and
training the reader to ignore the colour.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import COLORWAYS, PALETTE, current_theme, mono_font, ui_font
from uavaero.app.widgets.plots import ResponseCanvas, RootLocusCanvas
from uavaero.dynamics.modes import ACCEPTABLE_DAMPING, Mode
from uavaero.dynamics.report import DynamicsReport, excite, time_response
from uavaero.dynamics.statespace import LATERAL_STATES, LONGITUDINAL_STATES

#: How long to run a free response for each mode. A phugoid needs minutes to show its shape; a
#: roll subsidence is over in under a second, and showing it on a 120 s axis would draw a
#: vertical line at the origin.
RESPONSE_DURATION = {
    "short period": 8.0, "phugoid": 180.0, "dutch roll": 25.0,
    "roll subsidence": 5.0, "spiral": 120.0,
}


class DynamicsScreen(QWidget):
    """The five modes, the root locus, and a response for whichever one is selected."""

    mode_selected = Signal(str)
    #: Watch one mode on the 3D view. Carries a `dynamics.shapes.ModeShape` - the screen does
    #: the physics, the window owns the viewport, and neither needs to know about the other.
    fly_mode_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._report: DynamicsReport | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._notes = QLabel()
        self._notes.setWordWrap(True)
        self._notes.setContentsMargins(16, 10, 16, 10)
        self._notes.hide()
        root.addWidget(self._notes)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_table(), 0)
        body.addWidget(self._build_plots(), 1)
        root.addLayout(body, 1)

    def _build_table(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ComparisonPanel")
        panel.setFixedWidth(440)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel("DYNAMIC MODES")
        header.setObjectName("PanelHeader")
        font = ui_font(9)
        font.setBold(True)
        header.setFont(font)
        header.setContentsMargins(16, 12, 16, 8)
        outer.addWidget(header)

        self._trim = QLabel("—")
        self._trim.setProperty("role", "derived-label")
        self._trim.setFont(mono_font(8))
        self._trim.setWordWrap(True)
        self._trim.setContentsMargins(16, 0, 16, 10)
        outer.addWidget(self._trim)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        self._rows = QVBoxLayout(host)
        self._rows.setContentsMargins(16, 0, 16, 16)
        self._rows.setSpacing(0)
        self._rows.addStretch(1)
        scroll.setWidget(host)
        outer.addWidget(scroll, 1)
        return panel

    def _build_plots(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._locus = RootLocusCanvas()
        self._response = ResponseCanvas()
        self._tabs.addTab(self._locus, "Root locus")
        self._tabs.addTab(self._response, "Response")
        layout.addWidget(self._tabs, 1)

        picker = QWidget()
        picker_layout = QHBoxLayout(picker)
        picker_layout.setContentsMargins(16, 8, 16, 10)
        picker_layout.setSpacing(10)
        caption = QLabel("Disturb")
        caption.setProperty("role", "field")
        caption.setFont(ui_font(9))
        self._mode_picker = QComboBox()
        self._mode_picker.setFont(mono_font(9))
        self._mode_picker.setFixedWidth(200)
        self._mode_picker.currentTextChanged.connect(self._show_response)
        # A mode table gives you a damping ratio; this gives you the wallow. The phase between
        # roll and yaw is the character of Dutch roll and it is not in any column.
        self._fly = QPushButton("Fly it in 3D")
        self._fly.setProperty("chrome", "true")
        self._fly.setCursor(Qt.CursorShape.PointingHandCursor)
        self._fly.clicked.connect(self._fly_current_mode)
        self._fly.setEnabled(False)

        picker_layout.addWidget(caption)
        picker_layout.addWidget(self._mode_picker)
        picker_layout.addWidget(self._fly)
        picker_layout.addStretch(1)
        layout.addWidget(picker)
        return container

    # ── content ────────────────────────────────────────────────────────────────────────

    def set_report(self, report: DynamicsReport | None) -> None:
        self._report = report
        self.refresh()

    def refresh(self) -> None:
        self._clear()
        report = self._report
        self._locus.set_report(report)

        blocked = self._mode_picker.blockSignals(True)
        self._mode_picker.clear()
        self._mode_picker.blockSignals(blocked)

        self._fly.setEnabled(report is not None and bool(report.modes))
        if report is None:
            self._notes.hide()
            self._trim.setText(
                "No dynamics yet. Enter mass and inertia on the Airframe screen, then "
                "Solve → Dynamic modes (Ctrl+M)."
            )
            return

        self._render_notes(report)
        self._trim.setText(
            f"level flight at {report.trim.velocity_mps:g} m/s, {report.trim.mass_kg:g} kg  ·  "
            f"CL {report.trim.cl_required:.3f} at {report.trim.alpha_deg:+.1f} deg\n"
            f"{report.summary()}"
        )

        colours = COLORWAYS.get(current_theme(), COLORWAYS["light"])
        for i, mode in enumerate(report.modes):
            self._rows.insertWidget(
                self._rows.count() - 1, _ModeRow(mode, colours[i % len(colours)])
            )
            self._mode_picker.addItem(mode.name)
        self._show_response(self._mode_picker.currentText())

    def _fly_current_mode(self) -> None:
        """Turn the selected mode's eigenvector into motion and hand it to the window.

        The shape is built here rather than in the viewport because it is physics, not drawing:
        `dynamics.shapes` is pure and tested without a graphics context, and the viewport only
        orients an actor.
        """
        from uavaero.dynamics.shapes import mode_shape

        report = self._report
        name = self._mode_picker.currentText()
        if report is None or not name:
            return
        mode = next((m for m in report.modes if m.name == name), None)
        if mode is None or not mode.vector:
            return
        self.fly_mode_requested.emit(mode_shape(mode))

    def _show_response(self, name: str) -> None:
        report = self._report
        if report is None or not name:
            return
        mode = report.named(name)
        if mode is None:
            return

        longitudinal = mode in report.longitudinal
        matrix = report.a_longitudinal if longitudinal else report.a_lateral
        states = LONGITUDINAL_STATES if longitudinal else LATERAL_STATES
        if matrix is None:
            return

        duration = RESPONSE_DURATION.get(name, 60.0)
        times, response = time_response(matrix, excite(mode, states), duration_s=duration)
        self._response.set_response(
            times, response, list(states),
            f"{name} — free response to a disturbance in its own states",
        )

    def _render_notes(self, report: DynamicsReport) -> None:
        if not report.notes:
            self._notes.hide()
            return
        self._notes.setText("⚠  " + "\n⚠  ".join(report.notes))
        self._notes.setStyleSheet(
            f"background: {PALETTE['accent_dim']}; color: {PALETTE['warn']};"
            f" border-bottom: 2px solid {PALETTE['rule_strong']};"
        )
        self._notes.show()

    def _clear(self) -> None:
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def retheme(self) -> None:
        self.refresh()


class _ModeRow(QFrame):
    """One mode: name, verdict, and the numbers that describe it."""

    def __init__(self, mode: Mode, colour: str) -> None:
        super().__init__()
        self.setObjectName("ModelRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 11, 0, 11)
        layout.setSpacing(5)

        top = QHBoxLayout()
        top.setSpacing(10)
        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background: {colour};")

        name = QLabel(mode.name)
        font = ui_font(9)
        font.setBold(True)
        name.setFont(font)

        # A divergent spiral is normal and gets its own word. Calling it UNSTABLE alongside a
        # divergent short period would make the two look equally serious, which they are not.
        if mode.stable:
            verdict, colour_for = "STABLE", PALETTE["good"]
        elif mode.name == "spiral":
            verdict, colour_for = "DIVERGENT", PALETTE["warn"]
        else:
            verdict, colour_for = "UNSTABLE", PALETTE["warn"]
        state = QLabel(verdict)
        state_font = mono_font(8)
        state_font.setBold(True)
        state.setFont(state_font)
        state.setStyleSheet(f"color: {colour_for};")
        state.setAlignment(Qt.AlignmentFlag.AlignRight)

        top.addWidget(swatch)
        top.addWidget(name, 1)
        top.addWidget(state)
        layout.addLayout(top)

        for label, text, flag in self._numbers(mode):
            row = QHBoxLayout()
            row.setSpacing(10)
            caption = QLabel(label)
            caption.setProperty("role", "derived-label")
            caption.setFont(ui_font(8))
            caption.setFixedWidth(150)
            value = QLabel(text)
            value.setFont(mono_font(8))
            value.setProperty("role", "derived")
            if flag is not None:
                value.setStyleSheet(
                    f"color: {PALETTE['good'] if flag else PALETTE['warn']};"
                )
            row.addWidget(caption)
            row.addWidget(value)
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            layout.addWidget(holder)

        self.setStyleSheet(f"#ModelRow {{ border-bottom: 2px solid {PALETTE['rule']}; }}")

    @staticmethod
    def _numbers(mode: Mode) -> list[tuple[str, str, bool | None]]:
        rows: list[tuple[str, str, bool | None]] = []
        if mode.oscillatory:
            floor = ACCEPTABLE_DAMPING.get(mode.name)
            label = "damping ratio" + (f"  (want > {floor})" if floor else "")
            rows.append((label, f"{mode.damping_ratio:.3f}", mode.acceptable))
            rows.append(("period", f"{mode.period:.2f} s", None))
            rows.append(("natural frequency", f"{mode.natural_frequency:.3f} rad/s", None))
        else:
            rows.append(("time constant", f"{mode.time_constant:.2f} s", None)
                        if mode.time_constant else ("root", f"{mode.real:.4f} /s", None))
        if mode.time_to_half is not None:
            rows.append(("time to half", f"{mode.time_to_half:.1f} s", None))
        if mode.time_to_double is not None:
            rows.append(("doubling time", f"{mode.time_to_double:.1f} s", None))

        top = sorted(mode.participation.items(), key=lambda kv: -kv[1])[:2]
        rows.append((
            "dominated by",
            ", ".join(f"{name} {share * 100:.0f}%" for name, share in top),
            None,
        ))
        return rows
