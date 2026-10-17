"""The drag comparison: three independent estimates of the same number, side by side.

This screen is the point of the tool. Every other aerodynamics package reports one drag coefficient
and leaves you to guess how much to trust it. This one reports three, says how each was arrived at,
and puts the disagreement between them on screen as a number — because that disagreement is the
honest error bar on a preliminary-design prediction, and it is information you cannot get any other
way at this stage.

So the layout puts the spread where it cannot be missed, and gives every model its `basis` sentence
rather than a bare label. When two methods differ by 40% the first question is always what each one
actually did, and an answer that has to be reconstructed from source is an answer nobody checks.

Warnings live at the top rather than the bottom. A model that could not run, a section the polar
source could not answer for, a station over its stall limit — all of those make the numbers below
mean something different, so they are read first or not at all.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import COLORWAYS, PALETTE, current_theme, mono_font, ui_font
from uavaero.app.widgets.plots import DragPolarCanvas, LiftToDragCanvas, SpanLoadCanvas
from uavaero.schema.results import AeroResult
from uavaero.store.airframe_file import AirframeFile

#: Above this, the models are telling materially different stories and the number is worth
#: colouring. Not a correctness threshold — a large spread on a preliminary estimate is a fact
#: about the state of knowledge, not a bug.
SPREAD_WATCH = 0.15


class ComparisonScreen(QWidget):
    """Read the drag models against each other."""

    reference_point_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file: AirframeFile | None = None
        self._result: AeroResult | None = None
        self._index = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._notes = QLabel()
        self._notes.setWordWrap(True)
        self._notes.setProperty("role", "notes")
        self._notes.setContentsMargins(16, 10, 16, 10)
        self._notes.hide()
        root.addWidget(self._notes)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_table(), 0)
        body.addWidget(self._build_plots(), 1)
        root.addLayout(body, 1)

    # ── the table ──────────────────────────────────────────────────────────────────────

    def _build_table(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ComparisonPanel")
        panel.setFixedWidth(430)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel("AERODYNAMIC DRAG COMPARISON")
        header.setObjectName("PanelHeader")
        font = ui_font(9)
        font.setBold(True)
        header.setFont(font)
        header.setContentsMargins(16, 12, 16, 10)
        outer.addWidget(header)

        self._point = QLabel("—")
        self._point.setProperty("role", "derived-label")
        self._point.setFont(mono_font(8))
        self._point.setContentsMargins(16, 0, 16, 10)
        outer.addWidget(self._point)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._rows_host = QWidget()
        self._rows = QVBoxLayout(self._rows_host)
        self._rows.setContentsMargins(16, 0, 16, 16)
        self._rows.setSpacing(0)
        self._rows.addStretch(1)
        scroll.setWidget(self._rows_host)
        outer.addWidget(scroll, 1)
        return panel

    def _build_plots(self) -> QWidget:
        self._tabs = QTabWidget()
        self._polar_plot = DragPolarCanvas()
        self._ld_plot = LiftToDragCanvas()
        self._span_plot = SpanLoadCanvas()
        self._tabs.addTab(self._polar_plot, "Drag polar")
        self._tabs.addTab(self._ld_plot, "L/D")
        self._tabs.addTab(self._span_plot, "Span load")
        return self._tabs

    # ── content ────────────────────────────────────────────────────────────────────────

    def set_file(self, file: AirframeFile | None) -> None:
        self._file = file
        solve = file.latest() if file else None
        self._result = solve.result if solve else None
        self.refresh()

    def refresh(self) -> None:
        self._clear_rows()
        result = self._result
        for canvas in (self._polar_plot, self._ld_plot, self._span_plot):
            canvas.set_result(result)

        if result is None:
            self._point.setText("No solve yet for this airframe")
            self._notes.hide()
            return

        self._index = min(max(len(result.polar.CL) // 2, 0), len(result.polar.CL) - 1)
        self._render_notes(result)
        self._render_rows(result)

    def _render_notes(self, result: AeroResult) -> None:
        """Warnings first. A model that could not run makes every number below mean something
        different, so it is read before them or not at all."""
        if not result.notes:
            self._notes.hide()
            return
        self._notes.setText("⚠  " + "\n⚠  ".join(result.notes))
        self._notes.setStyleSheet(
            f"background: {PALETTE['accent_dim']}; color: {PALETTE['warn']};"
            f" border-bottom: 2px solid {PALETTE['rule_strong']};"
        )
        self._notes.show()

    def _render_rows(self, result: AeroResult) -> None:
        index = self._index
        alpha = float(result.polar.alpha_deg[index])
        cl = float(result.polar.CL[index])
        self._point.setText(
            f"at alpha {alpha:+.1f} deg   CL {cl:.3f}   ·   {result.solver} "
            f"{result.fidelity} {result.scope}"
        )

        colours = COLORWAYS.get(current_theme(), COLORWAYS["light"])
        for position, model in enumerate(result.drag.models.values()):
            self._rows.insertWidget(
                self._rows.count() - 1,
                _ModelRow(model, cl, index, colours[position % len(colours)]),
            )

        spread = result.drag.spread(index)
        relative = result.drag.relative_spread(index)
        self._rows.insertWidget(self._rows.count() - 1, _SpreadRow(spread, relative))

    def _clear_rows(self) -> None:
        while self._rows.count() > 1:
            item = self._rows.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def retheme(self) -> None:
        for canvas in (self._polar_plot, self._ld_plot, self._span_plot):
            canvas.redraw()
        self.refresh()


class _ModelRow(QFrame):
    """One model: its name in its series colour, its numbers, and how it got them."""

    def __init__(self, model, cl: float, index: int, colour: str) -> None:
        super().__init__()
        self.setObjectName("ModelRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 12, 0, 12)
        layout.setSpacing(5)

        cd = model.cd[index] if index < len(model.cd) else float("nan")
        ratio = cl / cd if cd > 0 else 0.0

        top = QHBoxLayout()
        top.setSpacing(10)
        swatch = QLabel()
        swatch.setFixedSize(10, 10)
        swatch.setStyleSheet(f"background: {colour};")
        name = QLabel(model.label + ("   (reference)" if model.reference else ""))
        font = ui_font(9)
        font.setBold(True)
        name.setFont(font)
        value = QLabel(f"{cd:.5f}")
        value.setFont(mono_font(10))
        value.setStyleSheet(f"color: {colour};")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        ld = QLabel(f"L/D {ratio:.1f}")
        ld.setFont(mono_font(9))
        ld.setProperty("role", "derived-label")
        ld.setFixedWidth(78)
        ld.setAlignment(Qt.AlignmentFlag.AlignRight)

        top.addWidget(swatch)
        top.addWidget(name, 1)
        top.addWidget(value)
        top.addWidget(ld)
        layout.addLayout(top)

        basis = QLabel(model.basis)
        basis.setWordWrap(True)
        basis.setProperty("role", "derived-label")
        basis.setFont(ui_font(8))
        basis.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        layout.addWidget(basis)

        self.setStyleSheet(f"#ModelRow {{ border-bottom: 2px solid {PALETTE['rule']}; }}")


class _SpreadRow(QFrame):
    """How far apart the predictions are — the number this whole screen exists to show."""

    def __init__(self, spread: float, relative: float) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(4)

        colour = PALETTE["warn"] if relative > SPREAD_WATCH else PALETTE["ink_2"]
        for label, text in (
            ("Spread", f"{spread:.5f}"),
            ("Relative spread", f"{relative * 100:.1f} %"),
        ):
            row = QHBoxLayout()
            row.setSpacing(10)
            caption = QLabel(label)
            caption.setFont(ui_font(9))
            caption.setProperty("role", "derived-label")
            value = QLabel(text)
            value.setFont(mono_font(10))
            value.setStyleSheet(f"color: {colour};")
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(caption, 1)
            row.addWidget(value)
            layout.addLayout(row)

        note = QLabel(
            "Spread is between predictions only — a measurement added later is scored against "
            "them, never averaged into them."
        )
        note.setWordWrap(True)
        note.setProperty("role", "derived-label")
        note.setFont(ui_font(8))
        layout.addWidget(note)
