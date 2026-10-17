"""Static stability: the derivative set, where it came from, and where the codes disagree.

Three things are on this screen and their order is the argument.

**The verdicts first**, because "is it stable" is the question, and it has three separate answers
on three axes with three different sign conventions. Nobody should have to remember that Cn_β
positive is stable while Cm_α negative is stable while Cl_β negative is stable — the screen says
STABLE or UNSTABLE and the derivative sits beside it.

**The static margin next**, with all three routes to it shown. They come from one solve by
different arithmetic, so when they agree the whole derivative set is being read correctly. Showing
only the answer would throw away the check.

**The disagreement last**, and it is not hidden. VSPAERO and AVL are two independent codes; where
they differ by a factor of two on Cm_α that is a real fact about how well this aircraft is known,
and the screen reports both rather than picking one. Same reasoning as the drag comparison.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import PALETTE, mono_font, ui_font
from uavaero.schema.stability import STATIC_MARGIN_TYPICAL, StabilityResult
from uavaero.store.airframe_file import AirframeFile

AXES = (
    ("longitudinal", "Longitudinal", "Cm_alpha", "Cm_a", "pitches nose-down when disturbed"),
    ("directional", "Directional", "Cn_beta", "Cn_b", "yaws to face the relative wind"),
    ("lateral", "Lateral", "Cl_beta", "Cl_b", "rolls away from sideslip (dihedral effect)"),
)

STATIC_DERIVATIVES = (
    ("CL_alpha", "CL_a", "lift-curve slope"),
    ("CD_alpha", "CD_a", "drag with incidence"),
    ("Cm_alpha", "Cm_a", "pitch stiffness"),
    ("CY_beta", "CY_b", "side force with sideslip"),
    ("Cl_beta", "Cl_b", "dihedral effect"),
    ("Cn_beta", "Cn_b", "weathercock stiffness"),
)

RATE_DERIVATIVES = (
    ("CL_q", "CL_q", "lift with pitch rate"),
    ("Cm_q", "Cm_q", "pitch damping"),
    ("Cl_p", "Cl_p", "roll damping"),
    ("Cn_p", "Cn_p", "yaw from roll rate"),
    ("Cl_r", "Cl_r", "roll from yaw rate"),
    ("Cn_r", "Cn_r", "yaw damping"),
)


class StabilityScreen(QWidget):
    """Read the derivative set."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file: AirframeFile | None = None
        self._result: StabilityResult | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._notes = QLabel()
        self._notes.setWordWrap(True)
        self._notes.setContentsMargins(16, 10, 16, 10)
        self._notes.hide()
        root.addWidget(self._notes)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._host = QWidget()
        self._body = QVBoxLayout(self._host)
        self._body.setContentsMargins(20, 16, 20, 24)
        self._body.setSpacing(0)
        self._body.addStretch(1)
        scroll.setWidget(self._host)
        root.addWidget(scroll, 1)

    def set_result(self, result: StabilityResult | None) -> None:
        self._result = result
        self.refresh()

    def set_file(self, file: AirframeFile | None) -> None:
        self._file = file
        self.refresh()

    def refresh(self) -> None:
        self._clear()
        result = self._result
        if result is None:
            self._notes.hide()
            self._body.insertWidget(0, _muted(
                "No stability run yet. Solve → Stability derivatives, or Ctrl+D."
            ))
            return

        self._render_notes(result)
        self._body.insertWidget(self._body.count() - 1, _Verdicts(result))
        self._body.insertWidget(self._body.count() - 1, _StaticMargin(result))
        self._body.insertWidget(
            self._body.count() - 1,
            _Derivatives("Static derivatives", STATIC_DERIVATIVES, result,
                         source=result.static_source),
        )
        self._body.insertWidget(
            self._body.count() - 1,
            _Derivatives("Rate derivatives", RATE_DERIVATIVES, result,
                         source=result.rate_source or "not available",
                         warn=not result.rates_validated),
        )
        if result.cross_check:
            self._body.insertWidget(self._body.count() - 1, _CrossCheck(result))

    def _render_notes(self, result: StabilityResult) -> None:
        if not result.notes:
            self._notes.hide()
            return
        self._notes.setText("⚠  " + "\n⚠  ".join(result.notes))
        self._notes.setStyleSheet(
            f"background: {PALETTE['accent_dim']}; color: {PALETTE['warn']};"
            f" border-bottom: 2px solid {PALETTE['rule_strong']};"
        )
        self._notes.show()

    def _clear(self) -> None:
        while self._body.count() > 1:
            item = self._body.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def retheme(self) -> None:
        self.refresh()


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "derived-label")
    label.setFont(ui_font(9))
    label.setWordWrap(True)
    return label


def _heading(text: str, trailing: str = "") -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 18, 0, 8)
    label = QLabel(text.upper())
    font = ui_font(9)
    font.setBold(True)
    label.setFont(font)
    label.setStyleSheet(f"color: {PALETTE['accent']};")
    layout.addWidget(label)
    layout.addStretch(1)
    if trailing:
        note = QLabel(trailing)
        note.setProperty("role", "derived-label")
        note.setFont(mono_font(8))
        layout.addWidget(note)
    return row


class _Verdicts(QFrame):
    """Is it stable, on each axis, with the derivative that decides it."""

    def __init__(self, result: StabilityResult) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_heading("Stability", f"at {result.alpha_deg:g} deg, {result.velocity_mps:g} m/s"))

        verdicts = result.verdicts()
        for key, label, field, symbol, meaning in AXES:
            stable = verdicts[key]
            row = QHBoxLayout()
            row.setSpacing(12)

            name = QLabel(label)
            name.setFont(ui_font(9))
            name.setFixedWidth(120)

            verdict = QLabel("STABLE" if stable else "UNSTABLE")
            font = mono_font(9)
            font.setBold(True)
            verdict.setFont(font)
            verdict.setFixedWidth(90)
            verdict.setStyleSheet(f"color: {PALETTE['good'] if stable else PALETTE['warn']};")

            value = QLabel(f"{symbol} {getattr(result, field):+.4f} /rad")
            value.setFont(mono_font(9))
            value.setFixedWidth(170)
            value.setProperty("role", "derived")

            because = QLabel(meaning)
            because.setProperty("role", "derived-label")
            because.setFont(ui_font(8))

            row.addWidget(name)
            row.addWidget(verdict)
            row.addWidget(value)
            row.addWidget(because, 1)
            container = QWidget()
            container.setLayout(row)
            layout.addWidget(container)


class _StaticMargin(QFrame):
    """The headline number, and the three independent routes to it."""

    def __init__(self, result: StabilityResult) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_heading("Static margin", f"CG at x {result.x_cg_m:.4f} m"))

        margin = result.static_margin
        low, high = STATIC_MARGIN_TYPICAL
        colour = PALETTE["good"] if margin is not None and low <= margin <= high else PALETTE["warn"]

        headline = QLabel("—" if margin is None else f"{margin * 100:.1f} % MAC")
        font = mono_font(18)
        font.setBold(True)
        headline.setFont(font)
        headline.setStyleSheet(f"color: {colour};")
        layout.addWidget(headline)

        if result.neutral_point_x_m is not None:
            layout.addWidget(_muted(
                f"neutral point at x {result.neutral_point_x_m:.4f} m, "
                f"MAC {result.cref_m:.4f} m"
            ))

        routes = (
            ("solver reported", result.static_margin_reported),
            ("from neutral point", result.static_margin_from_neutral_point),
            ("from -Cm_a / CL_a", result.static_margin_from_derivatives),
        )
        for label, value in routes:
            if value is None:
                continue
            row = QHBoxLayout()
            row.setSpacing(12)
            caption = QLabel(label)
            caption.setProperty("role", "derived-label")
            caption.setFont(ui_font(8))
            caption.setFixedWidth(170)
            number = QLabel(f"{value:.4f}")
            number.setFont(mono_font(8))
            number.setProperty("role", "derived")
            row.addWidget(caption)
            row.addWidget(number)
            row.addStretch(1)
            holder = QWidget()
            holder.setLayout(row)
            layout.addWidget(holder)

        agreement = result.consistency()
        layout.addWidget(_muted(
            agreement or "All three routes agree, so the perturbation arithmetic is being read "
                         "correctly and the derivatives below can be trusted."
        ))


class _Derivatives(QFrame):
    """A block of derivatives, labelled with which code produced them."""

    def __init__(self, title: str, fields, result: StabilityResult,
                 source: str, warn: bool = False) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_heading(title, f"from {source}"))

        if warn:
            layout.addWidget(_muted(
                "These did not pass the damping sign check and must not be used."
            ))

        for field, symbol, meaning in fields:
            row = QHBoxLayout()
            row.setSpacing(12)
            name = QLabel(symbol)
            name.setFont(mono_font(9))
            name.setFixedWidth(70)
            value = QLabel(f"{getattr(result, field):+.4f}")
            value.setFont(mono_font(9))
            value.setProperty("role", "derived")
            value.setFixedWidth(90)
            value.setAlignment(Qt.AlignmentFlag.AlignRight)
            unit = QLabel("/rad")
            unit.setProperty("role", "unit")
            unit.setFont(mono_font(8))
            unit.setFixedWidth(40)
            because = QLabel(meaning)
            because.setProperty("role", "derived-label")
            because.setFont(ui_font(8))
            row.addWidget(name)
            row.addWidget(value)
            row.addWidget(unit)
            row.addWidget(because, 1)
            holder = QWidget()
            holder.setLayout(row)
            layout.addWidget(holder)


class _CrossCheck(QFrame):
    """VSPAERO against AVL, derivative by derivative."""

    def __init__(self, result: StabilityResult) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(_heading("Second opinion", "VSPAERO vs AVL"))
        layout.addWidget(_muted(
            "Two independent codes on the same geometry, moments taken about the same point. "
            "Where they differ, both are reported and neither is corrected — the same reasoning "
            "that keeps three drag models side by side rather than averaging them."
        ))

        header = QHBoxLayout()
        header.setSpacing(12)
        for text, width in (("", 70), ("VSPAERO", 90), ("AVL", 90), ("difference", 90)):
            label = QLabel(text)
            label.setProperty("role", "derived-label")
            label.setFont(mono_font(8))
            label.setFixedWidth(width)
            label.setAlignment(Qt.AlignmentFlag.AlignRight)
            header.addWidget(label)
        header.addStretch(1)
        holder = QWidget()
        holder.setLayout(header)
        layout.addWidget(holder)

        for name, row in result.cross_check.items():
            if not isinstance(row, dict):
                continue
            agrees = row.get("agrees", True)
            line = QHBoxLayout()
            line.setSpacing(12)
            for text, width, colour in (
                (name, 70, PALETTE["ink_2"]),
                (f"{row['vspaero']:+.4f}", 90, PALETTE["ink"]),
                (f"{row['avl']:+.4f}", 90, PALETTE["ink"]),
                (f"{row['relative_error'] * 100:+.0f}%", 90,
                 PALETTE["ink_3"] if agrees else PALETTE["warn"]),
            ):
                label = QLabel(text)
                label.setFont(mono_font(8))
                label.setFixedWidth(width)
                label.setAlignment(Qt.AlignmentFlag.AlignRight)
                label.setStyleSheet(f"color: {colour};")
                line.addWidget(label)
            line.addStretch(1)
            container = QWidget()
            container.setLayout(line)
            layout.addWidget(container)
