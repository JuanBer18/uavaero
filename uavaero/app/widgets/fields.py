"""The screen's whole vocabulary: a thing you type into, and a thing that was worked out for you.

Those two are visually distinct on purpose and the distinction is load-bearing. An entered value
sits on a two-pixel rule; a derived value has no rule and is amber. In a tool where span, aspect
ratio, area and chord are four views of two independent numbers, being able to see at a glance
which ones you own is the difference between editing an aircraft and fighting one.

**A blank field is "not entered", not zero.** `Field.value()` returns `None` for an empty box, and
callers must treat that as "leave it alone". Returning 0.0 would silently collapse a wing the first
time someone cleared a box to retype it.
"""
from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import mono_font, ui_font

LABEL_WIDTH = 96
VALUE_WIDTH = 72
UNIT_WIDTH = 26


def _label(text: str, role: str, width: int | None = None) -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", role)
    widget.setFont(ui_font(9))
    if width is not None:
        widget.setFixedWidth(width)
    return widget


class Field(QWidget):
    """Label, an underlined value, and a unit. Numeric.

    Commits on editing-finished rather than on every keystroke: a solve is expensive and a
    half-typed `2.` is not a span.
    """

    changed = Signal()

    def __init__(
        self,
        label: str,
        unit: str = "",
        decimals: int = 3,
        minimum: float | None = None,
        maximum: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._minimum = minimum
        self._maximum = maximum

        self.edit = QLineEdit()
        self.edit.setFont(mono_font(9))
        self.edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.edit.setFixedWidth(VALUE_WIDTH)
        self.edit.setPlaceholderText("—")
        self.edit.editingFinished.connect(self._commit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        layout.addWidget(_label(label, "field"), 1)
        layout.addWidget(self.edit)
        layout.addWidget(_label(unit, "unit", UNIT_WIDTH))

    # -- value ---------------------------------------------------------------------------

    def value(self) -> float | None:
        """The number, or None when the box is blank or unparseable. Never 0.0 by default."""
        text = self.edit.text().strip().replace("−", "-")  # accept a real minus sign
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def set_value(self, value: float | None) -> None:
        blocked = self.edit.blockSignals(True)
        self.edit.setText("" if value is None else f"{value:.{self._decimals}f}")
        self.edit.blockSignals(blocked)
        self._mark_valid(True)

    def _commit(self) -> None:
        value = self.value()
        if value is None and self.edit.text().strip():
            self._mark_valid(False)
            return
        if value is not None and not self._in_range(value):
            self._mark_valid(False)
            return
        self._mark_valid(True)
        self.changed.emit()

    def _in_range(self, value: float) -> bool:
        if self._minimum is not None and value < self._minimum:
            return False
        return not (self._maximum is not None and value > self._maximum)

    def _mark_valid(self, valid: bool) -> None:
        """Refusing a value has to be visible. An input that silently keeps the old number is how
        someone comes to believe they set something they did not."""
        self.edit.setProperty("invalid", "false" if valid else "true")
        self.edit.style().unpolish(self.edit)
        self.edit.style().polish(self.edit)


class TextField(QWidget):
    """Label and a free-text underlined value — airfoil names, mostly."""

    changed = Signal()

    def __init__(self, label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.edit = QLineEdit()
        self.edit.setFont(mono_font(9))
        self.edit.setFixedWidth(VALUE_WIDTH + UNIT_WIDTH + 9)
        self.edit.editingFinished.connect(self.changed.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        layout.addWidget(_label(label, "field"), 1)
        layout.addWidget(self.edit)

    def value(self) -> str:
        return self.edit.text().strip()

    def set_value(self, value: str) -> None:
        blocked = self.edit.blockSignals(True)
        self.edit.setText(value)
        self.edit.blockSignals(blocked)


class ChoiceField(QWidget):
    """Label and a fixed set of options."""

    changed = Signal()

    def __init__(self, label: str, options: Iterable[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.combo = QComboBox()
        self.combo.setFont(mono_font(9))
        self.combo.addItems(list(options))
        self.combo.setFixedWidth(VALUE_WIDTH + UNIT_WIDTH + 9)
        self.combo.currentIndexChanged.connect(lambda _: self.changed.emit())

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        layout.addWidget(_label(label, "field"), 1)
        layout.addWidget(self.combo)

    def value(self) -> str:
        return self.combo.currentText()

    def set_value(self, value: str) -> None:
        index = self.combo.findText(value)
        if index >= 0:
            blocked = self.combo.blockSignals(True)
            self.combo.setCurrentIndex(index)
            self.combo.blockSignals(blocked)


class Readout(QWidget):
    """A value the tool worked out. No rule beneath it, and amber — so it can never be mistaken
    for something you can type into."""

    def __init__(
        self, label: str, unit: str = "", decimals: int = 4, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self.value_label = QLabel("—")
        self.value_label.setProperty("role", "derived")
        self.value_label.setFont(mono_font(9))
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value_label.setFixedWidth(VALUE_WIDTH)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        layout.addWidget(_label(label, "derived-label"), 1)
        layout.addWidget(self.value_label)
        layout.addWidget(_label(unit, "unit", UNIT_WIDTH))

    def set_value(self, value: float | None) -> None:
        """None renders as an em dash, never as 0.0000.

        A tailless aircraft does not have a tail volume of nought — it does not have one, and a
        readout showing 0.0000 invites the reader to conclude the aircraft is dangerously
        under-tailed rather than that the question does not apply.
        """
        self.value_label.setText("—" if value is None else f"{value:.{self._decimals}f}")


class GroupLabel(QLabel):
    """A rule with a name on it, separating what you set from what was worked out."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text.upper(), parent)
        self.setProperty("role", "group")
        font = ui_font(8)
        font.setBold(True)
        self.setFont(font)


class FieldColumn(QWidget):
    """A vertical run of fields with the spacing the mockup specifies."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(9)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def add(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget
