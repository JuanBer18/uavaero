"""The drawing. It owns the window, and clicking it is how you select a part.

Painted with QPainter rather than embedded matplotlib. The specification is line art — two-pixel
rules, no fills except a wash on the selected surface — and matplotlib would mean fighting a figure,
its margins and its DPI to get there, for a drawing that has no axes and no ticks.

All geometry comes from `uavaero.geometry.outline`, which is pure Python and tested without a
window open. This widget only fits those coordinates to its rectangle and strokes them.

**The drawing never scrolls and always fills its space** — the one rule that survived from the
previous tool. Zoom is fit-to-widget and recomputed on every resize.
"""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from uavaero.app.style import PALETTE, mono_font
from uavaero.geometry.outline import Outline, bounds, mean_aerodynamic_chord_line, plan_outlines
from uavaero.schema.airframe import ParametricAirframe

#: Fraction of the widget left empty around the aircraft. Enough that a leader line and the
#: caption never touch the geometry.
MARGIN = 0.09


class PlanformView(QWidget):
    """Plan view of the airframe, with the selected surface picked out in amber."""

    selected = Signal(str)  # the kind that was clicked: "wing" | "htail" | "vtail" | "fuselage"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlanformView")
        self.setMinimumSize(320, 220)
        self.setMouseTracking(True)
        self._airframe: ParametricAirframe | None = None
        self._outlines: list[Outline] = []
        self._selection: str = "wing"
        self._hover: str | None = None
        self._scale = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._right_inset = 0

    # -- state ---------------------------------------------------------------------------

    def set_airframe(self, airframe: ParametricAirframe | None) -> None:
        self._airframe = airframe
        self._outlines = plan_outlines(airframe) if airframe is not None else []
        self.update()

    def set_selection(self, kind: str) -> None:
        self._selection = kind
        self.update()

    def selection(self) -> str:
        return self._selection

    def set_right_inset(self, pixels: int) -> None:
        """Keep the aircraft clear of the selection card.

        The one real weakness of a floating card is that it sits on top of the thing it describes —
        select the wing and the card lands on the starboard tip. Rather than moving the card
        somewhere it points at nothing, the drawing gives up the width the card occupies and
        re-fits into what is left. The aircraft gets smaller; it never gets hidden.
        """
        pixels = max(0, int(pixels))
        if pixels != self._right_inset:
            self._right_inset = pixels
            self.update()

    # -- mapping -------------------------------------------------------------------------

    def _recompute_transform(self) -> None:
        """Fit the aircraft to the widget, preserving aspect.

        The plan is drawn with span across the widget and the streamwise axis down it, which is how
        an aircraft is drawn on paper — nose at the top.
        """
        x_min, x_max, y_min, y_max = bounds(self._outlines)
        span_extent = max(y_max - y_min, 1e-9)
        length_extent = max(x_max - x_min, 1e-9)

        drawable_w = max(self.width() - self._right_inset, 120)
        usable_w = drawable_w * (1 - 2 * MARGIN)
        usable_h = self.height() * (1 - 2 * MARGIN)
        self._scale = min(usable_w / span_extent, usable_h / length_extent)

        centre_y = 0.5 * (y_min + y_max)
        centre_x = 0.5 * (x_min + x_max)
        self._offset = QPointF(
            drawable_w / 2.0 - centre_y * self._scale,
            self.height() / 2.0 - centre_x * self._scale,
        )

    def _to_screen(self, point: tuple[float, float]) -> QPointF:
        x, y = point  # x streamwise, y spanwise
        return QPointF(self._offset.x() + y * self._scale, self._offset.y() + x * self._scale)

    def _polygon(self, outline: Outline) -> QPolygonF:
        return QPolygonF([self._to_screen(p) for p in outline.points])

    # -- painting ------------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(PALETTE["inset"]))

        if not self._outlines:
            self._paint_empty(painter)
            painter.end()
            return

        self._recompute_transform()

        for outline in self._outlines:
            selected = outline.kind == self._selection
            hovered = outline.kind == self._hover and not selected
            self._paint_outline(painter, outline, selected=selected, hovered=hovered)

        self._paint_mac(painter)
        self._paint_scale_bar(painter)
        painter.end()

    def _paint_outline(
        self, painter: QPainter, outline: Outline, *, selected: bool, hovered: bool
    ) -> None:
        polygon = self._polygon(outline)
        if selected:
            painter.setBrush(QColor(PALETTE["accent"]).lighter(100))
            painter.setOpacity(0.10)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(polygon)
            painter.setOpacity(1.0)
            pen = QPen(QColor(PALETTE["accent"]), 3.0)
        elif hovered:
            pen = QPen(QColor(PALETTE["ink"]), 2.0)
        elif outline.kind == "fuselage":
            pen = QPen(QColor(PALETTE["ink_3"]), 1.4)
        else:
            pen = QPen(QColor(PALETTE["ink"]), 1.4)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolygon(polygon)

    def _paint_mac(self, painter: QPainter) -> None:
        """The mean aerodynamic chord, and the aerodynamic centre a quarter along it.

        Drawn because every tail arm in the tool is measured from that point, so the one mark that
        explains the tail volumes belongs on the drawing rather than only in a readout.
        """
        if self._airframe is None:
            return
        line = mean_aerodynamic_chord_line(self._airframe.wing)
        if len(line) != 2:
            return
        start, end = self._to_screen(line[0]), self._to_screen(line[1])
        painter.setPen(QPen(QColor(PALETTE["accent"]), 1.6))
        painter.drawLine(start, end)

        quarter = QPointF(
            start.x() + 0.25 * (end.x() - start.x()), start.y() + 0.25 * (end.y() - start.y())
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(quarter, 3.4, 3.4)

    def _paint_scale_bar(self, painter: QPainter) -> None:
        """A metre, drawn to scale. Without it the drawing has no size at all — every aircraft
        fills the window identically, so a 2.4 m wing and a 24 m wing look the same."""
        if self._scale <= 0:
            return
        metre = 1.0 * self._scale
        while metre > self.width() * 0.4:
            metre /= 2.0
        length_m = metre / self._scale

        y = self.height() - 22
        x0 = 20.0
        painter.setPen(QPen(QColor(PALETTE["ink_3"]), 2.0))
        painter.drawLine(QPointF(x0, y), QPointF(x0 + metre, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x0 + metre, y - 4), QPointF(x0 + metre, y + 4))

        painter.setFont(mono_font(8))
        painter.drawText(
            QRectF(x0, y + 5, 120, 16),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            f"{length_m:g} m",
        )

    def _paint_empty(self, painter: QPainter) -> None:
        painter.setPen(QColor(PALETTE["ink_3"]))
        font = QFont()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(
            self.rect(), int(Qt.AlignmentFlag.AlignCenter), "No airframe open"
        )

    # -- interaction ---------------------------------------------------------------------

    def _hit(self, position: QPoint) -> str | None:
        """Which part is under the cursor. Surfaces before bodies, because a wing root sits over
        the fuselage and the wing is what you meant."""
        point = QPointF(position)
        for outline in reversed(self._outlines):
            path = QPainterPath()
            path.addPolygon(self._polygon(outline))
            if path.contains(point):
                return outline.kind
        return None

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        kind = self._hit(event.position().toPoint())
        if kind is not None:
            self._selection = kind
            self.selected.emit(kind)
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        kind = self._hit(event.position().toPoint())
        if kind != self._hover:
            self._hover = kind
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if kind
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._hover = None
        self.update()
