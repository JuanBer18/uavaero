"""The airframe screen: the drawing owns the window, you click a part, its boxes open beside it.

Built against `docs/mockups/phase1-airframe-screen.html`, which is the specification rather than an
inspiration — panel order, which values are derived, where the chips live and what the footer holds
all come from it.

The one structural idea: **nothing floats over the drawing except the card for the part you
selected**. Component chips and the whole-aircraft readout live in the footer bar, so the aircraft
is never obscured by chrome you are not using.

Edits go through `geometry.authoring`, which applies them to the one `ParametricAirframe` the file
holds. No napkin numbers are stored anywhere — every box on this screen is read back off the
geometry when the card is built, so nothing here can disagree with what the solver is handed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QStackedWidget,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import PALETTE, mono_font, ui_font
from uavaero.app.widgets.fields import (
    ChoiceField,
    Field,
    FieldColumn,
    GroupLabel,
    Readout,
    TextField,
)
from uavaero.app.viewport.viewport import Viewport3D
from uavaero.app.widgets.planform_view import PlanformView
from uavaero.geometry import authoring
from uavaero.geometry.fuselage_gen import FUSELAGE_STYLES, WING_POSITIONS
from uavaero.geometry.planform import SWEEP_MODES, Planform
from uavaero.schema.condition import SCOPES, Fidelity, OperatingCondition
from uavaero.schema.mass import MassProperties
from uavaero.solvers.vspaero import widest_runnable_scope
from uavaero.store.airframe_file import AirframeFile

CARD_MARGIN = 18


class AirframeScreen(QWidget):
    """Author a complete aircraft. No mass, no mission, no requirement, no sizing."""

    changed = Signal()          # the airframe was edited
    notice = Signal(str)        # something the user needs told, e.g. the scope was narrowed
    solve_requested = Signal()
    save_requested = Signal()
    open_in_vsp_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file: AirframeFile | None = None
        self._selection = "wing"
        self._card: QFrame | None = None
        self._readouts: dict[str, QLabel] = {}
        self._component_buttons: dict[str, QPushButton] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_chrome())
        # Planform and 3D share one stage, and the selection card floats over whichever is on
        # top. Rule 10: the drawing fills its space at all times — a stack keeps that true for
        # both without either having to shrink for the other.
        self.view = PlanformView()
        self.view.selected.connect(self._on_view_selection)
        self.viewport = Viewport3D()
        self._stage = QStackedWidget()
        self._stage.addWidget(self.view)
        self._stage.addWidget(self.viewport)
        root.addWidget(self._stage, 1)
        root.addWidget(self._build_footer())

    # ── chrome ─────────────────────────────────────────────────────────────────────────

    def _build_chrome(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("ChromeBar")
        bar.setFixedHeight(46)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(14)

        name = QLabel("UAV Studio")
        name.setObjectName("AppName")
        font = ui_font(12)
        font.setBold(True)
        name.setFont(font)

        self._doc_label = QLabel("—")
        self._doc_label.setObjectName("DocName")
        self._doc_label.setFont(mono_font(9))

        # Plan / 3D, as a pair of chips rather than a menu: it is a view switch you make
        # constantly while authoring, and it belongs next to the thing it switches.
        self._view_buttons: dict[str, QPushButton] = {}
        for label, mode in (("Plan", "plan"), ("3D", "3d")):
            button = QPushButton(label)
            button.setProperty("chip", "true")
            button.setCheckable(True)
            button.setChecked(mode == "plan")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, m=mode: self.set_view_mode(m))
            self._view_buttons[mode] = button

        layout.addWidget(name)
        layout.addWidget(self._doc_label)
        layout.addStretch(1)
        layout.addWidget(self._view_buttons["plan"])
        layout.addWidget(self._view_buttons["3d"])
        for text, slot, primary in (
            ("Open in OpenVSP", self.open_in_vsp_requested.emit, False),
            ("Save", self.save_requested.emit, False),
            ("Solve", self.solve_requested.emit, True),
        ):
            button = QPushButton(text)
            button.setProperty("chrome", "true")
            button.setProperty("primary", "true" if primary else "false")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return bar

    # ── footer ─────────────────────────────────────────────────────────────────────────

    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("FooterBar")
        bar.setFixedHeight(46)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        chips = QWidget()
        chip_layout = QHBoxLayout(chips)
        chip_layout.setContentsMargins(0, 0, 0, 0)
        chip_layout.setSpacing(0)
        self._chip_group = QButtonGroup(self)
        self._chip_group.setExclusive(True)
        self._chip_layout = chip_layout

        self._add_component = QToolButton()
        self._add_component.setText("+ Add component")
        self._add_component.setProperty("chip", "true")
        self._add_component.setProperty("add", "true")
        self._add_component.setCursor(Qt.CursorShape.PointingHandCursor)
        self._add_component.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._add_component.setMenu(QMenu(self._add_component))
        chip_layout.addWidget(self._add_component)
        chip_layout.addStretch(1)

        layout.addWidget(chips, 1)
        layout.addWidget(self._build_readout())
        return bar

    def _build_readout(self) -> QWidget:
        strip = QFrame()
        strip.setObjectName("ReadoutStrip")
        strip.setFixedWidth(400)
        layout = QHBoxLayout(strip)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(20)
        for key, label in (("vh", "Vh"), ("vv", "Vv"), ("cg", "CG"), ("re", "Re")):
            cell = QHBoxLayout()
            cell.setSpacing(6)
            caption = QLabel(label)
            caption.setProperty("role", "readout-label")
            caption.setFont(mono_font(8))
            value = QLabel("—")
            value.setProperty("role", "readout-value")
            value.setFont(mono_font(9))
            self._readouts[key] = value
            cell.addWidget(caption)
            cell.addWidget(value)
            layout.addLayout(cell)
        layout.addStretch(1)
        return strip

    # ── document ───────────────────────────────────────────────────────────────────────

    def set_file(self, file: AirframeFile | None) -> None:
        self._file = file
        self._doc_label.setText(file.name if file else "—")
        self.view.set_airframe(file.airframe if file else None)
        self._refresh_viewport(reset_camera=True)
        self._rebuild_component_chips()
        self.refresh()

    def refresh(self) -> None:
        """Redraw everything from the airframe. Derived values are recomputed rather than cached,
        so there is no way for the readout to be showing a number the geometry no longer supports."""
        if self._file is None:
            self._close_card()
            for label in self._readouts.values():
                label.setText("—")
            return
        self.view.set_airframe(self._file.airframe)
        self.view.set_selection(self._selection)
        self._refresh_viewport()
        self._refresh_readout()
        self._build_card()

    def _refresh_readout(self) -> None:
        assert self._file is not None
        facts = authoring.summary(self._file.airframe)
        condition = self._file.condition
        mac = facts["mac_m"] or 0.0
        mac_le = facts["mac_le_x_m"] or 0.0
        cg_fraction = ((condition.xyz_ref[0] - mac_le) / mac * 100.0) if mac > 0 else None
        reynolds = condition.atmosphere.reynolds(condition.velocity_mps, mac)

        self._set_readout("vh", facts["horizontal_tail_volume"], "{:.3f}", authoring.VH_TYPICAL)
        self._set_readout("vv", facts["vertical_tail_volume"], "{:.4f}", authoring.VV_TYPICAL)
        self._set_readout("cg", cg_fraction, "{:.1f}%", None)
        self._set_readout("re", reynolds, "{:,.0f}", None)

    def _set_readout(
        self, key: str, value: float | None, fmt: str, band: tuple[float, float] | None
    ) -> None:
        label = self._readouts[key]
        label.setText("—" if value is None else fmt.format(value))
        # State colour is separate from the accent: amber means "derived", green and orange mean
        # "this is the state of your design".
        state = ""
        if value is not None and band is not None:
            state = "good" if band[0] <= value <= band[1] else "warn"
        label.setProperty("state", state)
        label.style().unpolish(label)
        label.style().polish(label)

    # ── selection and the card ─────────────────────────────────────────────────────────

    def set_view_mode(self, mode: str) -> None:
        """Switch the stage between the planform drawing and the 3D scene.

        The 3D widget builds itself on first activation, not before — see `Viewport3D`. Switching
        back and forth after that is just a stack index, and the camera survives it.
        """
        for name, button in self._view_buttons.items():
            button.setChecked(name == mode)
        if mode == "3d":
            self._stage.setCurrentWidget(self.viewport)
            self.viewport.activate()
            if self._file is not None:
                self.viewport.show_airframe(self._file.airframe, reset_camera=True)
        else:
            self._stage.setCurrentWidget(self.view)
        self._place_card()

    def _refresh_viewport(self, reset_camera: bool = False) -> None:
        """Push the current aircraft at the 3D scene.

        Cheap when the view has never been opened — `show_airframe` just remembers the airframe
        and returns, so authoring with the 3D view closed costs nothing.
        """
        if self._file is not None:
            self.viewport.show_airframe(self._file.airframe, reset_camera=reset_camera)

    def _select(self, kind: str) -> None:
        self._selection = kind
        self.view.set_selection(kind if kind != "condition" else "")
        self._sync_chips()
        self._build_card()

    def _on_view_selection(self, kind: str) -> None:
        self._selection = kind
        self._sync_chips()
        self._build_card()

    def _sync_chips(self) -> None:
        for kind, button in self._component_buttons.items():
            button.setChecked(kind == self._selection)

    def _rebuild_component_chips(self) -> None:
        """Show the components that exist, followed by a compact add menu.

        The old footer showed every possible component as an equal-width tab, including absent
        ones. That made a wing-only aircraft look complete and left no natural way to add or
        remove geometry. The footer is now a small component collection rather than navigation.
        """
        for button in list(self._component_buttons.values()):
            self._chip_group.removeButton(button)
            self._chip_layout.removeWidget(button)
            button.deleteLater()
        self._component_buttons.clear()

        airframe = self._file.airframe if self._file is not None else None
        items: list[tuple[str, str]] = []
        if airframe is not None:
            items.append(("wing", "Main wing"))
            if airframe.htail is not None:
                items.append(
                    ("htail", "V-tail" if authoring.is_v_tail(airframe) else "Horizontal tail")
                )
            if airframe.vtail is not None:
                items.append(("vtail", "Vertical tail"))
            if airframe.fuselages:
                items.append(("fuselage", "Fuselage"))
            items.extend((("mass", "Mass"), ("condition", "Condition")))

        insertion = self._chip_layout.indexOf(self._add_component)
        for offset, (kind, label) in enumerate(items):
            button = QPushButton(label)
            button.setProperty("chip", "true")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, k=kind: self._select(k))
            self._chip_group.addButton(button)
            self._chip_layout.insertWidget(insertion + offset, button)
            self._component_buttons[kind] = button

        menu = self._add_component.menu()
        assert menu is not None
        menu.clear()
        if airframe is not None:
            actions = (
                ("Horizontal tail", "htail", airframe.htail is None),
                ("Vertical tail", "vtail", airframe.vtail is None),
                ("V-tail", "v_tail", airframe.htail is None and airframe.vtail is None),
                ("Fuselage", "fuselage", not airframe.fuselages),
            )
            for label, kind, enabled in actions:
                action = menu.addAction(label)
                action.setEnabled(enabled)
                action.triggered.connect(lambda _=False, k=kind: self._add_optional_component(k))
        self._add_component.setEnabled(
            airframe is not None and any(action.isEnabled() for action in menu.actions())
        )
        self._sync_chips()

    def _add_optional_component(self, kind: str) -> None:
        """The menu already disables what cannot be added, so a refusal here means the menu was
        stale. Say so rather than raising into Qt, where it surfaces as a console traceback and
        leaves the screen half-updated."""
        assert self._file is not None
        try:
            self._selection = authoring.add_default_component(self._file.airframe, kind)
        except ValueError as exc:
            self.notice.emit(f"could not add that component: {exc}")
            self._rebuild_component_chips()
            return
        self._rebuild_component_chips()
        self._after_component_change()

    def _remove_optional_component(self, kind: str) -> None:
        assert self._file is not None
        try:
            authoring.remove_component(self._file.airframe, kind)
        except ValueError as exc:
            self.notice.emit(f"could not remove that component: {exc}")
            return
        self._selection = "wing"
        self._rebuild_component_chips()
        self._after_component_change()

    def _after_component_change(self) -> None:
        assert self._file is not None
        self._narrow_scope_to_the_geometry()
        self.view.set_airframe(self._file.airframe)
        self.view.set_selection(self._selection)
        self._refresh_viewport()
        self._refresh_readout()
        self._build_card()
        self.changed.emit()

    def _narrow_scope_to_the_geometry(self) -> None:
        """Keep the operating condition solvable after a component is deleted.

        Removing the fuselage while the scope says `aircraft` leaves a file that cannot be solved,
        and the only symptom is a refusal when Solve is pressed — which reads as the tool being
        broken rather than as one setting that needs changing. So the scope follows the geometry
        down, and says so rather than doing it quietly.

        It never widens. Adding a fuselage back does not promote a `wing_tail` scope, because
        solving the wing and tails without the body may be exactly the question being asked.
        """
        assert self._file is not None
        before = self._file.condition.scope
        after = widest_runnable_scope(self._file.airframe, before)
        if after == before:
            return
        self._file.condition = replace(self._file.condition, scope=after)
        self.notice.emit(
            f"scope narrowed from '{before}' to '{after}' - the {before} scope needs geometry "
            f"this aircraft no longer has"
        )

    def _close_card(self) -> None:
        if self._card is not None:
            self._card.deleteLater()
            self._card = None
        self.view.set_right_inset(0)

    def _build_card(self) -> None:
        self._close_card()
        if self._file is None:
            return
        builder = {
            "wing": self._wing_card,
            "htail": lambda body: self._tail_card(body, "htail"),
            "vtail": lambda body: self._tail_card(body, "vtail"),
            "fuselage": self._fuselage_card,
            "mass": self._mass_card,
            "condition": self._condition_card,
        }.get(self._selection)
        if builder is None:
            return

        htail_title = "Horizontal tail"
        if self._file is not None and authoring.is_v_tail(self._file.airframe):
            htail_title = "V-tail"
        title = {"wing": "Wing", "htail": htail_title, "vtail": "Vertical tail",
                 "fuselage": "Fuselage", "mass": "Mass and inertia",
                 "condition": "Condition"}[self._selection]
        card, body = self._card_frame(title)
        builder(body)
        card.adjustSize()
        self._place_card()
        card.show()

    def _card_frame(self, title: str) -> tuple[QFrame, FieldColumn]:
        card = QFrame(self._stage)
        card.setObjectName("SelectionCard")
        outer = QVBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setObjectName("CardHeader")
        header.setFixedHeight(26)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 8, 0)
        caption = QLabel(title.upper())
        caption.setObjectName("CardTitle")
        font = ui_font(8)
        font.setBold(True)
        caption.setFont(font)
        close = QPushButton("×")
        close.setObjectName("CardClose")
        close.setCursor(Qt.CursorShape.PointingHandCursor)
        close.clicked.connect(self._close_card)
        header_layout.addWidget(caption)
        header_layout.addStretch(1)
        header_layout.addWidget(close)

        body_holder = QFrame()
        body_holder.setObjectName("CardBody")
        holder_layout = QVBoxLayout(body_holder)
        holder_layout.setContentsMargins(13, 12, 13, 13)
        body = FieldColumn()
        holder_layout.addWidget(body)

        outer.addWidget(header)
        outer.addWidget(body_holder)
        self._card = card
        return card, body

    def _place_card(self) -> None:
        if self._card is None:
            return
        self._card.move(
            max(CARD_MARGIN, self._stage.width() - self._card.width() - CARD_MARGIN), CARD_MARGIN
        )
        # The drawing gives up this width rather than being covered by it. Only the planform can
        # do that; the 3D scene keeps its full viewport and the card floats over a corner of it.
        self.view.set_right_inset(self._card.width() + 2 * CARD_MARGIN)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_card()

    # ── the cards ──────────────────────────────────────────────────────────────────────

    def _wing_card(self, body: FieldColumn) -> None:
        assert self._file is not None
        planform = authoring.wing_planform(self._file.airframe)
        if planform is None:
            body.add(QLabel("This wing has more than two stations.\nEdit it station by station."))
            return

        span = body.add(Field("Span", "m", minimum=1e-3))
        aspect = body.add(Field("Aspect ratio", "", decimals=2, minimum=1e-3))
        taper = body.add(Field("Taper", "", decimals=3, minimum=1e-3, maximum=1.0))
        mode = body.add(ChoiceField("Sweep mode", SWEEP_MODES.keys()))
        sweep = body.add(Field("LE sweep", "°", decimals=2, minimum=-60.0, maximum=75.0))
        dihedral = body.add(Field("Dihedral", "°", decimals=1, minimum=-30.0, maximum=90.0))
        twist = body.add(Field("Tip twist", "°", decimals=1, minimum=-15.0, maximum=15.0))
        # Root height above the aircraft datum. High, mid and low wing are this number, and the
        # tails carry their own, so the vertical arrangement is editable rather than implied by
        # the fuselage style.
        height = body.add(Field("Height", "m", decimals=3, minimum=-2.0, maximum=2.0))
        root_foil = body.add(TextField("Root airfoil"))
        tip_foil = body.add(TextField("Tip airfoil"))

        body.add(GroupLabel("Worked out"))
        area_out = body.add(Readout("Area", "m²"))
        root_out = body.add(Readout("Root chord", "m"))
        tip_out = body.add(Readout("Tip chord", "m"))
        mac_out = body.add(Readout("MAC", "m"))

        span.set_value(planform.span)
        aspect.set_value(planform.resolved_aspect_ratio())
        taper.set_value(planform.taper)
        mode.set_value(planform.sweep_mode)
        sweep.set_value(planform.resolved_sweep_deg())
        dihedral.set_value(planform.dihedral_deg)
        twist.set_value(planform.tip_twist_deg)
        height.set_value(authoring.surface_height(self._file.airframe, "wing"))
        root_foil.set_value(planform.airfoil_root)
        tip_foil.set_value(planform.airfoil_tip)

        def refresh_derived() -> None:
            facts = authoring.summary(self._file.airframe)
            area_out.set_value(facts["wing_area_m2"])
            root_out.set_value(facts["root_chord_m"])
            tip_out.set_value(facts["tip_chord_m"])
            mac_out.set_value(facts["mac_m"])
            sweep.set_value(
                (authoring.wing_planform(self._file.airframe) or planform).resolved_sweep_deg()
            )

        def apply() -> None:
            assert self._file is not None
            current = authoring.wing_planform(self._file.airframe) or planform
            authoring.apply_wing(self._file.airframe, Planform(
                span=_or(span.value(), current.span),
                aspect_ratio=_or(aspect.value(), current.resolved_aspect_ratio()),
                taper=_or(taper.value(), current.taper),
                sweep_mode=mode.value(),
                sweep_deg=_or(sweep.value(), current.sweep_deg),
                dihedral_deg=_or(dihedral.value(), current.dihedral_deg),
                tip_twist_deg=_or(twist.value(), current.tip_twist_deg),
                airfoil_root=root_foil.value() or current.airfoil_root,
                airfoil_tip=tip_foil.value() or current.airfoil_tip,
            ))
            # After apply_wing, which carries origin[2] through untouched.
            authoring.set_surface_height(
                self._file.airframe, "wing",
                _or(height.value(), authoring.surface_height(self._file.airframe, "wing") or 0.0),
            )
            self._after_edit(refresh_derived)

        for widget in (span, aspect, taper, mode, sweep, dihedral, twist, height,
                       root_foil, tip_foil):
            widget.changed.connect(apply)
        refresh_derived()

    def _tail_card(self, body: FieldColumn, which: str) -> None:
        assert self._file is not None
        surface = getattr(self._file.airframe, which)
        vertical = which == "vtail"
        if surface is None:
            body.add(QLabel(f"This aircraft has no {'fin' if vertical else 'horizontal tail'}."))
            return
        sizing = authoring.infer_tail_sizing(self._file.airframe, surface, vertical=vertical)

        area = body.add(Field("Area", "m²", decimals=4, minimum=1e-5))
        aspect = body.add(Field("Aspect ratio", "", decimals=2, minimum=1e-3))
        arm = body.add(Field("Arm", "m", minimum=1e-3))
        # Root height above the aircraft datum, the same reference the wing uses - so a T-tail is
        # simply a tail whose height is above the fin's, rather than a configuration flag.
        height = body.add(Field("Height", "m", decimals=3, minimum=-2.0, maximum=2.0))
        taper = body.add(Field("Taper", "", decimals=3, minimum=1e-3, maximum=1.0))
        sweep = body.add(Field("Sweep", "°", decimals=1, minimum=-45.0, maximum=75.0))
        incidence = None
        if not vertical:
            incidence = body.add(Field("Incidence", "°", decimals=1, minimum=-10.0, maximum=10.0))
        foil = body.add(TextField("Airfoil"))

        remove = QPushButton("Remove component")
        remove.setProperty("chrome", "true")
        remove.setProperty("danger", "true")
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.clicked.connect(lambda: self._remove_optional_component(which))
        body.add(remove)

        body.add(GroupLabel("Worked out"))
        volume_out = body.add(Readout("Tail volume", "", decimals=4))
        span_out = body.add(Readout("Height" if vertical else "Span", "m"))
        root_out = body.add(Readout("Root chord", "m"))

        area.set_value(sizing.area_m2)
        aspect.set_value(sizing.aspect_ratio)
        arm.set_value(sizing.arm_m)
        height.set_value(sizing.z_m)
        taper.set_value(sizing.taper)
        sweep.set_value(sizing.sweep_deg)
        if incidence is not None:
            incidence.set_value(sizing.incidence_deg)
        foil.set_value(sizing.airfoil)

        def refresh_derived() -> None:
            assert self._file is not None
            current = getattr(self._file.airframe, which)
            facts = authoring.summary(self._file.airframe)
            volume_out.set_value(
                facts["vertical_tail_volume"] if vertical else facts["horizontal_tail_volume"]
            )
            multiplier = 1.0 if vertical else 2.0
            span_out.set_value(current.span() * multiplier)
            root_out.set_value(current.sections[0].chord)

        def apply() -> None:
            assert self._file is not None
            authoring.apply_tail(self._file.airframe, authoring.TailSizing(
                area_m2=_or(area.value(), sizing.area_m2),
                aspect_ratio=_or(aspect.value(), sizing.aspect_ratio),
                arm_m=_or(arm.value(), sizing.arm_m),
                taper=_or(taper.value(), sizing.taper),
                incidence_deg=_or(incidence.value() if incidence else None, sizing.incidence_deg),
                airfoil=foil.value() or sizing.airfoil,
                sweep_deg=_or(sweep.value(), sizing.sweep_deg),
                z_m=_or(height.value(), sizing.z_m),
            ), which)
            self._after_edit(refresh_derived)

        widgets = [area, aspect, arm, height, taper, sweep, foil] + (
            [incidence] if incidence else []
        )
        for widget in widgets:
            widget.changed.connect(apply)
        refresh_derived()

    def _fuselage_card(self, body: FieldColumn) -> None:
        assert self._file is not None
        style = body.add(ChoiceField("Style", FUSELAGE_STYLES))
        position = body.add(ChoiceField("Wing position", WING_POSITIONS))
        scale = body.add(Field("Scale", "", decimals=2, minimum=0.1, maximum=5.0))

        remove = QPushButton("Remove component")
        remove.setProperty("chrome", "true")
        remove.setProperty("danger", "true")
        remove.setCursor(Qt.CursorShape.PointingHandCursor)
        remove.clicked.connect(lambda: self._remove_optional_component("fuselage"))
        body.add(remove)

        body.add(GroupLabel("Worked out"))
        length_out = body.add(Readout("Length", "m", decimals=3))
        bodies_out = body.add(Readout("Bodies", "", decimals=0))

        current_style = "conventional" if self._file.airframe.fuselages else "none"
        style.set_value(current_style)
        position.set_value("mid")
        scale.set_value(1.0)

        def refresh_derived() -> None:
            assert self._file is not None
            fuselages = self._file.airframe.fuselages
            length_out.set_value(max((f.length() for f in fuselages), default=None))
            bodies_out.set_value(float(len(fuselages)))

        def apply() -> None:
            assert self._file is not None
            authoring.apply_fuselage(
                self._file.airframe, style.value(), position.value(), _or(scale.value(), 1.0)
            )
            self._after_edit(refresh_derived)

        for widget in (style, position, scale):
            widget.changed.connect(apply)
        refresh_derived()

    def _mass_card(self, body: FieldColumn) -> None:
        """What it weighs and how the mass is distributed.

        **Stated, not sized.** This tool does not work mass out from a brief — that was the
        previous generation and it is deliberately gone. What it needs is the same kind of fact as
        the airspeed: a number about your aircraft that you supply.

        Inertias are the awkward part, because almost nobody knows their Izz. The estimate button
        derives all three from radii of gyration as fractions of span and length, which are
        well-bounded for conventional configurations — and marks the result as estimated, because a
        mode frequency scales with the square root of an inertia and a guessed Iyy is a guessed
        short period.
        """
        assert self._file is not None
        current = self._file.mass

        mass = body.add(Field("Mass", "kg", decimals=3, minimum=1e-3))
        body.add(GroupLabel("Inertia about the CG"))
        ixx = body.add(Field("Ixx  roll", "kg m²", decimals=4, minimum=0.0))
        iyy = body.add(Field("Iyy  pitch", "kg m²", decimals=4, minimum=0.0))
        izz = body.add(Field("Izz  yaw", "kg m²", decimals=4, minimum=0.0))
        ixz = body.add(Field("Ixz  coupling", "kg m²", decimals=4))

        body.add(GroupLabel("Worked out"))
        state = body.add(Readout("Complete", "", decimals=0))
        note = QLabel()
        note.setProperty("role", "derived-label")
        note.setWordWrap(True)
        note.setFont(ui_font(8))
        body.add(note)

        estimate = QPushButton("Estimate inertia from size")
        estimate.setProperty("chrome", "true")
        estimate.setCursor(Qt.CursorShape.PointingHandCursor)
        body.add(estimate)

        mass.set_value(current.mass_kg or None)
        ixx.set_value(current.ixx or None)
        iyy.set_value(current.iyy or None)
        izz.set_value(current.izz or None)
        ixz.set_value(current.ixz)

        def refresh_derived() -> None:
            assert self._file is not None
            properties = self._file.mass
            state.set_value(1.0 if properties.complete else 0.0)
            if not properties.complete:
                note.setText(f"Not ready for dynamics: {properties.missing()}.")
            elif properties.estimated:
                note.setText(
                    "Inertias are estimated from radii of gyration. Every mode frequency scales "
                    "with the square root of an inertia, so treat the numbers as indicative."
                )
            else:
                note.setText("Entered directly.")

        def apply() -> None:
            assert self._file is not None
            self._file.mass = MassProperties(
                mass_kg=_or(mass.value(), 0.0),
                ixx=_or(ixx.value(), 0.0), iyy=_or(iyy.value(), 0.0), izz=_or(izz.value(), 0.0),
                ixz=_or(ixz.value(), 0.0),
                estimated=False,
            )
            self._after_edit(refresh_derived)

        def estimate_inertia() -> None:
            assert self._file is not None
            entered = _or(mass.value(), self._file.mass.mass_kg)
            if entered <= 0:
                note.setText("Enter a mass first — the inertias are all proportional to it.")
                return
            facts = authoring.summary(self._file.airframe)
            length = max(
                (f.length() for f in self._file.airframe.fuselages),
                default=facts["mac_m"] * 4.0,
            )
            self._file.mass = MassProperties.from_radii_of_gyration(
                mass_kg=entered, span_m=facts["wing_span_m"], length_m=length,
            )
            properties = self._file.mass
            ixx.set_value(properties.ixx)
            iyy.set_value(properties.iyy)
            izz.set_value(properties.izz)
            ixz.set_value(properties.ixz)
            self._after_edit(refresh_derived)

        estimate.clicked.connect(estimate_inertia)
        for widget in (mass, ixx, iyy, izz, ixz):
            widget.changed.connect(apply)
        refresh_derived()

    def _condition_card(self, body: FieldColumn) -> None:
        assert self._file is not None
        condition = self._file.condition

        speed = body.add(Field("Speed", "m/s", decimals=1, minimum=1.0))
        altitude = body.add(Field("Altitude", "m", decimals=0, minimum=0.0))
        alpha_lo = body.add(Field("Alpha from", "°", decimals=1, minimum=-30.0, maximum=30.0))
        alpha_hi = body.add(Field("Alpha to", "°", decimals=1, minimum=-30.0, maximum=45.0))
        fidelity = body.add(ChoiceField("Fidelity", [f.value for f in Fidelity]))
        scope = body.add(ChoiceField("Scope", SCOPES))
        cg_x = body.add(Field("CG x", "m", minimum=-5.0, maximum=10.0))

        body.add(GroupLabel("Worked out"))
        points_out = body.add(Readout("Alpha points", "", decimals=0))
        re_out = body.add(Readout("Reynolds", "", decimals=0))
        mach_out = body.add(Readout("Mach", "", decimals=4))
        cg_out = body.add(Readout("CG", "% MAC", decimals=1))

        speed.set_value(condition.velocity_mps)
        altitude.set_value(condition.altitude_m)
        alpha_lo.set_value(condition.alpha_start_deg)
        alpha_hi.set_value(condition.alpha_end_deg)
        fidelity.set_value(condition.fidelity.value)
        scope.set_value(condition.scope)
        cg_x.set_value(condition.xyz_ref[0])

        def refresh_derived() -> None:
            assert self._file is not None
            current = self._file.condition
            facts = authoring.summary(self._file.airframe)
            mac, mac_le = facts["mac_m"] or 0.0, facts["mac_le_x_m"] or 0.0
            points_out.set_value(float(len(current.alphas())))
            re_out.set_value(current.atmosphere.reynolds(current.velocity_mps, mac))
            mach_out.set_value(current.mach())
            cg_out.set_value(
                ((current.xyz_ref[0] - mac_le) / mac * 100.0) if mac > 0 else None
            )

        def apply() -> None:
            assert self._file is not None
            current = self._file.condition
            self._file.condition = OperatingCondition(
                velocity_mps=_or(speed.value(), current.velocity_mps),
                altitude_m=_or(altitude.value(), current.altitude_m),
                alpha_start_deg=_or(alpha_lo.value(), current.alpha_start_deg),
                alpha_end_deg=_or(alpha_hi.value(), current.alpha_end_deg),
                alpha_points=current.alpha_points,
                beta_start_deg=current.beta_start_deg,
                beta_end_deg=current.beta_end_deg,
                beta_points=current.beta_points,
                fidelity=Fidelity(fidelity.value()),
                scope=scope.value(),
                thick=current.thick,
                xyz_ref=(_or(cg_x.value(), current.xyz_ref[0]), *current.xyz_ref[1:]),
                sref_m2=current.sref_m2, bref_m=current.bref_m, cref_m=current.cref_m,
                control_deflections=dict(current.control_deflections),
            )
            self._after_edit(refresh_derived)

        for widget in (speed, altitude, alpha_lo, alpha_hi, fidelity, scope, cg_x):
            widget.changed.connect(apply)
        refresh_derived()

    # ── shared ─────────────────────────────────────────────────────────────────────────

    def _after_edit(self, refresh_derived: Callable[[], None]) -> None:
        """Redraw, recompute the readouts, and tell the window the document is dirty.

        The card itself is not rebuilt — that would destroy the widget the user is still tabbing
        out of, and move focus somewhere unexpected mid-edit.
        """
        assert self._file is not None
        self.view.set_airframe(self._file.airframe)
        self.view.set_selection(self._selection)
        self._refresh_viewport()
        self._refresh_readout()
        refresh_derived()
        self.changed.emit()


def _or(value: float | None, fallback: float) -> float:
    """A blank box means "leave it alone", never zero."""
    return fallback if value is None else value
