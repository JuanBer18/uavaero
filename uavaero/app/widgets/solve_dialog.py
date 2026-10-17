"""One dialog for every run: what the solver is about to be asked, before it is asked.

Built from `docs/mockups/solve_dialog.html`, which is the specification.

Three problems it exists to fix, all of them observed rather than imagined.

**Most of the inputs were unreachable.** `OperatingCondition` carries around fifteen fields and the
Condition card exposed six. `thick` — the choice between a vortex-lattice treatment and a panel one,
and the only treatment that represents a fuselage at all — reached no control anywhere, so the more
accurate half of the solver could not be selected from the application that owns it.

**The stability run silently reused the sweep's starting angle.** `solvers/stability.py` takes
`alpha_start_deg` as its evaluation point, and that field is also the bottom of the alpha sweep. So
a sweep over −4…12 measured its derivatives at −4, which on a real aircraft is nowhere near trim,
and the modes computed from them described a different aeroplane. Here the two are separate fields
and the dialog says how far from trim the chosen point is.

**A run gave no idea what it would cost.** Seventeen angles at HIGH fidelity is about two minutes;
the same sweep at QUICK is seconds. Both looked identical from the outside — a window that stopped
repainting — so the footer states the price before the button is pressed.

The dialog only ever *returns* a condition. It does not start anything and it does not write to the
`AirframeFile`: the caller decides what to run and what to keep, which is what lets the stability
point be chosen for one run without overwriting the sweep range the file was authored with.
"""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import PALETTE, ui_font
from uavaero.app.widgets.fields import ChoiceField, Field, FieldColumn, GroupLabel, Readout
from uavaero.geometry import authoring
from uavaero.schema.condition import SCOPES, SECTION_SOURCES, Fidelity, OperatingCondition
from uavaero.solvers.vspaero import missing_for_scope

GRAVITY = 9.80665

#: The run kinds, in the order they appear. The value is what `configure` returns.
ALPHA, SPEED, STABILITY, CG, DYNAMICS = "alpha", "speed", "stability", "cg", "dynamics"

#: Surface treatment, spelled out. `thick` is a bool on the condition, but "True" is not a thing
#: anyone recognises and the difference between these two is the difference between having a
#: fuselage in the solve and not.
TREATMENTS = {
    "thin - vortex lattice": False,
    "thick - panels (has a fuselage)": True,
}


def _banner(lead: str, text: str, tone: str = "warn") -> QFrame:
    """A line the reader has to see before pressing Run, styled like the mockup's."""
    frame = QFrame()
    frame.setStyleSheet(
        f"background: {PALETTE['accent_dim']};"
        f" border-left: 3px solid {PALETTE[tone]};"
    )
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(11, 8, 11, 8)
    layout.setSpacing(8)
    head = QLabel(lead)
    head.setFont(ui_font(8))
    head.setStyleSheet(f"color: {PALETTE[tone]}; font-weight: 700;")
    head.setAlignment(Qt.AlignmentFlag.AlignTop)
    body = QLabel(text)
    body.setFont(ui_font(8))
    body.setWordWrap(True)
    body.setStyleSheet(f"color: {PALETTE['ink_2']};")
    layout.addWidget(head)
    layout.addWidget(body, 1)
    return frame


class SolveDialog(QDialog):
    """Configure one run. `configure()` is the whole API."""

    def __init__(
        self, file, parent: QWidget | None = None, start_on: str = ALPHA,
        derivatives=None,
    ) -> None:
        super().__init__(parent)
        self._file = file
        # The last stability solve, which the CG sweep is arithmetic on. It lives on the
        # window rather than the file - a derivative set belongs to a run, not to an aircraft.
        self._derivatives = derivatives
        self._mode = ALPHA
        self.setWindowTitle(f"Solve - {file.name}")
        self.setMinimumWidth(520)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._alpha_tab(), "Alpha sweep")
        self._tabs.addTab(self._speed_tab(), "Speed sweep")
        self._tabs.addTab(self._stability_tab(), "Stability")
        self._tabs.addTab(self._cg_tab(), "CG sweep")
        self._tabs.addTab(self._dynamics_tab(), "Dynamics")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self._tabs, 1)

        self._cost = QLabel("")
        self._cost.setFont(ui_font(8))
        self._cost.setStyleSheet(f"color: {PALETTE['ink_2']};")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok
        )
        self._run = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._run.setText("Run")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        footer = QFrame()
        footer.setStyleSheet(f"background: {PALETTE['inset']};")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 9, 14, 9)
        footer_layout.addWidget(self._cost, 1)
        footer_layout.addWidget(buttons)
        root.addWidget(footer)

        order = [ALPHA, SPEED, STABILITY, CG, DYNAMICS]
        self._tabs.setCurrentIndex(order.index(start_on) if start_on in order else 0)
        self._on_tab_changed(self._tabs.currentIndex())

    # ── the tabs ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _page() -> tuple[QWidget, FieldColumn]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 14, 16, 14)
        body = FieldColumn()
        layout.addWidget(body)
        layout.addStretch(1)
        return page, body

    def _flow_fields(self, body: FieldColumn) -> tuple[Field, Field]:
        condition = self._file.condition
        body.add(GroupLabel("Flow"))
        speed = body.add(Field("Speed", "m/s", decimals=3, minimum=0.1, maximum=200.0))
        altitude = body.add(Field("Altitude", "m", decimals=1, minimum=-500.0, maximum=20000.0))
        speed.set_value(condition.velocity_mps)
        altitude.set_value(condition.altitude_m)
        return speed, altitude

    def _alpha_tab(self) -> QWidget:
        page, body = self._page()
        condition = self._file.condition
        self._a_speed, self._a_altitude = self._flow_fields(body)

        body.add(GroupLabel("What goes into the mesh"))
        self._a_scope = body.add(ChoiceField("Scope", SCOPES))
        self._a_treatment = body.add(ChoiceField("Surface treatment", TREATMENTS.keys()))
        self._a_scope.set_value(condition.scope)
        self._a_treatment.set_value(self._treatment_name(condition.thick))

        body.add(GroupLabel("Angles"))
        self._a_lo = body.add(Field("Alpha from", "°", decimals=1, minimum=-30.0, maximum=30.0))
        self._a_hi = body.add(Field("Alpha to", "°", decimals=1, minimum=-30.0, maximum=45.0))
        self._a_points = body.add(Field("Alpha points", "", decimals=0, minimum=2, maximum=80))
        self._a_beta = body.add(Field("Beta", "°", decimals=1, minimum=-30.0, maximum=30.0))
        self._a_lo.set_value(condition.alpha_start_deg)
        self._a_hi.set_value(condition.alpha_end_deg)
        if condition.alpha_points is not None:
            self._a_points.set_value(float(condition.alpha_points))
        self._a_beta.set_value(condition.beta_start_deg)

        body.add(GroupLabel("Fidelity and sources"))
        self._a_fidelity = body.add(ChoiceField("Fidelity", [f.value for f in Fidelity]))
        self._a_section = body.add(ChoiceField("Section source", SECTION_SOURCES))
        self._a_fidelity.set_value(condition.fidelity.value)
        self._a_section.set_value(condition.section_source)

        self._a_blocked = _banner("Cannot run.", "")
        self._a_blocked.hide()
        body.add(self._a_blocked)

        for widget in (self._a_speed, self._a_lo, self._a_hi, self._a_points,
                       self._a_fidelity, self._a_scope, self._a_treatment):
            widget.changed.connect(self._refresh)
        return page

    def _speed_tab(self) -> QWidget:
        page, body = self._page()
        condition = self._file.condition

        body.add(GroupLabel("Speeds to trim at"))
        self._s_lo = body.add(Field("From", "m/s", decimals=1, minimum=0.5, maximum=200.0))
        self._s_hi = body.add(Field("To", "m/s", decimals=1, minimum=0.5, maximum=200.0))
        self._s_step = body.add(Field("Step", "m/s", decimals=1, minimum=0.1, maximum=20.0))
        self._s_lo.set_value(max(0.5, condition.velocity_mps * 0.5))
        self._s_hi.set_value(condition.velocity_mps * 1.6)
        self._s_step.set_value(1.0)

        body.add(GroupLabel("Reads"))
        mass_out = body.add(Readout("Stated mass", "kg", decimals=3))
        mass_out.set_value(self._file.mass.mass_kg or None)

        body.add(_banner(
            "One solve.",
            "The inviscid solution depends on lift coefficient, not on Reynolds number, so the "
            "alpha sweep already on file covers every speed. Only the viscous half is recomputed "
            "at each speed's own Reynolds.",
            tone="good",
        ))

        self._s_blocked = _banner("Cannot run.", "")
        self._s_blocked.hide()
        body.add(self._s_blocked)

        for widget in (self._s_lo, self._s_hi, self._s_step):
            widget.changed.connect(self._refresh)
        return page

    def _stability_tab(self) -> QWidget:
        page, body = self._page()
        condition = self._file.condition
        self._t_speed, self._t_altitude = self._flow_fields(body)

        body.add(GroupLabel("Where the derivatives are measured"))
        self._t_alpha = body.add(Field("Alpha", "°", decimals=2, minimum=-30.0, maximum=30.0))
        self._t_beta = body.add(Field("Beta", "°", decimals=2, minimum=-30.0, maximum=30.0))
        self._t_alpha.set_value(condition.alpha_start_deg)
        self._t_beta.set_value(condition.beta_start_deg)

        self._t_trim = _banner("Off trim.", "")
        self._t_trim.hide()
        body.add(self._t_trim)

        body.add(GroupLabel("Centre of gravity - moments are taken about this point"))
        self._t_cg_x = body.add(Field("CG x", "m", decimals=4, minimum=-10.0, maximum=10.0))
        self._t_cg_z = body.add(Field("CG z", "m", decimals=4, minimum=-10.0, maximum=10.0))
        self._t_cg_x.set_value(condition.xyz_ref[0])
        self._t_cg_z.set_value(condition.xyz_ref[2])
        self._t_margin = body.add(Readout("CG, percent MAC", "%", decimals=1))

        self._t_cg = _banner("Check the CG.", "")
        self._t_cg.hide()
        body.add(self._t_cg)

        body.add(GroupLabel("Method"))
        self._t_treatment = body.add(ChoiceField("Surface treatment", TREATMENTS.keys()))
        self._t_treatment.set_value(self._treatment_name(condition.thick))

        for widget in (self._t_speed, self._t_alpha, self._t_cg_x, self._t_cg_z):
            widget.changed.connect(self._refresh)
        return page

    def _cg_tab(self) -> QWidget:
        page, body = self._page()
        facts = authoring.summary(self._file.airframe)
        mac = facts["mac_m"] or 0.25
        mac_le = facts["mac_le_x_m"] or 0.0

        body.add(GroupLabel("CG positions to try"))
        self._g_lo = body.add(Field("From", "m", decimals=4, minimum=-5.0, maximum=5.0))
        self._g_hi = body.add(Field("To", "m", decimals=4, minimum=-5.0, maximum=5.0))
        self._g_step = body.add(Field("Step", "m", decimals=4, minimum=1e-4, maximum=1.0))
        # Default to the chord itself: the interesting range is always within a MAC of the
        # leading edge, and quoting it in metres keeps it in the same units as the geometry.
        self._g_lo.set_value(mac_le)
        self._g_hi.set_value(mac_le + mac)
        self._g_step.set_value(round(mac / 20.0, 4))

        body.add(GroupLabel("Reads"))
        body.add(Readout("MAC", "m", decimals=4)).set_value(mac)
        body.add(Readout("MAC leading edge, x", "m", decimals=4)).set_value(mac_le)

        body.add(_banner(
            "No solver.",
            "The neutral point is a property of the shape, not of where the mass sits, so one "
            "stability run answers the whole curve. Run Stability first if you have not.",
            tone="good",
        ))
        self._g_blocked = _banner("Cannot run.", "")
        self._g_blocked.hide()
        body.add(self._g_blocked)

        for widget in (self._g_lo, self._g_hi, self._g_step):
            widget.changed.connect(self._refresh)
        return page

    def _dynamics_tab(self) -> QWidget:
        page, body = self._page()
        body.add(GroupLabel("Built from the last stability run"))
        self._d_note = QLabel("")
        self._d_note.setFont(ui_font(8))
        self._d_note.setWordWrap(True)
        self._d_note.setStyleSheet(f"color: {PALETTE['ink_2']};")
        body.add(self._d_note)

        body.add(GroupLabel("Trim"))
        self._d_speed = body.add(Field("Speed", "m/s", decimals=3, minimum=0.1, maximum=200.0))
        self._d_speed.set_value(self._file.condition.velocity_mps)

        if self._file.mass.estimated:
            body.add(_banner(
                "Estimated inertia.",
                "Ixx, Iyy and Izz came from radii of gyration rather than from weighing. Every "
                "frequency scales with the square root of an inertia, so treat the modes as "
                "indicative.",
            ))
        self._d_speed.changed.connect(self._refresh)
        return page

    # ── live feedback ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _treatment_name(thick: bool) -> str:
        for name, value in TREATMENTS.items():
            if value == thick:
                return name
        return next(iter(TREATMENTS))

    def _trim_alpha(self, speed: float) -> tuple[float, float] | None:
        """(required CL, trim alpha) at this speed, from the last solve's lift curve.

        Needs a solve, because the lift-curve slope is a measured quantity. Returns None before
        one exists rather than assuming 2*pi - the whole point of this warning is that it is
        grounded in this aircraft.
        """
        solve = self._file.latest()
        mass = self._file.mass.mass_kg
        if solve is None or mass <= 0 or speed <= 0:
            return None
        result = solve.result
        sref = result.reference.sref_m2 or self._file.airframe.wing.area()
        density = self._file.condition.atmosphere.density
        if sref <= 0:
            return None
        cl_required = mass * GRAVITY / (0.5 * density * speed * speed * sref)
        slope = result.cl_alpha_per_rad
        if not slope:
            return None
        import numpy as np

        alphas = np.asarray(result.polar.alpha_deg, dtype=float)
        cls = np.asarray(result.polar.CL, dtype=float)
        if cls.size < 2:
            return None
        return cl_required, float(np.interp(cl_required, cls, alphas))

    def _refresh(self) -> None:
        """Recompute the warnings and the cost line for the tab in view."""
        airframe = self._file.airframe
        mode = self._mode

        if mode == ALPHA:
            blocked = missing_for_scope(airframe, self._a_scope.value())
            self._a_blocked.hide() if not blocked else self._show_banner(self._a_blocked, blocked)
            points = int(self._a_points.value() or 0) or self._fidelity_points()
            self._cost.setText(
                f"{points} angles · one solve · {self._estimate(points)}"
            )
            self._run.setEnabled(not blocked)
            return

        if mode == SPEED:
            solve = self._file.latest()
            problem = ""
            if solve is None:
                problem = "run an alpha sweep first - the speed sweep is built from its polar"
            elif self._file.mass.mass_kg <= 0:
                problem = "a speed sweep needs a stated mass; set it on the Mass card"
            self._s_blocked.hide() if not problem else self._show_banner(self._s_blocked, problem)
            self._cost.setText(f"{len(self._speeds())} speeds · no solver · instant")
            self._run.setEnabled(not problem)
            return

        if mode == CG:
            problem = "" if self._derivatives is not None else (
                "run Stability first - the CG sweep is arithmetic on its neutral point"
            )
            self._g_blocked.hide() if not problem else self._show_banner(self._g_blocked, problem)
            self._cost.setText(f"{len(self.cg_positions())} positions · no solver · instant")
            self._run.setEnabled(not problem)
            return

        if mode == STABILITY:
            self._update_stability_warnings()
            self._cost.setText("1 perturbed solve + AVL cross-check · about 25 s")
            self._run.setEnabled(True)
            return

        solve = self._file.latest()
        self._d_note.setText(
            "No stability run yet - run the Stability tab first."
            if solve is None else
            "Uses the most recent stability derivatives, this aircraft's stated mass, and the "
            "trim point at the speed below."
        )
        self._cost.setText("No solver · arithmetic on the last stability run · instant")
        self._run.setEnabled(True)

    def _show_banner(self, banner: QFrame, text: str) -> None:
        body = banner.findChildren(QLabel)[1]
        body.setText(text)
        banner.show()

    def _update_stability_warnings(self) -> None:
        speed = self._t_speed.value() or self._file.condition.velocity_mps
        alpha = self._t_alpha.value()
        trim = self._trim_alpha(speed)
        if trim is None or alpha is None:
            self._t_trim.hide()
        else:
            _, trim_alpha = trim
            gap = abs(trim_alpha - alpha)
            if gap < 1.0:
                self._t_trim.hide()
            else:
                self._show_banner(self._t_trim, (
                    f"Level flight at {speed:g} m/s and {self._file.mass.mass_kg:g} kg needs "
                    f"{trim_alpha:+.1f}°. Derivatives taken {gap:.1f}° away describe a different "
                    f"aircraft, and the modes inherit that."
                ))

        facts = authoring.summary(self._file.airframe)
        mac, mac_le = facts["mac_m"] or 0.0, facts["mac_le_x_m"] or 0.0
        cg_x = self._t_cg_x.value()
        if mac > 0 and cg_x is not None:
            percent = (cg_x - mac_le) / mac * 100.0
            self._t_margin.set_value(percent)
            if not -5.0 <= percent <= 60.0:
                self._show_banner(self._t_cg, (
                    f"The reference sits at {percent:.0f}% of MAC. A conventional aircraft is "
                    f"nearer 25%, and every longitudinal number depends on this."
                ))
            else:
                self._t_cg.hide()

    def _fidelity_points(self) -> int:
        try:
            return len(replace(
                self._file.condition, fidelity=Fidelity(self._a_fidelity.value())
            ).alphas())
        except Exception:  # noqa: BLE001 - a half-typed field is not an error worth showing
            return 0

    def _estimate(self, points: int) -> str:
        """A rough wall-clock, from the fidelity rung and the angle count. Deliberately coarse:
        the purpose is telling seconds from minutes, not predicting to the second."""
        per_angle = {"quick": 0.9, "standard": 3.0, "high": 7.0}.get(self._a_fidelity.value(), 3.0)
        seconds = 12.0 + points * per_angle * (2.5 if TREATMENTS[self._a_treatment.value()] else 1.0)
        return f"about {seconds:.0f} s" if seconds < 90 else f"about {seconds / 60:.0f} min"

    def _speeds(self) -> list[float]:
        low = self._s_lo.value() or 0.0
        high = self._s_hi.value() or 0.0
        step = self._s_step.value() or 1.0
        if high < low or step <= 0:
            return []
        count = int((high - low) / step) + 1
        return [low + i * step for i in range(max(count, 0))]

    def _on_tab_changed(self, index: int) -> None:
        self._mode = [ALPHA, SPEED, STABILITY, CG, DYNAMICS][index]
        self._refresh()

    # ── the answer ────────────────────────────────────────────────────────────────────────

    def condition(self) -> OperatingCondition:
        """The condition for the run that was configured, built fresh off the file's own.

        Fields the chosen tab does not offer are carried through untouched, so running a stability
        point never silently rewrites the sweep range or the reference overrides.
        """
        base = self._file.condition
        if self._mode in (ALPHA, SPEED):
            points = int(self._a_points.value()) if self._a_points.value() else None
            beta = self._a_beta.value()
            return replace(
                base,
                velocity_mps=self._a_speed.value() or base.velocity_mps,
                altitude_m=self._a_altitude.value() if self._a_altitude.value() is not None
                else base.altitude_m,
                alpha_start_deg=self._a_lo.value() if self._a_lo.value() is not None
                else base.alpha_start_deg,
                alpha_end_deg=self._a_hi.value() if self._a_hi.value() is not None
                else base.alpha_end_deg,
                alpha_points=points,
                beta_start_deg=beta if beta is not None else base.beta_start_deg,
                beta_end_deg=beta if beta is not None else base.beta_end_deg,
                fidelity=Fidelity(self._a_fidelity.value()),
                scope=self._a_scope.value(),
                thick=TREATMENTS[self._a_treatment.value()],
                section_source=self._a_section.value(),
            )
        if self._mode == STABILITY:
            alpha = self._t_alpha.value()
            return replace(
                base,
                velocity_mps=self._t_speed.value() or base.velocity_mps,
                altitude_m=self._t_altitude.value() if self._t_altitude.value() is not None
                else base.altitude_m,
                # The stability solver reads its evaluation point from `alpha_start_deg`. Setting
                # it on a *copy* is what keeps the file's sweep range intact.
                alpha_start_deg=alpha if alpha is not None else base.alpha_start_deg,
                beta_start_deg=self._t_beta.value() or 0.0,
                xyz_ref=(
                    self._t_cg_x.value() if self._t_cg_x.value() is not None else base.xyz_ref[0],
                    base.xyz_ref[1],
                    self._t_cg_z.value() if self._t_cg_z.value() is not None else base.xyz_ref[2],
                ),
                thick=TREATMENTS[self._t_treatment.value()],
            )
        return replace(base, velocity_mps=self._d_speed.value() or base.velocity_mps)

    def speeds(self) -> list[float]:
        return self._speeds()

    def cg_positions(self) -> list[float]:
        low = self._g_lo.value()
        high = self._g_hi.value()
        step = self._g_step.value() or 0.01
        if low is None or high is None or high < low or step <= 0:
            return []
        count = int((high - low) / step) + 1
        return [low + i * step for i in range(max(count, 0))]

    def mode(self) -> str:
        return self._mode

    @staticmethod
    def configure(file, parent: QWidget | None = None, start_on: str = ALPHA, derivatives=None):
        """Show the dialog. Returns `(mode, condition, values)` or None if it was cancelled.

        `values` is the speed list for a speed sweep, the CG positions for a CG sweep, and
        empty otherwise.
        """
        dialog = SolveDialog(file, parent, start_on=start_on, derivatives=derivatives)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        values = dialog.speeds() if dialog.mode() == SPEED else (
            dialog.cg_positions() if dialog.mode() == CG else []
        )
        return dialog.mode(), dialog.condition(), values
