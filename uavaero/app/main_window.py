"""The window: one open airframe, one screen, and the file menu around it.

Deliberately thin. There is no module registry, no workflow gate and no navigation rail — the
airframe screen *is* the application in Phase 1, and the later screens (drag comparison, stability,
dynamics, optimiser) become siblings of it rather than steps in a chain you must walk in order.

The solve runs on a `QThread`, and that is the one threading exception in the whole tree: a
VSPAERO sweep is tens of seconds and a frozen window is indistinguishable from a crash. It is one
thread and two signals, not a job system.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, QThread, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
)

from uavaero.app.screens.airframe import AirframeScreen
from uavaero.app.screens.comparison import ComparisonScreen
from uavaero.app.screens.dynamics import DynamicsScreen
from uavaero.app.screens.optimise import OptimiseScreen
from uavaero.app.widgets.console import DETAIL, ERROR, GOOD, INFO, WARN, Console, ConsoleHandler
from uavaero.app.screens.stability import StabilityScreen
from uavaero.paths import APP_ROOT, RUNS
from uavaero.schema.airframe import ParametricAirframe
from uavaero.schema.condition import OperatingCondition
from uavaero.solvers.base import SolveFailed
from uavaero.solvers.vspaero import VSPAeroSolver
from uavaero.store import airframe_file as af
from uavaero.store.cache import CachedSolver, ResultCache


class _SolveWorker(QObject):
    """Runs one solve off the GUI thread. Never raises into Qt — a failure comes back as text."""

    finished = Signal(object, str)  # (AeroResult | None, error message)
    output = Signal(str)
    #: (fraction 0..1 or -1 for indeterminate, what it is doing)
    progress = Signal(float, str)

    def __init__(self, airframe: ParametricAirframe, condition: OperatingCondition) -> None:
        super().__init__()
        self._airframe = airframe
        self._condition = condition

    def _line(self, line: str) -> None:
        """Every solver line goes to the console, and the ones that mean something drive the bar.

        `parse_progress` reads VSPAERO's own wake-iteration and per-angle chatter. It was written
        in Phase 0 and wired to nothing until now, which is why a solve looked identical whether it
        was working or hung.
        """
        from uavaero.solvers.vspaero import parse_progress

        self.output.emit(line)
        step = parse_progress(line, len(self._condition.alphas()))
        if step is not None:
            self.progress.emit(step[0], step[1])

    def run(self) -> None:
        try:
            solver = CachedSolver(VSPAeroSolver(), ResultCache())
            result = solver.evaluate(
                self._airframe,
                self._condition,
                workdir=RUNS / (self._airframe.name or "airframe") / "vspaero",
                on_output=self._line,
            )
            self.finished.emit(result, "")
        except SolveFailed as exc:
            self.finished.emit(None, str(exc))
        except Exception as exc:  # noqa: BLE001 - anything else is still a message, not a crash
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class _StabilityWorker(QObject):
    """Runs one stability point off the GUI thread. Never raises into Qt."""

    finished = Signal(object, str)
    output = Signal(str)

    def __init__(self, airframe: ParametricAirframe, condition: OperatingCondition) -> None:
        super().__init__()
        self._airframe = airframe
        self._condition = condition

    def run(self) -> None:
        from uavaero.solvers import stability as stability_solver

        try:
            result = stability_solver.solve(
                self._airframe, self._condition, on_output=self.output.emit,
            )
            self.finished.emit(result, "")
        except SolveFailed as exc:
            self.finished.emit(None, str(exc))
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UAV Studio")
        self._file: af.AirframeFile | None = None
        self._derivatives = None  # the last stability solve, which dynamics is built from
        self._dirty = False
        self._thread: QThread | None = None
        self._worker: _SolveWorker | None = None

        self.screen = AirframeScreen()
        self.screen.changed.connect(self._on_changed)
        self.screen.notice.connect(self._on_screen_notice)
        self.screen.save_requested.connect(self.save)
        self.screen.solve_requested.connect(self.solve)
        self.screen.open_in_vsp_requested.connect(self.open_in_openvsp)

        self.comparison = ComparisonScreen()
        self.stability = StabilityScreen()
        self.dynamics = DynamicsScreen()
        self.dynamics.fly_mode_requested.connect(self._fly_mode)
        self.optimise = OptimiseScreen()
        self.optimise.notice.connect(self._on_screen_notice)

        # Two sibling screens rather than a workflow. Neither is a step you must walk through to
        # reach the other, which is the whole difference from the previous tool.
        self._screens = QStackedWidget()
        self._screens.addWidget(self.screen)
        self._screens.addWidget(self.comparison)
        self._screens.addWidget(self.stability)
        self._screens.addWidget(self.dynamics)
        self._screens.addWidget(self.optimise)
        self.setCentralWidget(self._screens)

        self.console = Console()
        self.console.open_folder_requested.connect(self._open_folder)
        self._console_dock = QDockWidget("Console", self)
        self._console_dock.setObjectName("ConsoleDock")
        self._console_dock.setWidget(self.console)
        self._console_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)
        self._console_dock.hide()

        self._progress = QProgressBar()
        self._progress.setMaximumWidth(240)
        self._progress.setTextVisible(True)
        self._progress.hide()
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Ready")

        # Anything under uavaero/ can report through the standard logging module without knowing
        # this window exists.
        self._log_handler = ConsoleHandler(self._log)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger("uavaero").addHandler(self._log_handler)
        logging.getLogger("uavaero").setLevel(logging.INFO)

        self._build_menu()
        self._restore_last_or_create()

    # ── menu ───────────────────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for text, shortcut, slot in (
            ("&New airframe…", QKeySequence.StandardKey.New, self.new_airframe),
            ("&Open…", QKeySequence.StandardKey.Open, self.open_airframe),
            ("&Save", QKeySequence.StandardKey.Save, self.save),
            ("Save &as…", QKeySequence.StandardKey.SaveAs, self.save_as),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        solve_menu = self.menuBar().addMenu("&Solve")
        run = QAction("&Run VSPAERO", self)
        run.setShortcut("Ctrl+R")
        run.triggered.connect(self.solve)
        solve_menu.addAction(run)
        derivatives = QAction("Stability &derivatives", self)
        derivatives.setShortcut("Ctrl+D")
        derivatives.triggered.connect(self.solve_stability)
        solve_menu.addAction(derivatives)
        modes = QAction("Dynamic &modes", self)
        modes.setShortcut("Ctrl+M")
        modes.triggered.connect(self.solve_dynamics)
        solve_menu.addAction(modes)
        vsp = QAction("Open in &OpenVSP", self)
        vsp.triggered.connect(self.open_in_openvsp)
        solve_menu.addAction(vsp)

        view_menu = self.menuBar().addMenu("&View")
        for label, index, shortcut in (
            ("&Airframe", 0, "Ctrl+1"), ("&Drag comparison", 1, "Ctrl+2"),
            ("&Stability", 2, "Ctrl+3"), ("D&ynamics", 3, "Ctrl+4"),
            ("&Optimise", 4, "Ctrl+5"),
        ):
            action = QAction(label, self)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _=False, i=index: self.show_screen(i))
            view_menu.addAction(action)
        view_menu.addSeparator()
        console = QAction("&Console", self)
        console.setShortcut("Ctrl+`")
        console.setCheckable(True)
        console.triggered.connect(self.toggle_console)
        view_menu.addAction(console)
        self._console_action = console
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("&Theme")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        from uavaero.app.style import current_theme

        selected_theme = current_theme()
        for label, theme in (("&Light", "light"), ("&Dark", "dark")):
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(theme == selected_theme)
            action.triggered.connect(
                lambda checked=False, value=theme: self._set_theme(value) if checked else None
            )
            theme_group.addAction(action)
            theme_menu.addAction(action)

    def toggle_console(self, visible: bool | None = None) -> None:
        show = (not self._console_dock.isVisible()) if visible is None else bool(visible)
        self._console_dock.setVisible(show)
        if hasattr(self, "_console_action"):
            self._console_action.setChecked(show)

    def _on_solver_output(self, line: str) -> None:
        """One line of solver output, on the GUI thread.

        **This must stay a bound method and never become a lambda.** A signal connected to a plain
        callable has no receiver QObject, so Qt cannot compare threads and resolves AutoConnection
        to a *direct* call — the slot then runs in the worker thread. That is what this was, and it
        meant `console.append` was mutating a QPlainTextEdit from a non-GUI thread ~3800 times per
        sweep. Qt widgets are not thread-safe: the window stopped repainting a second after Solve
        was pressed and the process died natively with no Python traceback. A bound method of this
        QObject gives Qt a receiver, the connection resolves to queued, and the line arrives here.
        """
        self._log(line, DETAIL)

    def _log(self, text: str, level: str = INFO) -> None:
        self.console.append(text, level)

    def _begin_run(self, title: str, directory: Path | None = None) -> None:
        """Open the console, name the run, and start the progress bar.

        The console opens itself on the first solve of a session rather than waiting to be found:
        the whole problem it fixes is not knowing whether anything is happening.
        """
        self.console.set_run_directory(directory)
        self.console.rule(title)
        self.console.set_status("running")
        if not self._console_dock.isVisible():
            self.toggle_console(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("starting…")
        self._progress.show()

    def _end_run(self, message: str, level: str = GOOD) -> None:
        self._log(message, level)
        self.console.set_status("idle")
        self._progress.hide()

    def _on_progress(self, fraction: float, what: str) -> None:
        if fraction < 0:
            self._progress.setRange(0, 0)  # indeterminate: wake iterations, count unknown
        else:
            self._progress.setRange(0, 100)
            self._progress.setValue(int(fraction * 100))
        self._progress.setFormat(what[:48])

    def _open_folder(self, directory) -> None:
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def show_screen(self, index: int) -> None:
        self._screens.setCurrentIndex(index)

    def _set_theme(self, theme: str) -> None:
        from uavaero.app.style import set_theme

        app = QApplication.instance()
        if app is None:
            return
        set_theme(app, theme)
        self.screen.view.update()
        # VTK paints the 3D background itself, so the stylesheet cannot reach it.
        self.screen.viewport.retheme()
        self.comparison.retheme()
        self.stability.retheme()
        self.dynamics.retheme()
        self.optimise.retheme()
        self.statusBar().showMessage(f"{theme.capitalize()} theme applied", 3000)

    # ── document ───────────────────────────────────────────────────────────────────────

    def _restore_last_or_create(self) -> None:
        settings = QSettings()
        last = settings.value("last_airframe", "")
        if last and Path(str(last)).is_file():
            try:
                self._set_file(af.load(Path(str(last))))
                return
            except af.AirframeFileError:
                pass  # a file that will not open is not a reason to refuse to start
        names = af.names()
        if names:
            try:
                self._set_file(af.load(names[0]))
                return
            except af.AirframeFileError:
                pass
        self._set_file(af.create("new airframe"))
        self._dirty = True

    def _set_file(self, file: af.AirframeFile) -> None:
        self._file = file
        self._dirty = False
        self.screen.set_file(file)
        self.comparison.set_file(file)
        self.stability.set_result(None)
        self.dynamics.set_report(None)
        self.optimise.set_file(file)
        self._derivatives = None
        self._update_title()
        if file.path is not None:
            QSettings().setValue("last_airframe", str(file.path))

    def _on_changed(self) -> None:
        self._dirty = True
        self._update_title()
        # A mode animation is a picture of one derivative set. Editing the aircraft invalidates it,
        # and leaving it flying would show a shape that no longer belongs to what is on screen.
        if self.screen.viewport.playing:
            self.screen.viewport.stop_mode()
        # The optimiser only offers the knobs that can move *this* aircraft, so removing a tail
        # has to reach the bounds panel. Otherwise it keeps showing rows for a surface that is
        # gone until the file is reopened, and those rows do nothing.
        self.optimise.set_file(self._file)

    def _fly_mode(self, shape) -> None:
        """Watch one dynamic mode on the 3D view.

        The Dynamics screen owns the physics and the Airframe screen owns the viewport, so the
        window is what joins them: it brings the 3D view forward and hands the shape over. Bound
        method, not a lambda - CLAUDE.md rule 6.
        """
        self.show_screen(0)
        self.screen.set_view_mode("3d")
        if not self.screen.viewport.play_mode(shape):
            self.statusBar().showMessage(
                "No 3D view on this machine - the mode table and root locus are unaffected", 6000,
            )
            return
        self._log(shape.caption, INFO)
        self.statusBar().showMessage(
            f"Flying the {shape.name}. Change the view or press Solve to stop.", 0,
        )

    def _on_screen_notice(self, message: str) -> None:
        """Something the geometry screen changed on the user's behalf, said out loud.

        Bound method rather than a lambda: same-thread here, but the rule in CLAUDE.md 6 is not
        worth carving an exception into, and this is one connect away from a worker.
        """
        self._log(message, WARN)
        self.statusBar().showMessage(message, 8000)

    def _update_title(self) -> None:
        if self._file is None:
            self.setWindowTitle("UAV Studio")
            return
        self.setWindowTitle(f"UAV Studio — {self._file.name}{' •' if self._dirty else ''}")

    def new_airframe(self) -> None:
        name, ok = QInputDialog.getText(self, "New airframe", "Name")
        if not ok or not name.strip():
            return
        self._set_file(af.create(name.strip()))
        self._dirty = True
        self._update_title()

    def open_airframe(self) -> None:
        directory = af.library_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open airframe", str(directory), "Airframe (*.yaml)"
        )
        if not path:
            return
        try:
            self._set_file(af.load(Path(path)))
        except af.AirframeFileError as exc:
            QMessageBox.warning(self, "Could not open", str(exc))

    def save(self) -> None:
        if self._file is None:
            return
        if self._file.path is None:
            self.save_as()
            return
        path = af.save(self._file)
        self._dirty = False
        self._update_title()
        QSettings().setValue("last_airframe", str(path))
        self.statusBar().showMessage(f"Saved {path.name}", 4000)

    def save_as(self) -> None:
        if self._file is None:
            return
        directory = af.library_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save airframe", str(af.path_for(self._file.name, directory)),
            "Airframe (*.yaml)",
        )
        if not path:
            return
        saved = af.save(self._file, Path(path))
        self._dirty = False
        self._update_title()
        QSettings().setValue("last_airframe", str(saved))
        self.statusBar().showMessage(f"Saved {saved.name}", 4000)

    # ── solving ────────────────────────────────────────────────────────────────────────

    def solve(self, start_on: str = "alpha") -> None:
        """Open the setup dialog, then run whatever was configured.

        Every run goes through here, because the choice of *which* run is one of the inputs. The
        dialog returns a condition and never writes it back to the file: a stability point chosen
        for one run must not silently become the aircraft's sweep range.
        """
        from uavaero.app.widgets.solve_dialog import SolveDialog

        if self._file is None or self._thread is not None:
            return
        chosen = SolveDialog.configure(
            self._file, self, start_on=start_on, derivatives=self._derivatives,
        )
        if chosen is None:
            return
        mode, condition, speeds = chosen

        if mode == "speed":
            self._run_speed_sweep(condition, speeds)
            return
        if mode == "stability":
            self._run_stability(condition)
            return
        if mode == "cg":
            self._run_cg_sweep(condition, speeds)
            return
        if mode == "dynamics":
            self.solve_dynamics()
            return

        # The alpha sweep is the one that changes the aircraft's own operating point, so it is the
        # one whose condition is kept.
        self._file.condition = condition
        self._dirty = True
        self._update_title()

        blocked = VSPAeroSolver().unavailable_for(self._file.airframe, condition)
        if blocked:
            QMessageBox.information(self, "Cannot solve yet", blocked.capitalize())
            return

        directory = RUNS / (self._file.airframe.name or "airframe") / "vspaero"
        self._begin_run(
            f"VSPAERO alpha sweep · {condition.scope} · "
            f"{condition.fidelity.value} · {condition.velocity_mps:g} m/s"
            f"{' · thick' if condition.thick else ''}",
            directory,
        )
        self.statusBar().showMessage("Solving…")
        self._thread = QThread(self)
        self._worker = _SolveWorker(self._file.airframe, condition)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.output.connect(self._on_solver_output)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_solved)
        self._thread.start()

    def solve_stability(self) -> None:
        """Open the dialog on the Stability tab. The run itself is `_run_stability`."""
        self.solve(start_on="stability")

    def _run_stability(self, condition) -> None:
        """Derivatives, neutral point and static margin at one operating point.

        A separate run from the sweep: it is a different analysis (one angle, perturbed about it)
        and it costs about a quarter as much, so tying it to the sweep would make the common case
        slower for no reason.

        `condition` is the dialog's, not the file's. The stability solver reads its evaluation
        point from `alpha_start_deg`, which is also the bottom of the sweep range — taking it from
        a copy is what lets the derivatives be measured at trim without moving the sweep.
        """
        if self._file is None or self._thread is not None:
            return
        blocked = VSPAeroSolver().unavailable_for(self._file.airframe, condition)
        if blocked:
            QMessageBox.information(self, "Cannot solve yet", blocked.capitalize())
            return

        self._begin_run(
            f"Stability derivatives · {condition.scope} · "
            f"{condition.alpha_start_deg:g}° · CG x {condition.xyz_ref[0]:g} m",
            RUNS / (self._file.airframe.name or "airframe") / "stability",
        )
        self.statusBar().showMessage("Solving stability derivatives...")
        self._thread = QThread(self)
        self._worker = _StabilityWorker(self._file.airframe, condition)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.output.connect(self._on_solver_output)
        self._worker.finished.connect(self._on_stability)
        self._thread.start()

    def _on_stability(self, result, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

        if result is None:
            headline = (error or "").splitlines()[0] if error else "the stability run failed"
            self._log(f"STABILITY RUN FAILED — {headline}", ERROR)
            self.console.block(error, ERROR)
            self._end_run("see the console for the captured solver output", ERROR)
            self.statusBar().showMessage("Stability run failed — see the console", 0)
            QMessageBox.warning(
                self, "Stability run failed",
                f"{headline}\n\nFull output is in the console (Ctrl+`).",
            )
            return

        self._derivatives = result
        self.stability.set_result(result)
        self.show_screen(2)
        for note in result.notes:
            self._log(f"note: {note}", WARN)
        margin = result.static_margin
        verdicts = result.verdicts()
        unstable = [name for name, ok in verdicts.items() if not ok]
        summary = "stable on all three axes" if not unstable else (
            "UNSTABLE: " + ", ".join(unstable)
        )
        self._end_run(
            f"static margin {margin * 100:.1f}% MAC — {summary} — "
            f"rates from {result.rate_source or 'nowhere'}"
        )
        self.statusBar().showMessage(
            f"Static margin {margin * 100:.1f}% MAC  -  {summary}  -  "
            f"rates from {result.rate_source or 'nowhere'}",
            0,
        )

    def _run_speed_sweep(self, condition, speeds: list[float]) -> None:
        """Level-flight drag and power across a speed range.

        No solver runs: the alpha sweep already on file carries the inviscid solution, which
        depends on lift coefficient and not on Reynolds number, so this is a trim solve and a
        strip re-integration per speed. Milliseconds, and therefore on the GUI thread.
        """
        from uavaero.aero.performance import speed_sweep
        from uavaero.solvers.sections.base import SectionLookup
        from uavaero.solvers.sections.neuralfoil import NeuralFoilSource

        assert self._file is not None
        solve = self._file.latest()
        if solve is None:
            QMessageBox.information(
                self, "Nothing to sweep",
                "Run an alpha sweep first - the speed sweep is built from its polar.",
            )
            return

        self._begin_run(
            f"Speed sweep · {len(speeds)} speeds · {self._file.mass.mass_kg:g} kg",
            RUNS / (self._file.airframe.name or "airframe") / "vspaero",
        )
        try:
            sweep = speed_sweep(
                solve.result, self._file.airframe, self._file.mass.mass_kg, speeds,
                lookup=SectionLookup(NeuralFoilSource()),
                altitude_m=condition.altitude_m,
            )
        except Exception as exc:  # noqa: BLE001 - a failed sweep is a message, not a crash
            self._log(f"SPEED SWEEP FAILED — {type(exc).__name__}: {exc}", ERROR)
            self._end_run("see the console", ERROR)
            return

        for note in sweep.notes:
            self._log(f"note: {note}", WARN)
        best_range, best_endurance = sweep.best_range(), sweep.best_endurance()
        if best_range is None:
            self._end_run("no speed in that range could be trimmed against the solved polar", WARN)
            return

        self._log(
            f"{'V':>6} {'CL':>7} {'alpha':>7} {'CD':>8} {'L/D':>6} {'power':>8}", INFO
        )
        for point in sweep.points:
            self._log(
                f"{point.velocity_mps:6.1f} {point.cl_required:7.3f} {point.alpha_deg:7.2f} "
                f"{point.cd_total:8.5f} {point.lift_to_drag:6.1f} "
                f"{point.power_watts(sweep.density, sweep.sref_m2):7.1f} W",
                DETAIL,
            )
        power = best_endurance.power_watts(sweep.density, sweep.sref_m2)
        self._end_run(
            f"best range {best_range.velocity_mps:g} m/s at L/D {best_range.lift_to_drag:.1f} · "
            f"best endurance {best_endurance.velocity_mps:g} m/s at {power:.1f} W"
        )
        self.statusBar().showMessage(
            f"Best range {best_range.velocity_mps:g} m/s (L/D {best_range.lift_to_drag:.1f})  ·  "
            f"best endurance {best_endurance.velocity_mps:g} m/s ({power:.1f} W)", 0,
        )

    def _run_cg_sweep(self, condition, positions: list[float]) -> None:
        """Static margin and the longitudinal modes against CG position.

        No solver runs. The neutral point is a property of the shape rather than of where the mass
        sits, so the stability run already on file answers the whole curve.
        """
        from uavaero.dynamics.cg import cg_sweep
        from uavaero.geometry import authoring

        assert self._file is not None
        if self._derivatives is None:
            QMessageBox.information(
                self, "Nothing to sweep",
                "Run Stability first - the CG sweep is arithmetic on its neutral point.",
            )
            return

        facts = authoring.summary(self._file.airframe)
        self._begin_run(
            f"CG sweep · {len(positions)} positions · "
            f"MAC {facts['mac_m'] or 0:.4f} m",
            RUNS / (self._file.airframe.name or "airframe") / "stability",
        )
        try:
            sweep = cg_sweep(
                self._derivatives, self._file.mass, condition, positions,
                mac_leading_edge_x_m=facts["mac_le_x_m"] or 0.0,
            )
        except Exception as exc:  # noqa: BLE001 - a failed sweep is a message, not a crash
            self._log(f"CG SWEEP FAILED — {type(exc).__name__}: {exc}", ERROR)
            self._end_run("see the console", ERROR)
            return

        for note in sweep.notes:
            self._log(f"note: {note}", WARN)
        if not sweep.points:
            self._end_run("nothing to report", WARN)
            return

        self._log(
            f"neutral point at x {sweep.neutral_point_x_m:.4f} m "
            f"({sweep.neutral_point_percent_mac():.0f}% MAC)", INFO,
        )
        self._log(f"{'x':>8} {'%MAC':>7} {'SM':>7} {'Cm_a':>7} {'sp wn':>7} {'sp z':>6}", INFO)
        for point in sweep.points:
            level = GOOD if point.conventional else (DETAIL if point.stable else WARN)
            self._log(
                f"{point.x_m:8.4f} {point.percent_mac:7.1f} {point.static_margin * 100:6.1f}% "
                f"{point.Cm_alpha:7.2f} {point.short_period_wn or 0:7.2f} "
                f"{point.short_period_damping or 0:6.2f}"
                f"{'' if point.stable else '   UNSTABLE'}",
                level,
            )

        best = sweep.recommended()
        span = sweep.conventional_span()
        if best is None:
            self._end_run("no position in that range is conventionally stable", WARN)
            return
        message = (
            f"put the CG at x {best.x_m:.4f} m ({best.percent_mac:.0f}% MAC) for a "
            f"{best.static_margin * 100:.0f}% static margin"
            + (f" · conventional from {span[0]:.4f} to {span[1]:.4f} m" if span else "")
        )
        self._end_run(message)
        self.statusBar().showMessage(message, 0)

    def solve_dynamics(self) -> None:
        """Modes from the last stability run plus the mass properties.

        Deliberately not a solver call — the aerodynamics is already measured and this is arithmetic
        on it, so it is instant and runs on the GUI thread. What it needs that the aerodynamics does
        not is mass and inertia, and it says so rather than failing obscurely.
        """
        from uavaero.dynamics.report import analyse_aircraft

        if self._file is None:
            return
        if self._derivatives is None:
            QMessageBox.information(
                self, "Run the stability solve first",
                "Dynamic modes are built from the stability derivatives.\n"
                "Solve → Stability derivatives (Ctrl+D), then come back.",
            )
            return
        if not self._file.mass.complete:
            QMessageBox.information(
                self, "Mass and inertia needed",
                f"Dynamic modes need mass and inertia: {self._file.mass.missing()}.\n\n"
                "Enter them on the Airframe screen under Mass. They are stated, not sized - "
                "this tool does not work them out for you.",
            )
            return

        try:
            report = analyse_aircraft(
                self._derivatives, self._file.mass, self._file.condition,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not analyse", f"{type(exc).__name__}: {exc}")
            return

        self.dynamics.set_report(report)
        self.show_screen(3)
        self.console.rule("Dynamic modes")
        for mode in report.modes:
            self._log(mode.summary(), INFO)
        for note in report.notes:
            self._log(f"note: {note}", WARN)
        self._log(report.summary(), GOOD)
        self.statusBar().showMessage(f"Dynamic modes: {report.summary()}", 0)

    def _on_solved(self, result, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None

        if result is None:
            headline = (error or "").splitlines()[0] if error else "the solve failed"
            self._log(f"SOLVE FAILED — {headline}", ERROR)
            self.console.block(error, ERROR)
            self._end_run("see the console for the captured solver output", ERROR)
            self.statusBar().showMessage("Solve failed — see the console", 0)
            QMessageBox.warning(
                self, "Solve failed",
                f"{headline}\n\nThe full solver output is in the console "
                f"(View → Console, Ctrl+`), and 'Open run folder' has the log and the .vsp3.",
            )
            return

        assert self._file is not None
        self._file.record(result)
        self._dirty = True
        self._update_title()

        # Land on the comparison. Solving is not the goal; reading the disagreement is.
        self.comparison.set_file(self._file)
        self.show_screen(1)

        best, alpha = result.best_lift_to_drag
        spread = result.drag.relative_spread(len(result.polar.CL) // 2)
        models = len(result.drag.models)
        for note in result.notes:
            self._log(f"note: {note}", WARN)
        self._end_run(
            f"solved in {result.elapsed_s:.1f} s — {models} drag models, "
            f"best L/D {best:.1f} at {alpha:g}°, spread {spread * 100:.1f}%"
        )
        self.statusBar().showMessage(
            f"Solved in {result.elapsed_s:.1f} s  ·  {models} drag models  ·  "
            f"best L/D {best:.1f} at {alpha:g} deg  ·  they disagree by {spread * 100:.1f}%",
            0,
        )

    def open_in_openvsp(self) -> None:
        """Write the .vsp3 and hand it to OpenVSP's own GUI — a deliberate trip out of this app
        into theirs, for measuring and slicing this tool does not do."""
        import subprocess

        from uavaero.paths import openvsp_gui

        if self._file is None:
            return
        gui = openvsp_gui()
        if gui is None:
            QMessageBox.information(
                self, "OpenVSP not found",
                "No OpenVSP GUI is installed under external/openvsp.\n"
                "Set $UAVAERO_VSP_GUI to point at vsp.exe if it lives elsewhere.",
            )
            return
        target = RUNS / (self._file.airframe.name or "airframe")
        target.mkdir(parents=True, exist_ok=True)
        vsp3 = target / f"{self._file.airframe.name or 'airframe'}.vsp3"
        try:
            from uavaero.geometry.adapters import vsp_worker  # noqa: F401  (presence check)

            self._write_vsp3(vsp3)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not export", f"{type(exc).__name__}: {exc}")
            return
        subprocess.Popen([str(gui), str(vsp3)])
        self.statusBar().showMessage(f"Opened {vsp3.name} in OpenVSP", 5000)

    def _write_vsp3(self, path: Path) -> None:
        """Build through the disposable worker, so an OpenVSP crash is an exit code."""
        import json
        import subprocess
        import sys

        assert self._file is not None
        job = path.parent / "export.job.json"
        out = path.parent / "export.result.json"
        job.write_text(json.dumps({
            "spec": self._file.airframe.to_dict(),
            "vsp3_out": str(path),
            "run_compgeom": False,
            "result_out": str(out),
        }), encoding="utf-8")
        # An explicit cwd, for the same reason `vspaero.run_isolated` sets one: OpenVSP names its
        # artefacts after the model and writes them beside wherever the process happens to be, so a
        # worker that inherits the app's directory drops `<name>_CompGeom.csv` and friends into the
        # repo root. Three of those had been committed by accident before anyone noticed.
        #
        # `env` is not optional company for `cwd`, it is the other half of it: `-m` resolves the
        # module through the working directory, so moving cwd breaks the import and the worker dies
        # with "No module named 'uavaero'" before it runs a line. That is CLAUDE.md rule 21, and it
        # has now been the same mistake twice - the first time in the optimiser's parallel workers.
        from uavaero.solvers.vspaero import _worker_environment

        completed = subprocess.run(
            [sys.executable, "-m", "uavaero.geometry.adapters.vsp_worker", str(job)],
            cwd=str(path.parent), env=_worker_environment(), capture_output=True, timeout=180,
            text=True, encoding="utf-8", errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "the OpenVSP worker exited non-zero")

    # ── lifecycle ──────────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._dirty and self._file is not None:
            answer = QMessageBox.question(
                self, "Unsaved changes",
                f"Save changes to {self._file.name} before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if answer == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if answer == QMessageBox.StandardButton.Save:
                self.save()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        QSettings().setValue("window_geometry", self.saveGeometry())
        super().closeEvent(event)


def _carry_settings_across_the_rename() -> None:
    """One-time: bring the old `uavaero` settings forward under the `UAV Studio` name.

    `QSettings()` keys on the organisation and application names, so renaming the app moves the
    whole store. The theme, the window geometry and the last-opened file are all silently left
    behind, and the app comes up with default everything and nothing saying why — which reads as
    settings not persisting at all.

    Only fills in keys the new location does not already have, so anything set since the rename
    wins. Safe to delete once nobody is upgrading across it.
    """
    current = QSettings()
    previous = QSettings("uavaero", "uavaero")
    for key in previous.allKeys():
        if not current.contains(key):
            current.setValue(key, previous.value(key))
    current.sync()


def launch() -> int:
    """Entry point used by both `main.py` and `python -m uavaero.app`."""
    from uavaero.app.style import apply_style

    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("UAV Studio")
    app.setApplicationName("UAV Studio")
    _carry_settings_across_the_rename()
    icon = QIcon(str(APP_ROOT / "assets" / "uav-studio.ico"))
    app.setWindowIcon(icon)
    apply_style(app)
    window = MainWindow()
    window.setWindowIcon(icon)
    geometry = QSettings().value("window_geometry")
    if geometry is not None:
        window.restoreGeometry(geometry)
    window.showMaximized()
    return app.exec()
