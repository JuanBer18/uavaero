"""The search: set the bounds, run it, and read what it threw away as well as what it kept.

The rejection ledger gets equal billing with the convergence curve, which is unusual and
deliberate. A run that accepts 5% of its candidates is not converging slowly — it is searching a
box that is mostly outside the model's validity, and the objective history shows that as a flat
line indistinguishable from a hard problem. The ledger says *why*, in one glance, and it is the
difference between widening a bound and running overnight for nothing.

The bounds panel leads because bounds are the real input. An optimiser cannot find an aircraft
outside the box it was given, and most disappointing runs are a box drawn wrong rather than a
search that failed.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from uavaero.app.style import COLORWAYS, PALETTE, current_theme, mono_font, ui_font
from uavaero.app.widgets.fields import Field, FieldColumn, GroupLabel
from uavaero.app.widgets.plots import _Canvas
from uavaero.optimisation.driver import Optimiser, OptimisationResult
from uavaero.optimisation.objectives import OBJECTIVES
from uavaero.optimisation.screening import Limits
from uavaero.optimisation.space import DesignSpace, standard_variables
from uavaero.schema.condition import Fidelity


class ConvergenceCanvas(_Canvas):
    """Best score so far against evaluation number.

    Only accepted candidates appear. Plotting the rejected ones at their penalty score would draw a
    solid band at the top of the axis and compress everything that matters into a line.
    """

    def set_result(self, result: OptimisationResult | None) -> None:
        self._outcome = result
        self.redraw()

    def _draw(self, _unused) -> None:  # pragma: no cover
        pass

    def redraw(self) -> None:
        self.figure.clear()
        self._style_figure()
        outcome = getattr(self, "_outcome", None)
        if outcome is None or not outcome.accepted:
            self.draw_idle()
            return

        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "accepted evaluation", "objective")
        colours = self._colours()
        scores = [c.score for c in outcome.accepted]
        curve = outcome.best_so_far()

        axes.plot(range(1, len(scores) + 1), scores, "o", markersize=3.5,
                  color=PALETTE["ink_3"], label="candidate")
        axes.plot(range(1, len(curve) + 1), curve, linewidth=1.8,
                  color=colours[0], label="best so far")
        self._legend(axes)
        self.draw_idle()


class _RunWorker(QObject):
    """Runs the search off the GUI thread. A search is minutes; nothing else is."""

    finished = Signal(object, str)
    progress = Signal(int, object)

    def __init__(self, optimiser: Optimiser, mode: str, samples: int,
                 population: int, generations: int) -> None:
        super().__init__()
        self._optimiser = optimiser
        self._mode = mode
        self._samples = samples
        self._population = population
        self._generations = generations

    def run(self) -> None:
        try:
            if self._mode == "explore":
                outcome = self._optimiser.explore(
                    samples=self._samples, on_progress=self.progress.emit,
                )
            else:
                outcome = self._optimiser.run(
                    population=self._population, generations=self._generations,
                    on_progress=self.progress.emit,
                )
            self.finished.emit(outcome, "")
        except Exception as exc:  # noqa: BLE001 - a failed search is a message, not a crash
            self.finished.emit(None, f"{type(exc).__name__}: {exc}")


class OptimiseScreen(QWidget):
    """Set the box, run the search, read both halves of what came back."""

    #: Something the screen decided on the user's behalf and must not do quietly.
    notice = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._file = None
        self._optimiser: Optimiser | None = None
        self._thread: QThread | None = None
        self._worker: _RunWorker | None = None
        self._result: OptimisationResult | None = None
        self._bounds: dict[str, tuple[Field, Field]] = {}
        self._bound_rows: dict[str, QWidget] = {}

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_setup(), 0)
        root.addWidget(self._build_output(), 1)

    # ── setup ──────────────────────────────────────────────────────────────────────────

    def _build_setup(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ComparisonPanel")
        panel.setFixedWidth(430)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QLabel("DESIGN SEARCH")
        header.setObjectName("PanelHeader")
        font = ui_font(9)
        font.setBold(True)
        header.setFont(font)
        header.setContentsMargins(16, 12, 16, 8)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        body = FieldColumn()
        holder = QVBoxLayout(host)
        holder.setContentsMargins(16, 0, 16, 16)
        holder.addWidget(body)
        holder.addStretch(1)
        scroll.setWidget(host)

        body.add(GroupLabel("What to maximise"))
        self._objective = QComboBox()
        self._objective.setFont(mono_font(9))
        self._objective.addItems(list(OBJECTIVES))
        self._objective.setCurrentText("max_lift_to_drag")
        body.add(self._objective)
        self._target_cl = body.add(Field("at CL", "", decimals=2, minimum=0.05, maximum=1.5))
        self._target_cl.set_value(0.6)

        body.add(GroupLabel("Search"))
        self._samples = body.add(Field("Explore samples", "", decimals=0, minimum=4, maximum=500))
        self._samples.set_value(16)
        self._population = body.add(Field("Population x vars", "", decimals=0, minimum=4, maximum=40))
        self._population.set_value(8)
        self._generations = body.add(Field("Generations", "", decimals=0, minimum=1, maximum=200))
        self._generations.set_value(10)
        self._workers = body.add(Field("Parallel workers", "", decimals=0, minimum=1, maximum=16))
        self._workers.set_value(4)

        body.add(GroupLabel("Bounds"))
        for variable in standard_variables():
            row = QWidget()
            layout = QHBoxLayout(row)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            name = QLabel(variable.name)
            name.setProperty("role", "field")
            name.setFont(ui_font(8))
            name.setFixedWidth(112)
            low = Field("", variable.unit, decimals=3)
            high = Field("", variable.unit, decimals=3)
            low.set_value(variable.low)
            high.set_value(variable.high)
            low.setFixedWidth(110)
            high.setFixedWidth(110)
            layout.addWidget(name)
            layout.addWidget(low)
            layout.addWidget(high)
            layout.addStretch(1)
            self._bounds[variable.name] = (low, high)
            self._bound_rows[variable.name] = row
            body.add(row)

        outer.addWidget(scroll, 1)
        outer.addWidget(self._build_controls())
        return panel

    def _build_controls(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("FooterBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(16, 10, 16, 12)
        layout.setSpacing(8)

        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self._explore_button = QPushButton("Explore")
        self._explore_button.setProperty("chrome", "true")
        self._explore_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._explore_button.clicked.connect(lambda: self._start("explore"))
        self._run_button = QPushButton("Optimise")
        self._run_button.setProperty("chrome", "true")
        self._run_button.setProperty("primary", "true")
        self._run_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._run_button.clicked.connect(lambda: self._start("optimise"))
        buttons.addWidget(self._explore_button)
        buttons.addWidget(self._run_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        hint = QLabel(
            "Explore first. It costs a known amount of time and exercises the bounds; if it "
            "accepts nothing, the search would have discovered the same thing one candidate at a "
            "time over an hour."
        )
        hint.setWordWrap(True)
        hint.setProperty("role", "derived-label")
        hint.setFont(ui_font(8))
        layout.addWidget(hint)
        return bar

    # ── output ─────────────────────────────────────────────────────────────────────────

    def _build_output(self) -> QWidget:
        self._tabs = QTabWidget()
        self._convergence = ConvergenceCanvas()
        self._candidates = QPlainTextEdit()
        self._rejections = QPlainTextEdit()
        for widget in (self._candidates, self._rejections):
            widget.setReadOnly(True)
            widget.setFont(mono_font(8))
        self._tabs.addTab(self._convergence, "Convergence")
        self._tabs.addTab(self._candidates, "Candidates")
        self._tabs.addTab(self._rejections, "Rejections")
        return self._tabs

    # ── running ────────────────────────────────────────────────────────────────────────

    def set_file(self, file) -> None:
        self._file = file
        self._hide_inapplicable_bounds()

    def _hide_inapplicable_bounds(self) -> None:
        """Only show the knobs that can move this aircraft.

        A tail variable on a tailless airframe is dropped by `DesignSpace`, so leaving its bounds
        row on screen invites you to set a range that is then silently ignored.
        """
        airframe = self._file.airframe if self._file is not None else None
        for variable in standard_variables():
            row = self._bound_rows.get(variable.name)
            if row is not None:
                row.setVisible(airframe is None or variable.relevant_for(airframe))

    def _limits(self) -> Limits:
        span_high = self._bounds["span"][1].value() or 4.0
        return Limits(max_span_m=max(span_high * 1.05, span_high + 0.05))

    def _variables(self):
        overrides = {}
        for name, (low, high) in self._bounds.items():
            bounds = (low.value(), high.value())
            if bounds[0] is not None and bounds[1] is not None and bounds[0] < bounds[1]:
                overrides[name] = bounds
        return standard_variables(**overrides)

    def _start(self, mode: str) -> None:
        from PySide6.QtWidgets import QMessageBox

        if self._file is None or self._thread is not None:
            return
        try:
            space = DesignSpace(self._file.airframe, self._variables())
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot search", f"{type(exc).__name__}: {exc}")
            return

        if space.omitted:
            # Not a warning about a mistake - it is the space adapting to the aircraft. But a
            # search over nine variables when the panel offered twelve has to be said, or the
            # result looks like the optimiser ignored half the settings.
            self.notice.emit(
                f"searching {len(space.variables)} variables, not {len(space.variables) + len(space.omitted)}: "
                f"{', '.join(space.omitted)} cannot apply to this aircraft"
            )

        factory = OBJECTIVES[self._objective.currentText()]
        self._optimiser = Optimiser(
            space,
            self._file.condition,
            objective=factory(self._target_cl.value() or 0.6),
            limits=self._limits(),
            search_fidelity=Fidelity.QUICK,
            workers=int(self._workers.value() or 4),
        )

        samples = int(self._samples.value() or 16)
        population = int(self._population.value() or 8)
        generations = int(self._generations.value() or 10)
        expected = samples if mode == "explore" else population * len(space.variables) * generations
        self._progress.setRange(0, expected)
        self._progress.setValue(0)
        self._set_running(True)

        self._thread = QThread(self)
        self._worker = _RunWorker(self._optimiser, mode, samples, population, generations)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, index: int, candidate) -> None:
        self._progress.setValue(index)
        verdict = (
            f"accepted   objective {candidate.score:10.4f}"
            if candidate.accepted else f"rejected   {candidate.reason}"
        )
        self._progress.setFormat(f"%v / %m   —   {verdict[:70]}")

    def _on_finished(self, outcome, error: str) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._set_running(False)

        if outcome is None:
            self._rejections.setPlainText(error)
            self._tabs.setCurrentIndex(2)
            return

        self._result = outcome
        self._convergence.set_result(outcome)
        self._rejections.setPlainText(outcome.ledger.summary())
        self._candidates.setPlainText(self._candidate_table(outcome))
        self._progress.setFormat(outcome.summary().splitlines()[0])
        self._tabs.setCurrentIndex(0 if outcome.accepted else 2)

    @staticmethod
    def _candidate_table(outcome: OptimisationResult) -> str:
        accepted = sorted(outcome.accepted, key=lambda c: c.score)
        if not accepted:
            return (
                "No candidate was accepted.\n\n"
                "Read the Rejections tab — it says which check threw them away, and the answer "
                "is usually that a bound is drawn outside where the model is valid."
            )
        names = list(accepted[0].values)
        header = "  score      L/D    " + "".join(f"{n[:11]:>12}" for n in names)
        lines = [outcome.summary(), "", header, "-" * len(header)]
        for candidate in accepted[:40]:
            ratio = f"{candidate.lift_to_drag:7.2f}" if candidate.lift_to_drag else "      -"
            values = "".join(f"{candidate.values[n]:12.3f}" for n in names)
            lines.append(f"{candidate.score:8.3f} {ratio}    {values}")
        return "\n".join(lines)

    def _set_running(self, running: bool) -> None:
        self._run_button.setEnabled(not running)
        self._explore_button.setEnabled(not running)

    def retheme(self) -> None:
        self._convergence.redraw()
