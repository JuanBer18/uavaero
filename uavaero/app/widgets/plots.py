"""Matplotlib canvases for the drag comparison, styled from the same palette as everything else.

Matplotlib here rather than QPainter — the opposite of the planform view's choice, and for the
opposite reason. That drawing is line art with no axes, where a figure's margins and DPI would be
something to fight. These are charts: they need ticks, a log-ish range of values, a legend and
sensible autoscaling, all of which matplotlib already does well.

Every colour comes from `style.PALETTE` and `style.COLORWAYS`, re-read on each draw, so the plots
follow a theme change without being rebuilt.
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from uavaero.app.style import COLORWAYS, PALETTE, current_theme
from uavaero.schema.results import AeroResult


class _Canvas(FigureCanvasQTAgg):
    """A figure that repaints itself in the current theme."""

    def __init__(self, width: float = 5.0, height: float = 3.4) -> None:
        self.figure = Figure(figsize=(width, height), layout="constrained")
        super().__init__(self.figure)
        self._result: AeroResult | None = None

    def set_result(self, result: AeroResult | None) -> None:
        self._result = result
        self.redraw()

    def redraw(self) -> None:
        self.figure.clear()
        self._style_figure()
        if self._result is not None:
            self._draw(self._result)
        self.draw_idle()

    def _style_figure(self) -> None:
        self.figure.set_facecolor(PALETTE["surface"])

    def _style_axes(self, axes, xlabel: str, ylabel: str) -> None:
        axes.set_facecolor(PALETTE["surface"])
        axes.tick_params(colors=PALETTE["ink_3"], labelsize=8)
        for spine in ("top", "right"):
            axes.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axes.spines[spine].set_color(PALETTE["rule_strong"])
        axes.grid(True, color=PALETTE["rule"], linewidth=0.6, alpha=0.7)
        axes.set_axisbelow(True)
        axes.set_xlabel(xlabel, color=PALETTE["ink_2"], fontsize=8)
        axes.set_ylabel(ylabel, color=PALETTE["ink_2"], fontsize=8)

    def _legend(self, axes) -> None:
        legend = axes.legend(frameon=False, fontsize=7.5, labelcolor=PALETTE["ink_2"])
        if legend is not None:
            legend.set_zorder(10)

    def _colours(self) -> list[str]:
        return COLORWAYS.get(current_theme(), COLORWAYS["light"])

    def _draw(self, result: AeroResult) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class DragPolarCanvas(_Canvas):
    """CD against CL, every model overlaid.

    The drag polar rather than CD against alpha, because that is the shape you read an aircraft
    from: the intercept is the parasite drag and the curvature is the induced drag, so two models
    that disagree about which of those is wrong separate visibly here and overlap on an alpha plot.
    """

    def _draw(self, result: AeroResult) -> None:
        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "CD", "CL")
        colours = self._colours()
        cl = result.polar.CL
        for i, model in enumerate(result.drag.models.values()):
            axes.plot(
                model.cd, cl,
                color=colours[i % len(colours)],
                # A reference is a measurement, not a prediction. Dashed so it never reads as one
                # more opinion in a row of opinions.
                linestyle="--" if model.reference else "-",
                marker="o" if model.reference else None,
                linewidth=1.6, markersize=3.5, label=model.label,
            )
        if result.drag.cd_induced:
            axes.plot(
                result.drag.cd_induced, cl, color=PALETTE["ink_3"], linewidth=1.0,
                linestyle=":", label="Induced only",
            )
        self._legend(axes)


class LiftToDragCanvas(_Canvas):
    """L/D against angle of attack, every model overlaid."""

    def _draw(self, result: AeroResult) -> None:
        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "alpha (deg)", "L/D")
        colours = self._colours()
        alpha = result.polar.alpha_deg
        for i, model in enumerate(result.drag.models.values()):
            ratios = result.drag.lift_to_drag(model.key, result.polar.CL)
            axes.plot(
                alpha, ratios, color=colours[i % len(colours)],
                linestyle="--" if model.reference else "-",
                linewidth=1.6, label=model.label,
            )
        axes.axhline(0.0, color=PALETTE["rule_strong"], linewidth=0.8)
        self._legend(axes)


class SpanLoadCanvas(_Canvas):
    """Local lift coefficient across the span against each section's own 2D limit.

    The one plot that shows *where* the wing runs out. A polar says the aircraft stalls somewhere
    above 12 degrees; this says it starts at 80% semispan, which is the difference between a docile
    aircraft and one that drops a wing.
    """

    def _draw(self, result: AeroResult) -> None:
        stall = result.native.get("stall")
        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "semispan fraction", "local CL")
        if not stall or not stall.get("stations"):
            axes.text(
                0.5, 0.5, "No span load", ha="center", va="center",
                transform=axes.transAxes, color=PALETTE["ink_3"], fontsize=9,
            )
            return

        stations = stall["stations"]
        fractions = [s["span_fraction"] for s in stations]
        local = [s["cl_local"] for s in stations]
        limits = [s["cl_max"] for s in stations]
        colours = self._colours()

        axes.plot(fractions, local, color=colours[0], linewidth=1.8,
                  label=f"local CL at {stall['alpha_deg']:g} deg")
        if any(v is not None for v in limits):
            axes.plot(
                [f for f, v in zip(fractions, limits) if v is not None],
                [v for v in limits if v is not None],
                color=PALETTE["warn"], linewidth=1.4, linestyle="--",
                label="section Cl_max",
            )
        over = [(f, c) for f, c, m in
                zip(fractions, local, (s["margin"] for s in stations))
                if m is not None and m < 0]
        if over:
            axes.scatter(
                [f for f, _ in over], [c for _, c in over],
                color=PALETTE["warn"], s=26, zorder=5, label="over the limit",
            )
        self._legend(axes)


class RootLocusCanvas(_Canvas):
    """The eigenvalues on the complex plane — every mode of the aircraft in one picture.

    The left half-plane is stable, so the vertical axis is the whole story and it is drawn heavily.
    Radial lines are constant damping ratio: a mode sitting on the 0.3 line has a damping ratio of
    0.3 regardless of how fast it is, which is what makes this plot better than a table for seeing
    whether an aircraft is comfortable rather than merely stable.
    """

    def set_report(self, report) -> None:
        self._report = report
        self.redraw()

    def _draw(self, _unused) -> None:  # pragma: no cover - overridden below
        pass

    def redraw(self) -> None:
        import numpy as np

        self.figure.clear()
        self._style_figure()
        report = getattr(self, "_report", None)
        if report is None:
            self.draw_idle()
            return

        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "real  (1/s)", "imaginary  (rad/s)")
        colours = self._colours()

        modes = report.modes
        reach = max([abs(m.eigenvalue) for m in modes] + [1.0]) * 1.25

        # The stability boundary, drawn heavily because crossing it is the only thing that matters.
        axes.axvline(0.0, color=PALETTE["warn"], linewidth=1.6, zorder=2)
        axes.axhline(0.0, color=PALETTE["rule_strong"], linewidth=0.8, zorder=1)
        axes.fill_betweenx([-reach, reach], 0, reach, color=PALETTE["warn"], alpha=0.06, zorder=0)

        for zeta in (0.3, 0.7):
            angle = np.arccos(zeta)
            for sign in (1, -1):
                axes.plot(
                    [0, -reach * np.cos(0)], [0, sign * reach * np.tan(angle)],
                    color=PALETTE["rule"], linewidth=0.7, linestyle=":", zorder=1,
                )
            axes.annotate(
                f"ζ {zeta}", xy=(-reach * 0.92, reach * 0.92 * float(np.tan(angle)) * 0.92),
                color=PALETTE["ink_3"], fontsize=7,
            )

        for i, mode in enumerate(modes):
            colour = colours[i % len(colours)]
            points = [mode.eigenvalue]
            if mode.oscillatory:
                points.append(complex(mode.eigenvalue.real, -mode.eigenvalue.imag))
            axes.scatter(
                [p.real for p in points], [p.imag for p in points],
                marker="x" if mode.stable else "o",
                s=64, linewidths=2.0, color=colour, label=mode.name, zorder=5,
            )

        axes.set_xlim(-reach, reach * 0.35)
        axes.set_ylim(-reach, reach)
        self._legend(axes)


class ResponseCanvas(_Canvas):
    """Free response to a disturbance in one mode."""

    def set_response(self, times, states, labels, title: str) -> None:
        self._times, self._states, self._labels, self._title = times, states, labels, title
        self.redraw()

    def _draw(self, _unused) -> None:  # pragma: no cover - overridden below
        pass

    def redraw(self) -> None:
        self.figure.clear()
        self._style_figure()
        if getattr(self, "_times", None) is None:
            self.draw_idle()
            return

        axes = self.figure.add_subplot(111)
        self._style_axes(axes, "time  (s)", "state (normalised)")
        colours = self._colours()
        for i, label in enumerate(self._labels):
            axes.plot(
                self._times, self._states[:, i], color=colours[i % len(colours)],
                linewidth=1.5, label=label,
            )
        axes.axhline(0.0, color=PALETTE["rule_strong"], linewidth=0.8)
        axes.set_title(self._title, color=PALETTE["ink_2"], fontsize=9)
        self._legend(axes)
        self.draw_idle()
