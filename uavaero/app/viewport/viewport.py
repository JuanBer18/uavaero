"""The embedded 3D view: the aircraft, in the app, rotatable.

**Why this is not OpenVSP's own window.** OpenVSP's GUI is a separate native application with its
own GL context and event loop and no embedding API, so there is no version of this that puts their
viewport inside our window. What there is instead suits the workflow better: this renders the same
`ParametricAirframe` that generates the `.vsp3` and that VSPAERO solves, so what you rotate is what
gets meshed — not a second model free to drift from it. OpenVSP stays reachable through the
explicit "Open in OpenVSP" action for the things it is genuinely better at.

**Nothing touches GL until the view is actually opened.** Constructing a `QtInteractor` where no
valid pixel format is available segfaults rather than raising, so eager construction would make the
whole app hostage to a graphics driver — an aircraft nobody asked to rotate would take the process
down at startup. The probe runs first, in a process we can afford to lose, and a failure degrades
to a message in the panel. A design tool that will not start because a driver is old is worse than
one with no 3D view.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from uavaero.app.style import PALETTE
from uavaero.schema.airframe import ParametricAirframe

#: How much tighter than `view_isometric`'s default framing to sit. Measured against the reference
#: aircraft: the default leaves it filling about a third of the window.
ZOOM = 1.5


class Viewport3D(QWidget):
    """A PyVista scene in a Qt panel. `show_airframe` is most of the API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._layout = layout
        self._plotter: Any | None = None
        self._fallback: QLabel | None = None
        self._tried = False
        self._pending: ParametricAirframe | None = None

        # Mode animation. The actor is kept so its transform can be replaced each frame rather
        # than the mesh rebuilt - rebuilding is 24 ms, which is a slideshow, and the whole point
        # is to see a rhythm.
        self._actors: list[Any] = []
        self._shape: Any = None
        self._frame = 0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)

    # ── construction ──────────────────────────────────────────────────────────────────────

    def ensure_built(self) -> bool:
        """Build the render window, once, on first genuine use. Deliberately not in `__init__`."""
        if self._tried:
            return self._plotter is not None
        self._tried = True

        from uavaero.app.viewport.probe import opengl_available

        if not opengl_available():
            self._show_fallback(
                "3D view unavailable - no usable OpenGL context on this machine.\n\n"
                "The planform view, the solvers and the .vsp3 export are unaffected."
            )
            return False

        try:
            from pyvistaqt import QtInteractor

            plotter = QtInteractor(self)
            plotter.set_background(PALETTE["inset"])
            self._layout.addWidget(plotter.interactor)
            self._plotter = plotter
        except Exception as exc:  # noqa: BLE001 - any driver or binding failure, not a specific one
            self._show_fallback(
                f"3D view unavailable on this machine.\n{type(exc).__name__}: {exc}\n\n"
                "The planform view and the .vsp3 export are unaffected."
            )
            return False
        return True

    def _show_fallback(self, message: str) -> None:
        label = QLabel(message)
        label.setWordWrap(True)
        label.setStyleSheet(f"color: {PALETTE['ink_3']}; padding: 24px;")
        self._layout.addWidget(label)
        self._fallback = label

    @property
    def available(self) -> bool:
        """Whether a scene can be drawn. Builds nothing — asking has to stay cheap."""
        return self._plotter is not None

    def activate(self) -> None:
        """Called when the 3D view is opened: build if needed, then draw what was last set."""
        if self.ensure_built() and self._pending is not None:
            self.show_airframe(self._pending, reset_camera=True)
            self._pending = None

    # ── drawing ───────────────────────────────────────────────────────────────────────────

    def show_airframe(self, airframe: ParametricAirframe, reset_camera: bool = False) -> None:
        """Redraw the scene.

        `reset_camera` only on the first draw or an explicit reset. Holding the camera across edits
        is what makes changing a number read as *this* aircraft changing rather than a new one
        appearing, which is the whole reason to have a live view rather than a render button.
        """
        if self._plotter is None:
            # Never opened: remember what to draw, so opening it shows the current aircraft
            # rather than an empty scene.
            self._pending = airframe
            return
        from uavaero.geometry.adapters.mesh import airframe_to_polydata

        try:
            mesh = airframe_to_polydata(airframe)
        except Exception as exc:  # noqa: BLE001
            self.show_message(f"Could not build the preview mesh.\n{type(exc).__name__}: {exc}")
            return

        camera = None if reset_camera else self._plotter.camera_position
        self._plotter.clear()
        self._actors = []
        if mesh.n_points:
            self._actors.append(self._plotter.add_mesh(
                mesh, color=PALETTE["accent"], smooth_shading=True,
                specular=0.25, specular_power=18, show_edges=False,
            ))
            # Edges as a faint second pass, so the planform reads as panels rather than as a
            # blob. Kept very low: the preview meshes 20 chords per panel, and at full strength
            # that grid is dense enough to look like surface texture instead of structure.
            self._actors.append(self._plotter.add_mesh(
                mesh, style="wireframe", color=PALETTE["rule_strong"], line_width=1, opacity=0.12,
            ))
        self._plotter.add_axes(color=PALETTE["ink_3"], xlabel="x", ylabel="y", zlabel="z")
        if camera is None:
            self._plotter.view_isometric()
            # `view_isometric` frames the bounding box with generous margin; an aircraft is a wide
            # flat object and ends up small in it.
            self._plotter.camera.zoom(ZOOM)
        else:
            self._plotter.camera_position = camera
        self._plotter.render()

    # ── mode animation ────────────────────────────────────────────────────────────────────

    def play_mode(self, shape) -> bool:
        """Fly the aircraft through one dynamic mode, on repeat.

        `shape` is a `dynamics.shapes.ModeShape` - a list of attitudes already sampled, with the
        decay envelope baked in. Nothing aerodynamic happens here: the physics was done in a pure
        module that can be tested without a graphics context, and this only orients an actor.
        """
        if not self.ensure_built() or not self._actors or not shape.poses:
            return False
        self._shape = shape
        self._frame = 0
        self._timer.start()
        return True

    def stop_mode(self) -> None:
        """Stop and return the aircraft to level. Idempotent."""
        self._timer.stop()
        self._shape = None
        self._frame = 0
        self._apply_pose(0.0, 0.0, 0.0)

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    def _advance(self) -> None:
        if self._shape is None or not self._shape.poses:
            self._timer.stop()
            return
        pose = self._shape.poses[self._frame % len(self._shape.poses)]
        self._frame += 1
        self._apply_pose(pose.roll_deg, pose.pitch_deg, pose.yaw_deg)

    def _apply_pose(self, roll_deg: float, pitch_deg: float, yaw_deg: float) -> None:
        """Rotate every actor to one attitude.

        Applied to the actors rather than the camera, because moving the camera would make the
        *world* appear to roll - which reads as the viewer banking, not the aeroplane. Rotations
        are set from identity each frame rather than accumulated, so floating-point drift cannot
        slowly tip the model over during a long animation.
        """
        if self._plotter is None:
            return
        for actor in self._actors:
            if actor is None:
                continue
            actor.SetOrientation(0.0, 0.0, 0.0)
            actor.RotateZ(yaw_deg)
            actor.RotateY(pitch_deg)
            actor.RotateX(roll_deg)
        self._plotter.render()

    def show_message(self, message: str) -> None:
        if self._plotter is None:
            if self._fallback is not None:
                self._fallback.setText(message)
            return
        self._plotter.clear()
        self._plotter.add_text(message, position="upper_left", font_size=9, color=PALETTE["ink_3"])
        self._plotter.render()

    def reset_view(self) -> None:
        if self._plotter is not None:
            self._plotter.view_isometric()
            self._plotter.camera.zoom(ZOOM)
            self._plotter.render()

    def retheme(self) -> None:
        """The theme is switchable at runtime, and the scene background is painted by VTK rather
        than by the stylesheet — so it has to be told."""
        if self._plotter is not None:
            self._plotter.set_background(PALETTE["inset"])
            self._plotter.render()

    # ── teardown ──────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # pyvistaqt holds a VTK render window that outlives the widget unless it is closed
        # explicitly; skipping this leaves the process alive after the main window shuts.
        self._timer.stop()
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
        super().closeEvent(event)
