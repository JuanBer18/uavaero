"""The Cobalt Technical light/dark themes and the stylesheet built from them.

The selected visual direction is Retro Instrument: cobalt structure, compact text-only controls,
rounded instrument-like buttons, and no decorative iconography.  Theme selection is persisted in
``QSettings`` and changes the shared palette in place so the custom-painted planform and Qt chrome
always use the same colours.

Colours live in `PALETTE` because the planform widget paints from the same dictionary the
stylesheet is generated from. A drawing whose colours drift from the chrome around it is the most
obvious way for a careful design to start looking careless.

`accent` means "selected, derived, or the primary action". `good`/`warn` mean "this is the state of
your design". Using one for the other makes a screen unreadable at a glance, which is the entire
job of a status readout.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QWidget

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "ground": "#f2f5fb",
        "surface": "#ffffff",
        "inset": "#e7ecf7",
        "rule": "#cbd3e7",
        "rule_strong": "#aebcdf",
        "ink": "#1a2650",
        "ink_2": "#4f5f8b",
        "ink_3": "#7582a7",
        "accent": "#5778df",
        "accent_dim": "#e4e9fb",
        "accent_ink": "#ffffff",
        "title": "#273b80",
        "good": "#247a68",
        "warn": "#b94f58",
    },
    "dark": {
        "ground": "#101426",
        "surface": "#151a2b",
        "inset": "#252d49",
        "rule": "#343d60",
        "rule_strong": "#48557c",
        "ink": "#e0e5f7",
        "ink_2": "#a8b3d1",
        "ink_3": "#7f8baa",
        "accent": "#6f8cff",
        "accent_dim": "#232d53",
        "accent_ink": "#0d1221",
        "title": "#273b80",
        "good": "#64c7ad",
        "warn": "#ef7780",
    },
}

#: Series colours for the drag-model comparison, per theme.
#:
#: The accent cannot do this job. It means "selected, derived, or the primary action", and a chart
#: where three unrelated series are all accent-coloured says they are all somehow the selected one.
#: These are ordered so the first three — the three drag models — stay distinguishable in both
#: themes and remain separable to the most common form of colour blindness, which is why there is
#: no red/green pair among them.
COLORWAYS: dict[str, list[str]] = {
    "light": ["#2f5fd0", "#c1782a", "#1d8a76", "#8d5bb5", "#b0485c", "#5b7a1f"],
    "dark":  ["#7fa2ff", "#e9a75a", "#5fc9ad", "#c297e8", "#f0808f", "#a8c95e"],
}

DEFAULT_THEME = "light"
# Mutated rather than rebound: planform_view imports this object directly.
PALETTE: dict[str, str] = dict(PALETTES[DEFAULT_THEME])

#: Preferred first, each with a fallback that exists on a stock Windows install. Archivo and DM
#: Mono are what the mockup was set in; if neither is installed the stack degrades to something of
#: the same character rather than to whatever Qt picks by default.
UI_FAMILIES = ["Cascadia Mono", "IBM Plex Mono", "Consolas", "Segoe UI", "monospace"]
MONO_FAMILIES = ["Cascadia Mono", "IBM Plex Mono", "Consolas", "monospace"]

STYLE_QSS_PATH = Path(__file__).parent / "style.qss"


def ui_font(size: int = 10, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(UI_FAMILIES)
    font.setPointSize(size)
    font.setWeight(weight)
    return font


def mono_font(size: int = 10) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FAMILIES)
    font.setPointSize(size)
    # Numbers in a column that do not line up read as sloppy in a tool full of them.
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def available_families() -> list[str]:
    """Which of the preferred families the machine actually has.

    Reported by the environment check rather than silently substituted: a screen set in the wrong
    face still works, and the difference is exactly the kind of thing nobody notices for months.

    Returns an empty list when there is no `QApplication` yet. That guard is not defensive
    tidiness — `QFontDatabase.families()` does not raise without one, it **aborts the process**
    (exit 127, no traceback), so a diagnostic helper called from a script would take the script
    down with it.
    """
    if QApplication.instance() is None:
        return []
    installed = set(QFontDatabase.families())
    return [f for f in (*UI_FAMILIES, *MONO_FAMILIES) if f in installed]


def normalise_theme(theme: str | None) -> str:
    return theme if theme in PALETTES else DEFAULT_THEME


def current_theme() -> str:
    return normalise_theme(str(QSettings().value("appearance/theme", DEFAULT_THEME)))


def load_stylesheet(theme: str | None = None) -> str:
    """Return QSS for ``theme`` and update the shared painter palette in place."""
    selected = normalise_theme(theme or current_theme())
    PALETTE.clear()
    PALETTE.update(PALETTES[selected])
    return STYLE_QSS_PATH.read_text(encoding="utf-8").format(**PALETTE)


def set_theme(app: QApplication, theme: str) -> str:
    """Apply and persist a theme. Returns the normalised selected name."""
    selected = normalise_theme(theme)
    QSettings().setValue("appearance/theme", selected)
    app.setStyleSheet(load_stylesheet(selected))
    for widget in app.allWidgets():
        # `QWidget.update(widget)` rather than `widget.update()`. Some subclasses override
        # `update` with an incompatible signature — `QListView.update(QModelIndex)` is the one
        # that bites, and every QComboBox owns a QListView for its popup, so the bare call raised
        # TypeError on any screen with a dropdown on it. Going through QWidget explicitly asks for
        # the repaint that was actually meant.
        QWidget.update(widget)
    return selected


def apply_style(app: QApplication) -> None:
    app.setFont(ui_font())
    app.setStyleSheet(load_stylesheet(current_theme()))
