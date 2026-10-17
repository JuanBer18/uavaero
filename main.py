"""Desktop entry point.

Run from this directory with the system Python 3.13:

    python main.py

The locale is forced to en_GB so engineering values use a decimal point rather than a comma — a
machine set to a comma locale otherwise turns every field in the app into a parse failure, which
presents as "the app rejects all my numbers" rather than as anything to do with locale.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtCore import QLocale  # noqa: E402

from uavaero.app.main_window import launch  # noqa: E402


def main() -> int:
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedKingdom))
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
