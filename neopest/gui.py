"""NeoPest desktop GUI — MainWindow orchestration."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow

from neopest import __version__
from neopest.gui_layout_builders import build_main_layout
from neopest.gui_styles import build_application_stylesheet


class MainWindow(QMainWindow):
    """Primary workbench window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"NeoPest {__version__}")
        self.resize(1280, 800)
        self.setMinimumSize(960, 640)
        build_main_layout(self)


def run_app(argv: list[str] | None = None) -> int:
    """Launch the NeoPest desktop application."""
    app = QApplication(argv or sys.argv)
    app.setApplicationName("NeoPest")
    app.setApplicationVersion(__version__)
    app.setStyleSheet(build_application_stylesheet())

    window = MainWindow()
    window.show()
    return app.exec()
