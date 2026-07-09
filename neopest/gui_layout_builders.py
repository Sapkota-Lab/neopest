"""Main window layout construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from neopest.explorer_tree import build_explorer_panel
from neopest.gui_canvas import ImageCanvas, build_image_canvas_host
from neopest.gui_styles import (
    CONTROL_ROW_SPACING,
    INSPECTOR_PANEL_MAX_WIDTH,
    INSPECTOR_PANEL_MIN_WIDTH,
    LEFT_PANEL_MIN_WIDTH,
    PANEL_MARGIN,
    apply_hint_style,
    apply_inspector_panel_style,
    apply_main_splitter_style,
    apply_panel_margins,
    apply_section_title_style,
    apply_workbench_controls_panel_style,
    apply_workspace_preview_panel_style,
    make_vertical_divider,
)

if TYPE_CHECKING:
    from neopest.gui import MainWindow


def build_inspector_panel(parent: QWidget | None = None) -> QWidget:
    """Right-side inspector / contextual settings placeholder."""
    panel = QWidget(parent)
    apply_inspector_panel_style(panel)
    panel.setMinimumWidth(INSPECTOR_PANEL_MIN_WIDTH)
    panel.setMaximumWidth(INSPECTOR_PANEL_MAX_WIDTH)
    panel.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

    layout = QVBoxLayout(panel)
    apply_panel_margins(layout)

    title = QLabel("Inspector")
    apply_section_title_style(title)
    layout.addWidget(title)

    subtitle = QLabel("Sample details and analysis settings")
    apply_hint_style(subtitle)
    layout.addWidget(subtitle)

    layout.addWidget(make_vertical_divider())

    fields = [
        ("Sample", "—"),
        ("Calibration", "Uncalibrated"),
        ("Leaf area", "—"),
        ("Disease area", "—"),
        ("Disease %", "—"),
    ]
    for label_text, value_text in fields:
        row = QHBoxLayout()
        row.setSpacing(CONTROL_ROW_SPACING)
        label = QLabel(label_text)
        apply_hint_style(label)
        value = QLabel(value_text)
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(value)
        layout.addLayout(row)

    layout.addStretch(1)
    return panel


def build_workbench_controls(parent: QWidget | None = None) -> tuple[QWidget, QPushButton]:
    """Lower status/control strip beneath the main workspace."""
    panel = QWidget(parent)
    apply_workbench_controls_panel_style(panel)
    panel.setFixedHeight(56)

    layout = QHBoxLayout(panel)
    layout.setContentsMargins(PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN, PANEL_MARGIN)
    layout.setSpacing(12)

    status = QLabel("Ready — Phase 0 scaffold")
    apply_hint_style(status)
    layout.addWidget(status, stretch=1)

    run_button = QPushButton("Run Analysis")
    run_button.setEnabled(False)
    run_button.setToolTip("Analysis pipeline not yet implemented")
    layout.addWidget(run_button)

    return panel, run_button


def build_main_layout(window: MainWindow) -> None:
    """Construct the Workbench shell and attach widgets to ``window``."""
    central = QWidget(window)
    window.setCentralWidget(central)

    root = QVBoxLayout(central)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    horizontal_splitter = QSplitter(Qt.Orientation.Horizontal, central)
    apply_main_splitter_style(horizontal_splitter)

    explorer_panel, explorer_tree = build_explorer_panel(horizontal_splitter)
    explorer_panel.setMinimumWidth(LEFT_PANEL_MIN_WIDTH)
    window.explorer_panel = explorer_panel
    window.explorer_tree = explorer_tree

    center_column = QWidget(horizontal_splitter)
    apply_workspace_preview_panel_style(center_column)
    center_layout = QVBoxLayout(center_column)
    center_layout.setContentsMargins(0, 0, 0, 0)
    center_layout.setSpacing(0)

    canvas_host, image_canvas = build_image_canvas_host(center_column)
    center_layout.addWidget(canvas_host, stretch=1)
    window.image_canvas = image_canvas

    controls_panel, run_button = build_workbench_controls(center_column)
    center_layout.addWidget(controls_panel)
    window.controls_panel = controls_panel
    window.run_analysis_button = run_button

    inspector_panel = build_inspector_panel(horizontal_splitter)
    window.inspector_panel = inspector_panel

    horizontal_splitter.addWidget(explorer_panel)
    horizontal_splitter.addWidget(center_column)
    horizontal_splitter.addWidget(inspector_panel)
    horizontal_splitter.setStretchFactor(0, 1)
    horizontal_splitter.setStretchFactor(1, 4)
    horizontal_splitter.setStretchFactor(2, 1)
    horizontal_splitter.setSizes([LEFT_PANEL_MIN_WIDTH, 900, INSPECTOR_PANEL_MIN_WIDTH])

    root.addWidget(horizontal_splitter, stretch=1)

    status_bar = QStatusBar(window)
    status_bar.showMessage("NeoPest — strawberry leaf disease analysis workbench")
    window.setStatusBar(status_bar)
    window.status_bar = status_bar
