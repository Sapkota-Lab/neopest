"""Shared Qt style tokens and helpers for NeoPest."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QSplitter, QVBoxLayout, QWidget

# Layout spacing
PANEL_MARGIN = 8
PANEL_SECTION_SPACING = 10
CONTROL_ROW_SPACING = 6

# Object names for QSS targeting
EXPLORER_PANEL_OBJECT_NAME = "explorerPanel"
WORKSPACE_PREVIEW_PANEL_OBJECT_NAME = "workspacePreviewPanel"
INSPECTOR_PANEL_OBJECT_NAME = "inspectorPanel"
WORKBENCH_CONTROLS_PANEL_OBJECT_NAME = "workbenchControlsPanel"
IMAGE_CANVAS_OBJECT_NAME = "imageCanvas"

# Dark scientific workbench palette
COLOR_EXPLORER_BACKGROUND = "#1e1e1e"
COLOR_WORKSPACE_BACKGROUND = "#252526"
COLOR_INSPECTOR_BACKGROUND = "#252526"
COLOR_WORKSPACE_DIVIDER = "#333333"
COLOR_EXPLORER_SELECTION = "#2a4a66"
COLOR_EXPLORER_HOVER = "#2a2d2e"

COLOR_CONTROL_BACKGROUND = "#2a2d2e"
COLOR_CONTROL_BORDER = "#3c3c3c"
COLOR_CONTROL_TEXT = "#cccccc"
COLOR_CONTROL_TEXT_DISABLED = "#666666"
COLOR_MUTED_TEXT = "#858585"
COLOR_ACCENT = "#3a7ca5"

COLOR_STATUS_BAR_BACKGROUND = "#007acc"
COLOR_STATUS_BAR_TEXT = "#ffffff"

LEFT_PANEL_MIN_WIDTH = 220
INSPECTOR_PANEL_MIN_WIDTH = 240
INSPECTOR_PANEL_MAX_WIDTH = 320


def apply_panel_margins(layout: QVBoxLayout, *, margin: int = PANEL_MARGIN) -> None:
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(PANEL_SECTION_SPACING)


def apply_explorer_panel_style(widget: QWidget) -> None:
    widget.setObjectName(EXPLORER_PANEL_OBJECT_NAME)
    widget.setStyleSheet(
        f"QWidget#{EXPLORER_PANEL_OBJECT_NAME} {{"
        f" background-color: {COLOR_EXPLORER_BACKGROUND};"
        f" color: {COLOR_CONTROL_TEXT};"
        f" border: none;"
        f"}}"
    )


def apply_workspace_preview_panel_style(widget: QWidget) -> None:
    widget.setObjectName(WORKSPACE_PREVIEW_PANEL_OBJECT_NAME)
    widget.setStyleSheet(
        f"QWidget#{WORKSPACE_PREVIEW_PANEL_OBJECT_NAME} {{"
        f" background-color: {COLOR_WORKSPACE_BACKGROUND};"
        f" color: {COLOR_CONTROL_TEXT};"
        f" border: none;"
        f"}}"
    )


def apply_inspector_panel_style(widget: QWidget) -> None:
    widget.setObjectName(INSPECTOR_PANEL_OBJECT_NAME)
    widget.setStyleSheet(
        f"QWidget#{INSPECTOR_PANEL_OBJECT_NAME} {{"
        f" background-color: {COLOR_INSPECTOR_BACKGROUND};"
        f" color: {COLOR_CONTROL_TEXT};"
        f" border-left: 1px solid {COLOR_WORKSPACE_DIVIDER};"
        f"}}"
    )


def apply_workbench_controls_panel_style(widget: QWidget) -> None:
    widget.setObjectName(WORKBENCH_CONTROLS_PANEL_OBJECT_NAME)
    widget.setStyleSheet(
        f"QWidget#{WORKBENCH_CONTROLS_PANEL_OBJECT_NAME} {{"
        f" background-color: {COLOR_WORKSPACE_BACKGROUND};"
        f" color: {COLOR_CONTROL_TEXT};"
        f" border-top: 1px solid {COLOR_WORKSPACE_DIVIDER};"
        f"}}"
    )


def apply_main_splitter_style(splitter: QSplitter) -> None:
    splitter.setHandleWidth(1)
    splitter.setStyleSheet(
        f"QSplitter::handle {{ background-color: {COLOR_WORKSPACE_DIVIDER}; }}"
    )


def apply_hint_style(label: QLabel) -> None:
    label.setStyleSheet(f"color: {COLOR_MUTED_TEXT}; font-size: 12px;")


def apply_section_title_style(label: QLabel) -> None:
    label.setStyleSheet(
        f"color: {COLOR_CONTROL_TEXT}; font-size: 13px; font-weight: 600;"
    )


def make_vertical_divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setStyleSheet(f"background-color: {COLOR_WORKSPACE_DIVIDER}; max-height: 1px;")
    return line


def build_application_stylesheet() -> str:
    """Global application stylesheet for controls and chrome."""
    return f"""
    QWidget {{
        font-family: "Helvetica Neue", "Segoe UI", sans-serif;
        font-size: 13px;
        color: {COLOR_CONTROL_TEXT};
    }}
    QMainWindow {{
        background-color: {COLOR_WORKSPACE_BACKGROUND};
    }}
    QTreeWidget {{
        background-color: {COLOR_EXPLORER_BACKGROUND};
        color: {COLOR_CONTROL_TEXT};
        border: none;
        outline: none;
    }}
    QTreeWidget::item:hover {{
        background-color: {COLOR_EXPLORER_HOVER};
    }}
    QTreeWidget::item:selected {{
        background-color: {COLOR_EXPLORER_SELECTION};
    }}
    QPushButton {{
        background-color: {COLOR_CONTROL_BACKGROUND};
        color: {COLOR_CONTROL_TEXT};
        border: 1px solid {COLOR_CONTROL_BORDER};
        border-radius: 4px;
        padding: 6px 12px;
        min-height: 24px;
    }}
    QPushButton:hover {{
        background-color: #333333;
    }}
    QPushButton:disabled {{
        color: {COLOR_CONTROL_TEXT_DISABLED};
    }}
    QStatusBar {{
        background-color: {COLOR_STATUS_BAR_BACKGROUND};
        color: {COLOR_STATUS_BAR_TEXT};
    }}
    """
