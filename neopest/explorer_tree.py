"""Explorer tree placeholder for workspace navigation."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from neopest.gui_styles import (
    apply_explorer_panel_style,
    apply_hint_style,
    apply_panel_margins,
    apply_section_title_style,
)


class ExplorerTreeWidget(QTreeWidget):
    """Left-side navigation tree — scaffold with placeholder hierarchy."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.refresh_placeholder()

    def refresh_placeholder(self, workspace_label: str = "No workspace open") -> None:
        self.clear()
        root = QTreeWidgetItem([workspace_label])
        root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.addTopLevelItem(root)

        samples = QTreeWidgetItem(["Samples"])
        samples.setFlags(samples.flags() & ~Qt.ItemFlag.ItemIsEditable)
        root.addChild(samples)

        hint = QTreeWidgetItem(["(import images to populate)"])
        hint.setFlags(hint.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        hint.setForeground(0, self.palette().brush(self.foregroundRole()))
        samples.addChild(hint)
        self.expandAll()


def build_explorer_panel(parent: QWidget | None = None) -> tuple[QWidget, ExplorerTreeWidget]:
    """Build the left Explorer panel shell."""
    panel = QWidget(parent)
    apply_explorer_panel_style(panel)

    layout = QVBoxLayout(panel)
    apply_panel_margins(layout)

    title = QLabel("Explorer")
    apply_section_title_style(title)
    layout.addWidget(title)

    subtitle = QLabel("Workspace navigation")
    apply_hint_style(subtitle)
    layout.addWidget(subtitle)

    tree = ExplorerTreeWidget(panel)
    layout.addWidget(tree, stretch=1)
    return panel, tree
