"""Explorer tree for workspace navigation."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHeaderView, QLabel, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from neopest.gui_styles import (
    apply_explorer_panel_style,
    apply_hint_style,
    apply_panel_margins,
    apply_section_title_style,
)
from neopest.metadata import WorkspaceMetadata

NODE_TYPE_ROLE = Qt.ItemDataRole.UserRole
NODE_ID_ROLE = Qt.ItemDataRole.UserRole + 1


class ExplorerTreeWidget(QTreeWidget):
    """Left-side navigation tree bound to workspace metadata."""

    sample_selected = Signal(str)
    selection_cleared = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.refresh_placeholder()

    def refresh_placeholder(self, workspace_label: str = "No workspace open") -> None:
        self.blockSignals(True)
        self.clear()
        root = QTreeWidgetItem([workspace_label])
        root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.addTopLevelItem(root)

        hint = QTreeWidgetItem(["Open or create a workspace to begin"])
        hint.setFlags(hint.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        root.addChild(hint)
        self.expandAll()
        self.blockSignals(False)

    def refresh_workspace(self, workspace: WorkspaceMetadata | None) -> None:
        self.blockSignals(True)
        self.clear()
        if workspace is None:
            self.refresh_placeholder()
            return

        root = QTreeWidgetItem([workspace.display_name])
        root.setData(0, NODE_TYPE_ROLE, "workspace")
        root.setData(0, NODE_ID_ROLE, workspace.workspace_id)
        root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.addTopLevelItem(root)

        if not workspace.condition_groups:
            hint = QTreeWidgetItem(["(no condition groups)"])
            hint.setFlags(hint.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            root.addChild(hint)
        else:
            for group in workspace.condition_groups:
                group_item = QTreeWidgetItem([group.display_name])
                group_item.setData(0, NODE_TYPE_ROLE, "condition_group")
                group_item.setData(0, NODE_ID_ROLE, group.condition_group_id)
                group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                root.addChild(group_item)

                group_samples = workspace.samples_for_group(group.condition_group_id)
                if not group_samples:
                    hint = QTreeWidgetItem(["(no samples)"])
                    hint.setFlags(hint.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                    group_item.addChild(hint)
                else:
                    for sample in sorted(group_samples, key=lambda item: item.display_name.lower()):
                        sample_item = QTreeWidgetItem([sample.display_name])
                        sample_item.setData(0, NODE_TYPE_ROLE, "sample")
                        sample_item.setData(0, NODE_ID_ROLE, sample.sample_id)
                        sample_item.setFlags(sample_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        group_item.addChild(sample_item)

        self.expandAll()
        self.blockSignals(False)

    def select_sample(self, sample_id: str) -> None:
        for index in range(self.topLevelItemCount()):
            root = self.topLevelItem(index)
            if root is None:
                continue
            match = self._find_sample_item(root, sample_id)
            if match is not None:
                self.setCurrentItem(match)
                return

    def _find_sample_item(
        self,
        parent: QTreeWidgetItem,
        sample_id: str,
    ) -> QTreeWidgetItem | None:
        for group_index in range(parent.childCount()):
            group_item = parent.child(group_index)
            if group_item is None:
                continue
            for sample_index in range(group_item.childCount()):
                sample_item = group_item.child(sample_index)
                if sample_item is None:
                    continue
                if (
                    sample_item.data(0, NODE_TYPE_ROLE) == "sample"
                    and sample_item.data(0, NODE_ID_ROLE) == sample_id
                ):
                    return sample_item
        return None

    def _on_selection_changed(self) -> None:
        item = self.currentItem()
        if item is None:
            self.selection_cleared.emit()
            return
        if item.data(0, NODE_TYPE_ROLE) != "sample":
            self.selection_cleared.emit()
            return
        sample_id = item.data(0, NODE_ID_ROLE)
        if sample_id:
            self.sample_selected.emit(str(sample_id))


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
