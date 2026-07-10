"""Explorer tree for workspace navigation."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QDrag, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from neopest.desktop_services import reveal_image_menu_label
from neopest.gui_styles import (
    apply_explorer_panel_style,
)
from neopest.metadata import WorkspaceMetadata

NODE_TYPE_ROLE = Qt.ItemDataRole.UserRole
NODE_ID_ROLE = Qt.ItemDataRole.UserRole + 1
MIME_SAMPLE_ID = "application/x-neopest-sample-id"
DROP_HIGHLIGHT_ROLE = Qt.ItemDataRole.UserRole + 2


class ExplorerTreeWidget(QTreeWidget):
    """Left-side navigation tree bound to workspace metadata."""

    sample_selected = Signal(str)
    selection_cleared = Signal()

    new_condition_group_requested = Signal()
    import_images_requested = Signal()
    import_images_into_group_requested = Signal(str)
    open_workspace_folder_requested = Signal()
    rename_condition_group_requested = Signal(str)
    remove_condition_group_requested = Signal(str)
    sample_preview_requested = Signal(str)
    rename_sample_requested = Signal(str)
    remove_sample_requested = Signal(str)
    reveal_sample_image_requested = Signal(str)
    sample_move_requested = Signal(str, str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setIndentation(16)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Workspace explorer")
        self.setToolTip("Browse condition groups and samples in the open workspace")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.viewport().setAcceptDrops(True)
        header = self.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self._drop_highlight_item: QTreeWidgetItem | None = None
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
        root.setFlags(
            (root.flags() & ~Qt.ItemFlag.ItemIsEditable)
            & ~Qt.ItemFlag.ItemIsDragEnabled
            & ~Qt.ItemFlag.ItemIsDropEnabled
        )
        root.setToolTip(0, "Workspace root")
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
                group_item.setFlags(
                    (group_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    | Qt.ItemFlag.ItemIsDropEnabled
                )
                group_item.setFlags(
                    group_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
                )
                group_item.setToolTip(0, "Condition group")
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
                        sample_item.setFlags(
                            (sample_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                            | Qt.ItemFlag.ItemIsDragEnabled
                        )
                        sample_item.setFlags(
                            sample_item.flags() & ~Qt.ItemFlag.ItemIsDropEnabled
                        )
                        sample_item.setToolTip(0, "Leaf image sample")
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

    def expand_condition_group(self, condition_group_id: str) -> None:
        root = self._workspace_root_item()
        if root is None:
            return
        for group_index in range(root.childCount()):
            group_item = root.child(group_index)
            if group_item is None:
                continue
            if self.item_node_id(group_item) == condition_group_id:
                group_item.setExpanded(True)
                root.setExpanded(True)
                return

    def item_node_type(self, item: QTreeWidgetItem | None) -> str | None:
        if item is None:
            return None
        value = item.data(0, NODE_TYPE_ROLE)
        return str(value) if value else None

    def item_node_id(self, item: QTreeWidgetItem | None) -> str | None:
        if item is None:
            return None
        value = item.data(0, NODE_ID_ROLE)
        return str(value) if value else None

    def _workspace_root_item(self) -> QTreeWidgetItem | None:
        if self.topLevelItemCount() == 0:
            return None
        root = self.topLevelItem(0)
        if root is not None and self.item_node_type(root) == "workspace":
            return root
        return None

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

    def _sample_id_from_mime(self, mime_data) -> str | None:
        if mime_data is None:
            return None
        if mime_data.hasFormat(MIME_SAMPLE_ID):
            payload = bytes(mime_data.data(MIME_SAMPLE_ID)).decode("utf-8")
            return payload or None
        return None

    def _drop_target_group_id(self, item: QTreeWidgetItem | None) -> str | None:
        if item is None:
            return None
        node_type = self.item_node_type(item)
        if node_type == "condition_group":
            return self.item_node_id(item)
        return None

    def accepts_sample_drop_on_item(self, item: QTreeWidgetItem | None) -> bool:
        """Return True when ``item`` is a valid sample drop target."""
        return self._drop_target_group_id(item) is not None

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

    def _show_context_menu(self, position: QPoint) -> None:
        item = self.itemAt(position)
        if item is None:
            if self._workspace_root_item() is not None:
                menu = QMenu(self)
                self._add_empty_space_menu_actions(menu)
                if not menu.isEmpty():
                    menu.exec(self.viewport().mapToGlobal(position))
            return

        self.setCurrentItem(item)
        node_type = self.item_node_type(item)
        node_id = self.item_node_id(item)
        if node_type is None or node_id is None:
            return

        menu = QMenu(self)
        if node_type == "workspace":
            self._add_workspace_menu_actions(menu)
        elif node_type == "condition_group":
            self._add_condition_group_menu_actions(menu, node_id)
        elif node_type == "sample":
            self._add_sample_menu_actions(menu, node_id)
        else:
            return

        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(position))

    def _add_empty_space_menu_actions(self, menu: QMenu) -> None:
        action = QAction("New Condition Group", self)
        action.setToolTip("Create a new condition group in this workspace")
        action.triggered.connect(self.new_condition_group_requested.emit)
        menu.addAction(action)

    def _add_workspace_menu_actions(self, menu: QMenu) -> None:
        action = QAction("Open Workspace Folder", self)
        action.setToolTip("Reveal the workspace folder in Finder or File Explorer")
        action.triggered.connect(self.open_workspace_folder_requested.emit)
        menu.addAction(action)

    def _add_condition_group_menu_actions(self, menu: QMenu, group_id: str) -> None:
        action = QAction("Import Image into Group", self)
        action.setToolTip("Import images into this condition group")
        action.triggered.connect(
            lambda: self.import_images_into_group_requested.emit(group_id)
        )
        menu.addAction(action)

        action = QAction("Rename Condition Group", self)
        action.setToolTip("Change the display name of this condition group")
        action.triggered.connect(
            lambda: self.rename_condition_group_requested.emit(group_id)
        )
        menu.addAction(action)

        action = QAction("Remove Condition Group", self)
        action.setToolTip("Remove this condition group and its samples from the workspace")
        action.triggered.connect(
            lambda: self.remove_condition_group_requested.emit(group_id)
        )
        menu.addAction(action)

    def _add_sample_menu_actions(self, menu: QMenu, sample_id: str) -> None:
        action = QAction("Select / Preview Sample", self)
        action.setToolTip("Load this sample image in the workbench")
        action.triggered.connect(lambda: self.sample_preview_requested.emit(sample_id))
        menu.addAction(action)

        action = QAction("Rename Sample", self)
        action.setToolTip("Change the display name of this sample")
        action.triggered.connect(lambda: self.rename_sample_requested.emit(sample_id))
        menu.addAction(action)

        action = QAction("Remove Sample", self)
        action.setToolTip("Remove this sample from the workspace")
        action.triggered.connect(lambda: self.remove_sample_requested.emit(sample_id))
        menu.addAction(action)

        reveal_label = reveal_image_menu_label()
        action = QAction(reveal_label, self)
        action.setToolTip("Show the managed workspace image file in the system file manager")
        action.triggered.connect(lambda: self.reveal_sample_image_requested.emit(sample_id))
        menu.addAction(action)

    def _set_drop_highlight(self, item: QTreeWidgetItem | None) -> None:
        if item is self._drop_highlight_item:
            return
        self._clear_drop_highlight()
        if item is None:
            return
        self._drop_highlight_item = item
        item.setData(0, DROP_HIGHLIGHT_ROLE, item.background(0))
        item.setBackground(0, self.palette().highlight())

    def _clear_drop_highlight(self) -> None:
        if self._drop_highlight_item is None:
            return
        previous = self._drop_highlight_item.data(0, DROP_HIGHLIGHT_ROLE)
        if previous is not None:
            self._drop_highlight_item.setBackground(0, previous)
        else:
            self._drop_highlight_item.setBackground(0, self.palette().base())
        self._drop_highlight_item = None

    def startDrag(self, supportedActions) -> None:  # noqa: N802 - Qt API
        item = self.currentItem()
        if item is None or self.item_node_type(item) != "sample":
            return

        sample_id = self.item_node_id(item)
        if not sample_id:
            return

        from PySide6.QtCore import QMimeData

        mime_data = QMimeData()
        mime_data.setData(MIME_SAMPLE_ID, sample_id.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime_data)
        drag.exec(Qt.DropAction.MoveAction)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._sample_id_from_mime(event.mimeData()) is None:
            event.ignore()
            return
        if self.accepts_sample_drop_on_item(self.itemAt(event.position().toPoint())):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._sample_id_from_mime(event.mimeData()) is None:
            event.ignore()
            self._clear_drop_highlight()
            return

        item = self.itemAt(event.position().toPoint())
        if self.accepts_sample_drop_on_item(item):
            event.acceptProposedAction()
            self._set_drop_highlight(item)
            return

        event.ignore()
        self._clear_drop_highlight()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._clear_drop_highlight()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._clear_drop_highlight()
        sample_id = self._sample_id_from_mime(event.mimeData())
        destination_group_id = self._drop_target_group_id(
            self.itemAt(event.position().toPoint())
        )
        if sample_id is None or destination_group_id is None:
            event.ignore()
            return

        self.sample_move_requested.emit(sample_id, destination_group_id)
        event.acceptProposedAction()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        item = self.itemAt(event.position().toPoint())
        if item is None:
            self.clearSelection()
            self.setCurrentItem(None)
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            item = self.currentItem()
            if item is not None and self.item_node_type(item) == "sample":
                sample_id = self.item_node_id(item)
                if sample_id:
                    self.sample_preview_requested.emit(sample_id)
                    event.accept()
                    return
        super().keyPressEvent(event)


def build_explorer_panel(parent: QWidget | None = None) -> tuple[QWidget, ExplorerTreeWidget]:
    """Build the left Explorer panel shell."""
    panel = QWidget(parent)
    apply_explorer_panel_style(panel)
    panel.setAccessibleName("Explorer panel")

    layout = QVBoxLayout(panel)
    layout.setContentsMargins(4, 4, 4, 4)
    layout.setSpacing(0)

    tree = ExplorerTreeWidget(panel)
    layout.addWidget(tree, stretch=1)
    return panel, tree
