"""Tests for workspace and sample metadata dataclasses."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neopest.metadata import (
    ConditionGroupRecord,
    SampleRecord,
    WorkspaceMetadata,
)
from neopest.project_manager import WORKSPACE_DIRS, ensure_workspace_layout


class TestMetadata(unittest.TestCase):
    def test_condition_group_record_create_uses_stable_id_prefix(self) -> None:
        group = ConditionGroupRecord.create(display_name="WT Control")
        self.assertTrue(group.condition_group_id.startswith("condition_group_"))
        self.assertEqual(group.display_name, "WT Control")
        self.assertTrue(group.created_at.endswith("+00:00"))

    def test_sample_record_create_uses_stable_id_prefix(self) -> None:
        group = ConditionGroupRecord.create(display_name="WT Control")
        record = SampleRecord.create(
            Path("/tmp/leaf_01.jpg"),
            condition_group_id=group.condition_group_id,
        )
        self.assertTrue(record.sample_id.startswith("sample_"))
        self.assertEqual(record.condition_group_id, group.condition_group_id)
        self.assertEqual(record.display_name, "leaf_01.jpg")
        self.assertTrue(record.imported_at.endswith("+00:00"))

    def test_workspace_metadata_create(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = WorkspaceMetadata.create(Path(tmp))
            self.assertTrue(ws.workspace_id.startswith("workspace_"))
            self.assertEqual(ws.root_path, Path(tmp).resolve())
            self.assertEqual(ws.condition_groups, [])
            self.assertEqual(ws.samples, [])

    def test_ensure_workspace_layout_creates_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my_workspace"
            ws = ensure_workspace_layout(root)
            self.assertEqual(ws.root_path, root.resolve())
            for dirname in WORKSPACE_DIRS:
                self.assertTrue((root / dirname).is_dir())


if __name__ == "__main__":
    unittest.main()
