"""Tests for workspace and sample metadata dataclasses."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neopest import __version__
from neopest.metadata import (
    DEFAULT_CONDITION_GROUP_NAME,
    SCHEMA_VERSION,
    WORKSPACE_JSON_FILENAME,
    ConditionGroupRecord,
    SampleRecord,
    WorkspaceMetadata,
)
from neopest.project_manager import (
    ANALYSIS_DIR,
    EXPORTS_DIR,
    RAW_IMAGES_DIR,
    WORKSPACE_DIRS,
    ensure_default_condition_group,
    ensure_workspace_layout,
    initialize_workspace,
    load_workspace,
    save_workspace,
)
from neopest.results import AnalysisStatus, CalibrationStatus


class TestMetadata(unittest.TestCase):
    def test_condition_group_record_create_uses_stable_id_prefix(self) -> None:
        group = ConditionGroupRecord.create(display_name="WT Control")
        self.assertTrue(group.condition_group_id.startswith("condition_group_"))
        self.assertEqual(group.display_name, "WT Control")
        self.assertTrue(group.created_at.endswith("+00:00"))

    def test_sample_record_create_uses_stable_id_prefix(self) -> None:
        group = ConditionGroupRecord.create(display_name="WT Control")
        record = SampleRecord.create(
            condition_group_id=group.condition_group_id,
            display_name="leaf_01.jpg",
            raw_image_path="raw_images/group/leaf_01.jpg",
            original_source_path="/tmp/leaf_01.jpg",
        )
        self.assertTrue(record.sample_id.startswith("sample_"))
        self.assertEqual(record.condition_group_id, group.condition_group_id)
        self.assertEqual(record.display_name, "leaf_01.jpg")
        self.assertTrue(record.imported_at.endswith("+00:00"))
        self.assertEqual(record.calibration_status, CalibrationStatus.UNCALIBRATED)
        self.assertEqual(record.analysis_status, AnalysisStatus.NOT_RUN)

    def test_workspace_metadata_create(self) -> None:
        with TemporaryDirectory() as tmp:
            ws = WorkspaceMetadata.create(Path(tmp), display_name="Test Lab")
            self.assertTrue(ws.workspace_id.startswith("workspace_"))
            self.assertEqual(ws.root_path, Path(tmp).resolve())
            self.assertEqual(ws.display_name, "Test Lab")
            self.assertEqual(ws.condition_groups, [])
            self.assertEqual(ws.samples, [])

    def test_ensure_workspace_layout_creates_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "my_workspace"
            ensure_workspace_layout(root)
            for dirname in WORKSPACE_DIRS:
                self.assertTrue((root / dirname).is_dir())
            self.assertEqual(WORKSPACE_DIRS, (RAW_IMAGES_DIR, ANALYSIS_DIR, EXPORTS_DIR))

    def test_workspace_round_trip_preserves_stable_ids(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Round Trip")
            group_id = workspace.condition_groups[0].condition_group_id
            workspace_id = workspace.workspace_id

            loaded = load_workspace(root)
            self.assertEqual(loaded.workspace_id, workspace_id)
            self.assertEqual(len(loaded.condition_groups), 1)
            self.assertEqual(loaded.condition_groups[0].condition_group_id, group_id)
            self.assertEqual(loaded.condition_groups[0].display_name, DEFAULT_CONDITION_GROUP_NAME)

            json_path = root / WORKSPACE_JSON_FILENAME
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], SCHEMA_VERSION)
            self.assertEqual(payload["app_version"], __version__)
            self.assertEqual(payload["workspace_id"], workspace_id)

    def test_save_workspace_updates_timestamp(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            workspace = initialize_workspace(root, display_name="Timestamp")
            original_updated = workspace.updated_at
            workspace.display_name = "Renamed"
            save_workspace(workspace)
            loaded = load_workspace(root)
            self.assertEqual(loaded.display_name, "Renamed")
            self.assertGreaterEqual(loaded.updated_at, original_updated)

    def test_ensure_default_condition_group_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            first_id = ensure_default_condition_group(workspace, save=False)
            second_id = ensure_default_condition_group(workspace, save=False)
            self.assertEqual(first_id, second_id)
            self.assertEqual(len(workspace.condition_groups), 1)


if __name__ == "__main__":
    unittest.main()
