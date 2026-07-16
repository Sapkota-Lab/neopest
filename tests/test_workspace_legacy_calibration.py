"""Tests for legacy Phase 3F inherited-calibration workspace repair."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from neopest.metadata import WORKSPACE_JSON_FILENAME, WorkspaceMetadata
from neopest.project_manager import import_images, initialize_workspace, load_workspace, save_workspace
from neopest.results import CalibrationStatus
from neopest.workspace_compat import (
    LEGACY_INHERITED_CALIBRATION_REMOVED_REASON,
    legacy_inherited_calibration_detected,
    legacy_inherited_calibration_repair_message,
    repair_legacy_inherited_calibrations_in_workspace_data,
    repair_legacy_inherited_sample_dict,
)


def _manual_calibration_payload() -> dict:
    return {
        "calibration_method": "manual",
        "calibration_origin": "manual",
        "pixels_per_mm": 8.0,
        "known_distance_mm": 25.4,
        "known_distance_value": 1.0,
        "known_distance_unit": "in",
        "point_1_x": 0.0,
        "point_1_y": 0.0,
        "point_2_x": 50.0,
        "point_2_y": 0.0,
        "source_sample_id": None,
        "source_condition_group_id": None,
        "calibrated_at": "2026-01-01T00:00:00+00:00",
        "notes": "",
    }


def _legacy_phase_3f_sample(**overrides: object) -> dict:
    payload = {
        "sample_id": "sample_legacy01",
        "condition_group_id": "condition_group_test",
        "display_name": "Inherited Leaf",
        "raw_image_path": "raw_images/group/leaf.png",
        "original_source_path": "/tmp/leaf.png",
        "width_px": 400,
        "height_px": 300,
        "image_hash": "sha256:legacy",
        "imported_at": "2026-01-01T00:00:00+00:00",
        "notes": "legacy note",
        "calibration_status": "inherited",
        "analysis_status": "not_run",
        "calibration": {
            "calibration_method": "inherited",
            "calibration_origin": "inherited",
            "pixels_per_mm": 7.5,
            "known_distance_mm": 25.4,
            "known_distance_value": 1.0,
            "known_distance_unit": "in",
            "point_1_x": None,
            "point_1_y": None,
            "point_2_x": None,
            "point_2_y": None,
            "source_sample_id": "sample_source01",
            "source_condition_group_id": "condition_group_test",
            "calibrated_at": "2026-01-01T00:00:00+00:00",
            "notes": "inherited from source",
        },
        "leaf_measurement": {
            "status": "valid",
            "leaf_area_pixels": 1200,
            "leaf_area_in2": 1.23,
            "leaf_area_mm2": 793.0,
            "pixels_per_mm_used": 7.5,
            "pixels_per_inch_used": 190.5,
            "leaf_mask_method": "green_seeded",
            "leaf_mask_settings_version": 1,
            "calculated_at": "2026-01-01T00:00:00+00:00",
            "source_image_hash": "sha256:legacy",
        },
    }
    payload.update(overrides)
    return payload


def _workspace_payload(*samples: dict) -> dict:
    return {
        "schema_version": 1,
        "app_version": "0.4.0",
        "workspace_id": "workspace_test",
        "display_name": "Legacy Repair Test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "condition_groups": [
            {
                "condition_group_id": "condition_group_test",
                "display_name": "Group A",
                "created_at": "2026-01-01T00:00:00+00:00",
                "notes": "",
            }
        ],
        "samples": list(samples),
    }


def _write_workspace(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / WORKSPACE_JSON_FILENAME
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


class TestLegacyInheritedCalibrationRepair(unittest.TestCase):
    def test_repair_message_uses_singular_grammar(self) -> None:
        self.assertEqual(
            legacy_inherited_calibration_repair_message(1),
            (
                "Reset 1 legacy inherited calibration. "
                "This sample requires manual recalibration."
            ),
        )

    def test_repair_message_uses_plural_grammar(self) -> None:
        self.assertEqual(
            legacy_inherited_calibration_repair_message(3),
            (
                "Reset 3 legacy inherited calibrations. "
                "These samples require manual recalibration."
            ),
        )

    def test_repair_counter_defaults_to_zero(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = WorkspaceMetadata.create(Path(tmp) / "ws")
            self.assertEqual(workspace.legacy_inherited_calibration_repairs, 0)

    def test_repair_counter_is_excluded_from_to_dict(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            workspace = load_workspace(root)
            payload = workspace.to_dict(app_version="0.4.0")
            self.assertNotIn("legacy_inherited_calibration_repairs", payload)

    def test_repair_counter_is_not_written_to_workspace_json(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            metadata_path = _write_workspace(
                root,
                _workspace_payload(_legacy_phase_3f_sample()),
            )
            load_workspace(root)
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertNotIn("legacy_inherited_calibration_repairs", saved)

    def test_repair_counter_does_not_carry_across_unrelated_workspace_loads(self) -> None:
        with TemporaryDirectory() as tmp:
            legacy_root = Path(tmp) / "legacy_ws"
            clean_root = Path(tmp) / "clean_ws"
            _write_workspace(legacy_root, _workspace_payload(_legacy_phase_3f_sample()))
            initialize_workspace(clean_root, display_name="Clean")
            repaired = load_workspace(legacy_root)
            self.assertEqual(repaired.legacy_inherited_calibration_repairs, 1)
            clean = load_workspace(clean_root)
            self.assertEqual(clean.legacy_inherited_calibration_repairs, 0)

    def test_contradictory_manual_status_with_inherited_provenance_is_repaired(self) -> None:
        """manual status + inherited provenance is treated as rejected inherited calibration."""
        contradictory = _legacy_phase_3f_sample(
            sample_id="sample_contra01",
            calibration_status="manual",
            calibration={
                "calibration_method": "inherited",
                "calibration_origin": "inherited",
                "pixels_per_mm": 7.5,
                "known_distance_mm": 25.4,
                "known_distance_value": 1.0,
                "known_distance_unit": "in",
                "point_1_x": None,
                "point_1_y": None,
                "point_2_x": None,
                "point_2_y": None,
                "source_sample_id": "sample_source01",
                "source_condition_group_id": "condition_group_test",
                "calibrated_at": "2026-01-01T00:00:00+00:00",
                "notes": "",
            },
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(contradictory))
            workspace = load_workspace(root)
            self.assertEqual(workspace.legacy_inherited_calibration_repairs, 1)
            sample = workspace.get_sample("sample_contra01")
            assert sample is not None
            self.assertEqual(sample.calibration_status, CalibrationStatus.UNCALIBRATED)
            self.assertIsNone(sample.calibration)

    def test_stale_measurement_from_repair_is_not_valid_for_display(self) -> None:
        from neopest.measurements import phase5_result_status_display
        from neopest.measurements import leaf_measurement_matches_provenance

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            sample = load_workspace(root).get_sample("sample_legacy01")
            assert sample is not None
            assert sample.leaf_measurement is not None
            self.assertEqual(sample.leaf_measurement.status, "stale")
            self.assertFalse(
                leaf_measurement_matches_provenance(
                    sample.leaf_measurement,
                    calibration=sample.calibration,
                    image_hash=sample.image_hash,
                )
            )
            self.assertEqual(phase5_result_status_display(sample), "Calibration required")

    def test_repair_save_failure_preserves_original_workspace_and_raises(self) -> None:
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            metadata_path = _write_workspace(
                root,
                _workspace_payload(_legacy_phase_3f_sample()),
            )
            original = metadata_path.read_text(encoding="utf-8")
            with patch(
                "neopest.project_manager.save_workspace",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaises(OSError):
                    load_workspace(root)
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), original)
            self.assertIn('"inherited"', original)

    def test_save_workspace_uses_temporary_file_before_replace(self) -> None:
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            workspace = initialize_workspace(Path(tmp) / "ws")
            target = workspace.workspace_json_path()
            with patch("neopest.project_manager.os.replace") as replace:
                save_workspace(workspace)
            replace.assert_called_once()
            temporary, destination = replace.call_args.args
            self.assertEqual(temporary.name, f"{target.name}.tmp")
            self.assertEqual(destination, target)

    def test_legacy_inherited_workspace_loads_without_crashing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            workspace = load_workspace(root)
            self.assertEqual(len(workspace.samples), 1)

    def test_legacy_inherited_sample_becomes_uncalibrated(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            sample = load_workspace(root).get_sample("sample_legacy01")
            self.assertIsNotNone(sample)
            assert sample is not None
            self.assertEqual(sample.calibration_status, CalibrationStatus.UNCALIBRATED)

    def test_legacy_inherited_scale_values_are_cleared(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            sample = load_workspace(root).get_sample("sample_legacy01")
            assert sample is not None
            self.assertIsNone(sample.calibration)

    def test_legacy_inherited_provenance_fields_are_cleared(self) -> None:
        repaired = repair_legacy_inherited_sample_dict(_legacy_phase_3f_sample())
        self.assertIsNone(repaired["calibration"])
        self.assertNotIn("source_sample_id", repaired)

    def test_calibration_dependent_leaf_measurement_becomes_stale(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            sample = load_workspace(root).get_sample("sample_legacy01")
            assert sample is not None
            self.assertIsNotNone(sample.leaf_measurement)
            assert sample.leaf_measurement is not None
            self.assertEqual(sample.leaf_measurement.status, "stale")
            self.assertEqual(
                sample.leaf_measurement.stale_reason,
                LEGACY_INHERITED_CALIBRATION_REMOVED_REASON,
            )
            self.assertEqual(sample.leaf_measurement.leaf_area_in2, 1.23)

    def test_unrelated_sample_metadata_is_preserved(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            sample = load_workspace(root).get_sample("sample_legacy01")
            assert sample is not None
            self.assertEqual(sample.sample_id, "sample_legacy01")
            self.assertEqual(sample.condition_group_id, "condition_group_test")
            self.assertEqual(sample.raw_image_path, "raw_images/group/leaf.png")
            self.assertEqual(sample.display_name, "Inherited Leaf")
            self.assertEqual(sample.image_hash, "sha256:legacy")
            self.assertEqual(sample.notes, "legacy note")

    def test_manual_calibration_sample_remains_unchanged(self) -> None:
        manual = _legacy_phase_3f_sample(
            sample_id="sample_manual01",
            display_name="Manual Leaf",
            calibration_status="manual",
            calibration=_manual_calibration_payload(),
            leaf_measurement=None,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(
                root,
                _workspace_payload(_legacy_phase_3f_sample(), manual),
            )
            workspace = load_workspace(root)
            self.assertEqual(workspace.legacy_inherited_calibration_repairs, 1)
            manual_sample = workspace.get_sample("sample_manual01")
            assert manual_sample is not None
            self.assertEqual(manual_sample.calibration_status, CalibrationStatus.MANUAL)
            self.assertIsNotNone(manual_sample.calibration)
            assert manual_sample.calibration is not None
            self.assertAlmostEqual(manual_sample.calibration.pixels_per_mm, 8.0)

    def test_uncalibrated_sample_remains_unchanged(self) -> None:
        uncalibrated = _legacy_phase_3f_sample(
            sample_id="sample_uncal01",
            display_name="Uncalibrated Leaf",
            calibration_status="uncalibrated",
            calibration=None,
            leaf_measurement=None,
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(
                root,
                _workspace_payload(_legacy_phase_3f_sample(), uncalibrated),
            )
            workspace = load_workspace(root)
            sample = workspace.get_sample("sample_uncal01")
            assert sample is not None
            self.assertEqual(sample.calibration_status, CalibrationStatus.UNCALIBRATED)
            self.assertIsNone(sample.calibration)

    def test_repair_is_deterministic(self) -> None:
        payload = _workspace_payload(_legacy_phase_3f_sample())
        first, count_a = repair_legacy_inherited_calibrations_in_workspace_data(payload)
        second, count_b = repair_legacy_inherited_calibrations_in_workspace_data(payload)
        self.assertEqual(count_a, 1)
        self.assertEqual(count_b, 1)
        self.assertEqual(first, second)

    def test_saved_workspace_no_longer_contains_inherited_state(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            metadata_path = _write_workspace(
                root,
                _workspace_payload(_legacy_phase_3f_sample()),
            )
            load_workspace(root)
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            serialized = json.dumps(saved)
            self.assertNotIn('"inherited"', serialized)
            sample_payload = saved["samples"][0]
            self.assertEqual(sample_payload["calibration_status"], "uncalibrated")
            self.assertIsNone(sample_payload["calibration"])
            self.assertEqual(sample_payload["leaf_measurement"]["status"], "stale")

    def test_second_load_does_not_repair_again(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            first = load_workspace(root)
            self.assertEqual(first.legacy_inherited_calibration_repairs, 1)
            second = load_workspace(root)
            self.assertEqual(second.legacy_inherited_calibration_repairs, 0)

    def test_inherited_enum_state_is_not_reintroduced(self) -> None:
        self.assertFalse(hasattr(CalibrationStatus, "INHERITED"))
        with self.assertRaises(ValueError):
            CalibrationStatus("inherited")

    def test_detection_by_calibration_origin_without_status(self) -> None:
        sample = _legacy_phase_3f_sample(calibration_status="uncalibrated")
        self.assertTrue(legacy_inherited_calibration_detected(sample))

    def test_real_imported_workspace_manual_path_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            from tests.test_calibration import _write_test_png
            from neopest.calibration import build_one_inch_manual_calibration_record
            from neopest.project_manager import save_sample_calibration

            source = Path(tmp) / "leaf.png"
            _write_test_png(source)
            workspace = initialize_workspace(Path(tmp) / "ws")
            imported = import_images(workspace, [source])
            save_sample_calibration(
                workspace,
                imported[0].sample_id,
                build_one_inch_manual_calibration_record((0.0, 0.0), (50.0, 0.0)),
            )
            loaded = load_workspace(workspace.root_path)
            sample = loaded.get_sample(imported[0].sample_id)
            assert sample is not None
            self.assertEqual(sample.calibration_status, CalibrationStatus.MANUAL)
            self.assertEqual(loaded.legacy_inherited_calibration_repairs, 0)

    def test_gui_repair_notice_uses_singular_wording(self) -> None:
        import os
        import sys
        from unittest.mock import patch

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox

        from neopest.gui import MainWindow
        from neopest.project_manager import WorkspaceError

        app = QApplication.instance() or QApplication(sys.argv)
        del app

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            workspace = load_workspace(root)
            with patch(
                "neopest.gui.ensure_default_workspace",
                side_effect=WorkspaceError("disabled"),
            ):
                with patch.object(QMessageBox, "warning"):
                    window = MainWindow()
            with patch.object(QMessageBox, "information") as info:
                window._workspace = workspace
                window._notify_legacy_inherited_calibration_repairs(workspace)
            message = info.call_args[0][2]
            self.assertEqual(
                message,
                legacy_inherited_calibration_repair_message(1),
            )

    def test_gui_repair_notice_uses_plural_wording(self) -> None:
        import os
        import sys
        from unittest.mock import patch

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox

        from neopest.gui import MainWindow
        from neopest.project_manager import WorkspaceError

        app = QApplication.instance() or QApplication(sys.argv)
        del app

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(
                root,
                _workspace_payload(
                    _legacy_phase_3f_sample(),
                    _legacy_phase_3f_sample(sample_id="sample_legacy02"),
                    _legacy_phase_3f_sample(sample_id="sample_legacy03"),
                ),
            )
            workspace = load_workspace(root)
            with patch(
                "neopest.gui.ensure_default_workspace",
                side_effect=WorkspaceError("disabled"),
            ):
                with patch.object(QMessageBox, "warning"):
                    window = MainWindow()
            with patch.object(QMessageBox, "information") as info:
                window._workspace = workspace
                window._notify_legacy_inherited_calibration_repairs(workspace)
            message = info.call_args[0][2]
            self.assertEqual(
                message,
                legacy_inherited_calibration_repair_message(3),
            )

    def test_gui_does_not_show_repair_notice_when_save_fails(self) -> None:
        import os
        import sys
        from unittest.mock import patch

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QMessageBox

        from neopest.gui import MainWindow
        from neopest.project_manager import WorkspaceError

        app = QApplication.instance() or QApplication(sys.argv)
        del app

        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            _write_workspace(root, _workspace_payload(_legacy_phase_3f_sample()))
            with patch(
                "neopest.gui.ensure_default_workspace",
                side_effect=WorkspaceError("disabled"),
            ):
                with patch.object(QMessageBox, "warning"):
                    window = MainWindow()
            with patch(
                "neopest.project_manager.save_workspace",
                side_effect=OSError("disk full"),
            ):
                with patch.object(QMessageBox, "information") as info:
                    with self.assertRaises(OSError):
                        load_workspace(root)
                    info.assert_not_called()


if __name__ == "__main__":
    unittest.main()
