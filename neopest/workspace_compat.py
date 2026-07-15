"""Compatibility repairs for legacy workspace metadata."""

from __future__ import annotations

from typing import Any

from neopest.results import CalibrationStatus

LEGACY_INHERITED_CALIBRATION_STATUS = "inherited"
LEGACY_INHERITED_CALIBRATION_ORIGIN = "inherited"
LEGACY_INHERITED_CALIBRATION_METHOD = "inherited"
LEGACY_INHERITED_CALIBRATION_REMOVED_REASON = "legacy_inherited_calibration_removed"


def legacy_inherited_calibration_repair_message(repair_count: int) -> str:
    """Return user-facing text for a completed legacy calibration repair."""
    if repair_count == 1:
        return (
            "Reset 1 legacy inherited calibration. "
            "This sample requires manual recalibration."
        )
    return (
        f"Reset {repair_count} legacy inherited calibrations. "
        "These samples require manual recalibration."
    )


def legacy_inherited_calibration_detected(sample_data: dict[str, Any]) -> bool:
    """Return True when serialized sample data reflects rejected Phase 3F inheritance."""
    status = str(sample_data.get("calibration_status", "")).strip().lower()
    if status == LEGACY_INHERITED_CALIBRATION_STATUS:
        return True

    calibration = sample_data.get("calibration")
    if not isinstance(calibration, dict):
        return False

    origin = str(calibration.get("calibration_origin", "")).strip().lower()
    method = str(calibration.get("calibration_method", "")).strip().lower()
    return (
        origin == LEGACY_INHERITED_CALIBRATION_ORIGIN
        or method == LEGACY_INHERITED_CALIBRATION_METHOD
    )


def repair_legacy_inherited_sample_dict(sample_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize one legacy inherited sample dict to uncalibrated scientific state."""
    repaired = dict(sample_data)
    repaired["calibration_status"] = CalibrationStatus.UNCALIBRATED.value
    repaired["calibration"] = None

    measurement = repaired.get("leaf_measurement")
    if isinstance(measurement, dict) and str(measurement.get("status")) == "valid":
        stale_measurement = dict(measurement)
        stale_measurement["status"] = "stale"
        stale_measurement["stale_reason"] = LEGACY_INHERITED_CALIBRATION_REMOVED_REASON
        repaired["leaf_measurement"] = stale_measurement

    return repaired


def repair_legacy_inherited_calibrations_in_workspace_data(
    workspace_data: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    """Repair legacy inherited samples in raw workspace JSON data."""
    repaired_data = dict(workspace_data)
    samples = []
    repair_count = 0
    for item in workspace_data.get("samples", []):
        if not isinstance(item, dict):
            samples.append(item)
            continue
        if legacy_inherited_calibration_detected(item):
            samples.append(repair_legacy_inherited_sample_dict(item))
            repair_count += 1
        else:
            samples.append(item)
    repaired_data["samples"] = samples
    return repaired_data, repair_count
