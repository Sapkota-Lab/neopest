"""Immutable configuration for the assisted auto-calibration research prototype."""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = "1.0"
ALGORITHM_NAME = "neopest_assisted_autocal"
ALGORITHM_VERSION = "0.1.0-prototype"

MM_PER_INCH = 25.4

# Reserved for future phases; not used by the AC-0 placeholder pipeline.
DEFAULT_MIN_OVERALL_CONFIDENCE = 0.0
DEFAULT_MIN_RULER_DETECTION_CONFIDENCE = 0.0
DEFAULT_MIN_TICK_DETECTION_CONFIDENCE = 0.0

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True, slots=True)
class AutoCalibrationConfig:
        """Validated, immutable prototype configuration."""

        schema_version: str = SCHEMA_VERSION
        algorithm_name: str = ALGORITHM_NAME
        algorithm_version: str = ALGORITHM_VERSION
        min_overall_confidence: float = DEFAULT_MIN_OVERALL_CONFIDENCE
        min_ruler_detection_confidence: float = DEFAULT_MIN_RULER_DETECTION_CONFIDENCE
        min_tick_detection_confidence: float = DEFAULT_MIN_TICK_DETECTION_CONFIDENCE
        supported_image_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS

        def __post_init__(self) -> None:
                if self.min_overall_confidence < 0.0 or self.min_overall_confidence > 1.0:
                        raise ValueError("min_overall_confidence must be in [0, 1].")
                if (
                        self.min_ruler_detection_confidence < 0.0
                        or self.min_ruler_detection_confidence > 1.0
                ):
                        raise ValueError("min_ruler_detection_confidence must be in [0, 1].")
                if (
                        self.min_tick_detection_confidence < 0.0
                        or self.min_tick_detection_confidence > 1.0
                ):
                        raise ValueError("min_tick_detection_confidence must be in [0, 1].")
                if not self.supported_image_extensions:
                        raise ValueError("supported_image_extensions must not be empty.")


DEFAULT_CONFIG = AutoCalibrationConfig()
