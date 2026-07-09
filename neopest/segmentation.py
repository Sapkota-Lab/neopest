"""Leaf and disease segmentation (scaffold)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SegmentationMasks:
    """Binary masks for leaf tissue and diseased/necrotic regions."""

    leaf_mask_path: Path | None
    disease_mask_path: Path | None


def segment_leaf_and_disease(*_args, **_kwargs) -> SegmentationMasks:
    """Separate leaf tissue from background and identify necrotic regions."""
    raise NotImplementedError("Leaf/disease segmentation is not implemented in Phase 0.")
