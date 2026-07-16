# NeoPest Assisted Auto-Calibration Research (AC-0)

This directory contains **isolated research code** for future ruler-based automatic
calibration proposals. It is **not** part of the production NeoPest application
package and must not modify researcher-facing GUI workflows.

## Scientific principles

- **Manual calibration remains the reference standard** for NeoPest science workflows.
- **Automatic output is a proposal (suggestion), not an applied calibration.**
- The prototype **fails closed**: when confidence or geometry is insufficient, it
  returns `NO_SUGGESTION` or `ERROR` with an explicit `failure_reason`.
- **OCR is not required** for this research track.
- Algorithms must be **deterministic** and produce **inspectable diagnostics**.

## Anticipated pipeline stages

Future phases are expected to implement, in order:

1. Ruler localization
2. Perspective correction
3. Tick detection
4. One-inch interval proposals
5. Confidence evaluation
6. Comparison against manual references

AC-0 provides the experiment harness only. The placeholder pipeline does **not**
detect rulers and does not emit fake successful calibrations.

## Package layout

```
research/auto_calibration/
  config.py            # Immutable configuration and algorithm versioning
  models.py            # Result schemas and scale math helpers
  manual_reference.py  # Manual 1-inch reference JSON format
  pipeline.py          # Placeholder proposal pipeline
  evaluation.py        # Reference comparison and aggregate metrics
  visualization.py     # Diagnostic overlays (does not affect numerics)
  cli.py               # Scriptable CLI entry point
```

## Algorithm versioning

- `algorithm_name`: `neopest_assisted_autocal`
- `algorithm_version`: `0.1.0-prototype`
- `schema_version`: `1.0`

Increment `algorithm_version` when any of the following change in a way that
affects numerical or diagnostic outputs:

- ruler localization logic
- perspective model
- tick detection or spacing rules
- confidence combination rules
- default thresholds that gate proposals

Documentation-only or visualization-only changes do not require a version bump.

## Result JSON (`AutoCalibrationResult`)

Key fields:

| Field | Description |
| --- | --- |
| `status` | `SUCCESS`, `NO_SUGGESTION`, or `ERROR` |
| `failure_reason` | Explicit enum-like reason (`NONE`, `NO_RULER_CANDIDATE`, …) |
| `proposed_endpoints_px` | Suggested 1-inch segment (null when no proposal) |
| `proposed_pixels_per_inch` / `proposed_pixels_per_mm` | Proposed scale (null when no proposal) |
| `overall_confidence` | Reserved; not scientifically meaningful in AC-0 |
| `component_scores` | Per-stage scores; unavailable components are `null` |
| `diagnostics` | ROI, ticks, residuals, warning codes (extensible) |
| `runtime` | Optional elapsed seconds (excluded from canonical comparisons) |

AC-0 placeholder results use `NO_SUGGESTION` with `NO_RULER_CANDIDATE`.

## Manual reference JSON

Ground-truth annotations for evaluation (not production `CalibrationRecord` objects):

- `endpoint_a_px`, `endpoint_b_px` — 1-inch segment endpoints
- `reference_pixels_per_inch`, `reference_pixels_per_mm` — derived from Euclidean
  distance (1 inch = 25.4 mm)
- `annotation_method`, `annotator`, `annotated_at`, optional `notes`

### Measurement vs artifact determinism

- **Measurement determinism:** identical endpoints always yield the same derived
  `reference_pixels_per_inch` and `reference_pixels_per_mm` (see
  `ManualReference.to_measurement_dict()`).
- **Byte-identical artifact determinism:** full JSON may differ when
  `annotated_at` is auto-generated. Use `--annotated-at` on `make-reference` for
  reproducible fixture files. Provenance fields are excluded from measurement
  comparisons.

## CLI examples

From the repository root:

```bash
python -m research.auto_calibration.cli run \
  --image path/to/image.jpg \
  --output-dir path/to/output

python -m research.auto_calibration.cli batch \
  --images-dir path/to/images \
  --output-dir path/to/output

python -m research.auto_calibration.cli make-reference \
  --image path/to/image.jpg \
  --x1 100 --y1 200 --x2 550 --y2 205 \
  --output path/to/reference.json

python -m research.auto_calibration.cli evaluate \
  --results-dir path/to/results \
  --references-dir path/to/references \
  --output path/to/evaluation.json
```

Add `--write-overlay` to `run` or `batch` for diagnostic PNG overlays.

## Deterministic behavior

- Batch processing sorts images by filename (case-insensitive).
- Result JSON uses sorted keys and stable indentation.
- Canonical equality comparisons exclude `runtime` metadata.
- Repeated runs on the same image and configuration produce identical canonical
  payloads when `capture_runtime=False`.

## Local validation data (not committed)

Keep private or large images outside Git. Suggested ignored layout:

```
research_data/
  auto_calibration/
    images/
    references/
    results/
    overlays/
    reports/
```

`research_data/` is listed in `.gitignore`.

## Current limitations (AC-0)

- No ruler localization, perspective correction, or tick detection
- No overall confidence scoring
- No GUI integration
- No automatic application of scale to NeoPest workspaces
- No OCR or machine-learning dependencies

## Running tests

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Research-specific tests are prefixed with `test_auto_calibration_`.
