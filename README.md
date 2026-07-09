# NeoPest

NeoPest is a researcher-focused desktop application for analyzing still images of strawberry leaves to quantify total leaf area, diseased/necrotic area, and percent disease.

## Scientific scope

Each image contains one strawberry leaf inside a petri dish with a blue interior background, a white table surface outside the dish, and a visible ruler for scale calibration. Disease and necrosis appear as brown regions, often starting near the leaf center.

**Experimental context:** Inoculum is typically applied to the central leaf region; the same experimental unit is imaged over multiple days to observe disease spread. Phase 3 leaf mask preview treats central brown/dark tissue as part of the leaf blade (inoculation/necrosis within the leaf), not as background or a separate disease mask. Time-course metrics, disease segmentation, and spread analysis are deferred to later phases.

The analysis approach is **traditional computer vision first** — no machine learning in the initial phases. Scientific algorithms live in pure Python modules separate from the PySide6 GUI.

## Current status (Phase 3)

Phase 3 adds a **deterministic leaf mask preview** on top of Phase 1 workspace/import/preview and Phase 2 manual calibration:

- Create or open a NeoPest workspace folder with human-readable JSON metadata (`neopest_workspace.json`)
- **Default workspace** at `~/Documents/NeoPest` is created or loaded automatically on app startup
- Import still images as samples and preview raw images in the workbench
- **Manual calibration** for a selected sample:
  - click two ruler marks exactly **1 inch** apart on the raw image
  - click **Save 1-Inch Calibration** (no distance/unit prompt)
  - compute and persist internal `pixels_per_mm` from a fixed 25.4 mm interval
  - display the saved calibration line when the sample is reselected
- **Leaf mask preview** for a selected sample:
  - click **Preview Leaf Mask** in the Inspector
  - generate a deterministic color-threshold leaf blade mask in memory
  - show a semi-transparent green overlay on the raw image canvas
  - Inspector reports preview status, method, and warnings
- Inspector shows calibration status, scale, entered distance, converted mm distance, and point summary
- **Run Analysis** remains disabled — no quantitative pipeline yet

### Current limitations

- Leaf mask preview is **preview only** — not final analysis
- No disease/necrosis segmentation (brown central tissue stays inside leaf mask only)
- No time-course / multi-day spread metrics
- No leaf area, disease area, or percent disease measurements
- No saved analysis artifacts or export/reporting workflow yet
- No automatic ruler detection
- Analysis does not run automatically when a sample is selected

NeoPest is inspired by the ActinTrackCV workbench philosophy but is a **standalone project** with no dependency on ActinTrackCV.

## Requirements

- Python 3.10+
- PySide6
- numpy
- opencv-python-headless (deterministic color segmentation and morphology)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the app

```bash
python run_app.py
```

Or:

```bash
chmod +x run_app.sh
./run_app.sh
```

### Manual workflow (Phase 3)

1. Launch the app — the default workspace at `~/Documents/NeoPest` opens automatically.
2. Import still leaf images and select a sample in the Explorer.
3. Optionally set manual 1-inch calibration (Phase 2 workflow).
4. Click **Preview Leaf Mask** in the Inspector.
5. Confirm the canvas shows the raw image with a green leaf mask overlay.
6. Confirm the Inspector shows **Preview Ready** (or **Failed Preview** with warnings).
7. Confirm **Run Analysis** remains disabled and no measurement outputs are created.

## Run tests

```bash
python -m compileall neopest tests
python -m unittest discover -s tests -v
```

## Workspace layout

```
my_workspace/
  neopest_workspace.json    # workspace metadata (stable IDs, samples, calibration)
  raw_images/                 # imported still images, grouped by condition_group_id
  analysis/                   # reserved for future analysis outputs
  exports/                    # reserved for future export outputs
```

## Project layout

```
neopest/           Python package
  gui.py           MainWindow orchestration
  gui_layout_builders.py
  gui_styles.py
  gui_canvas.py
  explorer_tree.py
  metadata.py
  project_manager.py
  image_io.py
  calibration.py
  segmentation.py
  analysis.py
  results.py
  export.py
tests/             Unit tests (no visible GUI launch required)
run_app.py         Entry script
```
