# NeoPest

NeoPest is a researcher-focused desktop application for analyzing still images of strawberry leaves to quantify total leaf area, diseased/necrotic area, and percent disease.

## Scientific scope

Each image contains one strawberry leaf inside a petri dish with a blue interior background, a white table surface outside the dish, and a visible ruler for scale calibration. Disease and necrosis appear as brown regions, often starting near the leaf center.

The analysis approach is **traditional computer vision first** — no machine learning in the initial phases. Scientific algorithms live in pure Python modules separate from the PySide6 GUI.

## Current status (Phase 1)

Phase 1 implements **workspace metadata and still-image import/preview**:

- Create or open a NeoPest workspace folder with human-readable JSON metadata (`neopest_workspace.json`)
- Default **Ungrouped** condition group created on workspace initialization
- Import `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff` still images as samples under a condition group
- Copy imported images into workspace-managed `raw_images/` with provenance metadata
- Explorer shows **Workspace → Condition Group → Sample**
- Selecting a sample previews the raw image in the center canvas (fit-to-view, aspect ratio preserved)
- Inspector shows sample metadata (paths, dimensions, hash, calibration/analysis status)
- **Run Analysis** remains disabled — no scientific pipeline yet

### Current limitations

- No ruler calibration
- No leaf or disease segmentation
- No leaf area, disease area, or percent disease measurements
- No export/reporting workflow yet
- Analysis does not run automatically when a sample is selected

NeoPest is inspired by the ActinTrackCV workbench philosophy but is a **standalone project** with no dependency on ActinTrackCV.

## Requirements

- Python 3.10+
- PySide6

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

### Manual workflow (Phase 1)

1. Launch the app.
2. Use **File → New Workspace…** or **File → Open Workspace…** to choose a folder.
3. Use **File → Import Images…** to add still leaf images (assigned to the default condition group).
4. Optionally use **File → New Condition Group…** before importing to organize samples.
5. Select a sample in the Explorer to preview the raw image and view metadata in the Inspector.

## Run tests

```bash
python -m compileall neopest tests
python -m unittest discover -s tests -v
```

## Workspace layout

```
my_workspace/
  neopest_workspace.json    # workspace metadata (stable IDs, samples, condition groups)
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
