# NeoPest

NeoPest is a researcher-focused desktop application for analyzing still images of strawberry leaves to quantify total leaf area, diseased/necrotic area, and percent disease.

## Scientific scope

Each image contains one strawberry leaf inside a petri dish with a blue interior background, a white table surface outside the dish, and a visible ruler for scale calibration. Disease and necrosis appear as brown regions, often starting near the leaf center.

The analysis approach is **traditional computer vision first** — no machine learning in the initial phases. Scientific algorithms live in pure Python modules separate from the PySide6 GUI.

## Current status (Phase 0)

This repository contains a **runnable project scaffold** only:

- Workbench shell: Explorer (left), image preview (center), inspector (right), status/control strip (bottom)
- Placeholder dataclasses and module boundaries for metadata, calibration, segmentation, analysis, and export
- No real leaf/disease segmentation or calibration yet

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

## Run tests

```bash
python -m compileall neopest tests
python -m unittest discover -s tests -v
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
tests/             Unit tests (no GUI launch required)
run_app.py         Entry script
```
