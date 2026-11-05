# NeoPest

NeoPest is a computer-vision application for measuring strawberry leaf area and candidate brown/necrotic tissue from one leaf image at a time. Images are expected to show the leaf on a white moisture pad.

## NeoPest Interface

### Empty Workspace

![NeoPest empty workspace](docs/images/neopest-empty-workspace.png)

### Analysis Workspace

![NeoPest with imported samples, previews, and measurements](docs/images/neopest-analysis-workspace.png)

## Current Capabilities

- Manual image-specific 1-inch calibration
- Whole-leaf segmentation on a white moisture pad
- Candidate disease segmentation
- Researcher correction of the candidate-disease mask, saved separately from the automatic result
- Display-only preview modes (**View** dropdown)
- Persisted measurements (total leaf area, candidate disease area, coverage, non-candidate leaf area)
- Workspace and condition-group management, including file drops onto a condition group
- JPEG, PNG, and, where the Qt HEIC plugin is present, HEIC/HEIF import
- Shared orientation for the image, overlays, calibration, and corrections
- Zoom and pan
- A status-bar progress indicator while calculation or first-time correction preparation is running

## Current Limitations

- Candidate disease detection is not biologically validated. Outputs are estimates, not a diagnosis.
- Removing area from the candidate-disease mask excludes those pixels from the candidate measurement. It does not confirm that the tissue is healthy.
- Corrections cannot change the leaf boundary.
- One image at a time. There is no batch processing.
- No CSV or JSON export, and no saved mask-image export.
- No automatic calibration, and calibration is not copied from one image to another.
- Leaf segmentation expects a white moisture pad. Older blue-dish results do not match the current method and need **Recalculate Results**.
- White-pad behavior has had limited checking on real assay photographs. Do not treat synthetic test images as proof of measurement accuracy.
- HEIC/HEIF import depends on the Qt image plugin. It has not been verified on Windows. If the plugin is missing, import fails with a message that HEIC/HEIF decoding is not available. A corrupt file fails with “Unable to decode HEIC/HEIF image”.
- The progress bar shows the current stage. It does not show a percent complete.
- The packaged macOS build, when published, is an unsigned Apple Silicon app and is not notarized. Intel and Windows packages are not part of this release.

## Installation (macOS)

Verified on **macOS** with **Python 3.10 or newer**. A graphical display is required.

**Command Line Tools** (includes Git):

```bash
xcode-select --install
git --version
```

**Python** — install from [python.org](https://www.python.org/downloads/macos/), then verify:

```bash
python3 --version
pip3 --version
```

**Clone, dependencies, and launch:**

```bash
git clone https://github.com/Sapkota-Lab/neopest.git
cd neopest
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 run_app.py
```

Or: `./run_app.sh` (with `.venv` activated)

First launch creates workspace `~/Documents/NeoPest` with metadata file `neopest_workspace.json`.

| Problem | What to try |
|---------|-------------|
| `python3` or `git` not found | Install Python or run `xcode-select --install`; reopen Terminal. |
| `ModuleNotFoundError` | Run `source .venv/bin/activate` before launching. |
| Workspace not visible | Check `~/Documents/NeoPest` or **File → Open Workspace**. |

## Verify Installation

```bash
python3 --version
git --version
source .venv/bin/activate
python3 -m compileall neopest
python3 run_app.py
```

If NeoPest opens and `~/Documents/NeoPest` is available, installation is complete. A full test-suite run is not required for normal install verification.

## First Use

1. Launch NeoPest (`python3 run_app.py` with `.venv` activated).
2. Photograph or choose a leaf on a **white moisture pad**. A blue dish background is not the current segmentation target.
3. **Import** a leaf image (**File → Import Images**), or drop image files onto a condition-group row in the Explorer. Drops onto empty space or onto a sample row are ignored. JPEG and PNG are supported. HEIC/HEIF is supported only when this computer’s Qt build includes that plugin.

   ![Importing strawberry leaf images into NeoPest](docs/images/workflows/import.gif)

4. Condition groups are the top level of the Explorer. Create or select one (**File → New Condition Group** or right-click in Explorer).

   ![Creating a condition group and organizing samples with drag and drop](docs/images/workflows/new-condition-drag-drop.gif)

5. Select the sample and perform **manual 1-inch calibration** on the ruler. NeoPest applies the photograph’s orientation before display, so the image, masks, and calibration points share one grid. If a saved calibration was made in an older orientation, the status asks you to set the 1-inch calibration again. A scale change alone does not discard mask corrections; it recalculates areas.

   ![Setting an image-specific 1-inch calibration](docs/images/workflows/calibration.gif)

6. Click **Calculate Results** or **Recalculate Results**. The status bar shows a busy indicator and a short stage label, such as Preparing image, Finding leaf, Identifying candidate disease, or Updating measurements. The window stays usable. A second calculation is ignored until the current one finishes.

7. Use **View** to inspect masks (display only): Raw Image, Whole Leaf Mask, Candidate Disease Mask, Healthy Tissue Preview.

   ![Inspecting whole-leaf, candidate-disease, and healthy-tissue views](docs/images/workflows/mask-view.gif)

8. Optional: click **Correct disease mask**. **Add area** includes pixels in the candidate-disease measurement. **Remove area** excludes them and does not mark them healthy. Brush size is a radius in image pixels. **Undo** and **Redo** step through complete strokes. **Reset corrections** restores the automatic mask and can itself be undone. **Done** leaves correction mode and keeps the corrections through the normal save. Use **View** to hide the overlay without clearing edits. The automatic result remains stored separately. The Inspector labels a saved correction **Researcher-corrected**.

9. **Fit** fits the image in the view. **100%** shows one image pixel per screen pixel. The zoom percentage is shown beside those buttons. The scroll wheel or trackpad zooms at the pointer. Drag with the middle button, or hold Space and drag, to pan. Zoom and pan do not change measured pixels or place calibration points.

10. Review measurements in the Inspector (see below).
11. If status shows **Recalculation required**, click **Recalculate Results**. Results saved before the white-pad leaf method, or before an orientation change, need recalculation. Opening an old workspace does not by itself rewrite those results.

Back up `~/Documents/NeoPest` to preserve images and results.

## Measurements

| Measurement | Description |
|-------------|-------------|
| Total leaf area | Area of the segmented leaf |
| Candidate disease area | Brown/necrotic tissue identified by the deterministic algorithm, after any researcher correction |
| Candidate disease coverage | Candidate disease area relative to total leaf area |
| Non-candidate leaf area | Remaining segmented leaf area. This is not a confirmed-healthy measurement. |

Areas are in square inches (in²); coverage is a percent.

## Result Status

| Status | Meaning |
|--------|---------|
| Current | Automatic measurements match the current image, calibration, and segmentation method |
| Researcher-corrected | A researcher has edited the candidate-disease mask. The automatic result is still stored |
| Recalculation required | Inputs changed since the last calculation, including an older leaf-segmentation version or a previous image orientation |
| Calibration required | Manual calibration is required before calculation |
| Not calculated | Results have not been generated |
| Calculation failed | Segmentation was unsuccessful |

## Scientific Notes

- One image at a time. Calibration is manual and specific to each image.
- Deterministic computer vision, not machine learning.
- Candidate disease estimates brown/necrotic-looking tissue. It is **not** confirmed disease or a diagnosis.
- Researcher corrections change which pixels count as candidate disease. They do not replace the stored automatic measurement and do not change the automatic thresholds.
- If the image, orientation convention, or leaf/disease method no longer matches a saved correction, the strokes are kept and marked for review instead of being painted onto the new mask.
- Full precision is stored internally. The display is rounded to three decimal places.

## Windows (PowerShell)

```powershell
git clone https://github.com/Sapkota-Lab/neopest.git
cd neopest
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_app.py
```

Default workspace: `Documents\NeoPest` with `neopest_workspace.json`.

HEIC/HEIF import on Windows has not been verified. JPEG and PNG use the same orientation handling as on macOS.

## Testing

```bash
source .venv/bin/activate
python3 -m compileall neopest tests
python3 -m unittest discover -s tests -v
```
