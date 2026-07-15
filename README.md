# NeoPest

NeoPest is a desktop application for measuring total whole-leaf area from still images of strawberry leaves.

## Current workflow

1. Open or create a workspace.
2. Right-click empty Explorer space to create a condition group, or use **File → New Condition Group**.
3. Right-click a condition group to import still leaf images.
4. Drag samples between condition groups to reorganize them.
5. Select a sample and set **1-inch calibration** on the ruler.
6. Click **Preview Whole Leaf** to generate the whole-leaf mask overlay.
7. Click **Preview Candidate Disease** to preview brown/necrotic candidate regions inside the leaf.
8. Use **Preview mode** to switch among Raw Image, Whole Leaf, Candidate Disease, and Healthy Tissue Preview overlays.
9. View **Total Leaf Area** in square inches (in²) in the Inspector.

## macOS

```bash
git clone https://github.com/Sapkota-Lab/neopest.git
cd neopest
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_app.py
```

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

## Current limitations

- Whole-leaf area only (includes green and connected brown/inoculated tissue).
- Candidate disease and healthy-tissue masks are **preview-only**; disease area, percent disease, and other disease metrics are not calculated or saved.
- Disease preview outputs are transient and are not persisted as final scientific results.
- Manual image-specific calibration is unchanged; calibration is not propagated between samples.
- Biological accuracy of disease preview has not been validated on a representative real-image set.
- No export yet.
- **Run Analysis** is not yet enabled.
