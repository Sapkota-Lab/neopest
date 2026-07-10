# NeoPest

NeoPest is a desktop application for measuring total whole-leaf area from still images of strawberry leaves.

## Current workflow

1. Open or create a workspace and import a still leaf image.
2. Select the sample and set **1-inch calibration** on the ruler.
3. Click **Preview Leaf Mask** to generate the whole-leaf mask overlay.
4. View **Total Leaf Area** in square inches (in²) in the Inspector.

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
- No disease segmentation yet.
- No healthy-tissue area yet.
- No export yet.
- **Run Analysis** is not yet enabled.
