# Leaf Health Analyzer

Command-line OpenCV tool that looks at a photo of a **single leaf** and labels it **healthy**, **yellowing**, or **spotted**. It uses color ratios (green / yellow / brown) and a simple dark-spot count. No neural networks.

Author: **ARNAV YADAV**

## How to capture photos

Use your own phone photos for the real project. Demo images in `sample_data/` are only synthetic placeholders.

- Photograph **one leaf** at a time
- Plain background (white paper, notebook, or table)
- Daylight or a window; avoid harsh flash
- Fill most of the frame with the leaf
- Keep the camera roughly above the leaf (top-down)
- Do not stack multiple leaves in one picture

Put real photos here:

- `sample_data/healthy/`
- `sample_data/yellowing/`
- `sample_data/spotted/`
- `sample_data/test/` (held-out photos you classify by hand)

## Setup

Python 3.10+ is required. Run these commands from the **repository root**.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

Single image:

```bash
python src/main.py --image sample_data/healthy/leaf1.jpg --output outputs
```

Whole dataset (recursive):

```bash
python src/main.py --folder sample_data --output outputs
```

Test folder only:

```bash
python src/main.py --folder sample_data/test --output outputs
```

Demo script (same as the folder command):

```bash
bash scripts/run_demo.sh
```

On Windows, use the Python commands above instead of the `.sh` file.

## Expected output files

After a run, `outputs/` contains:

| File | Meaning |
| --- | --- |
| `outputs/results.csv` | One row per image: filename, label, green%, yellow%, brown%, spots, score |
| `outputs/annotated/<name>_annotated.jpg` | Leaf contour, predicted label, percentages |
| `outputs/masks/<name>_mask.png` | Binary leaf mask |

`score` is a simple 0–1 confidence-like number from the rules, not a trained probability.

## How it works

1. Read one image or a folder
2. Resize the long side to 800 pixels
3. Light Gaussian blur
4. HSV color mask (green / yellow / brown) and keep the **largest contour** as the leaf
5. Inside that mask compute green, yellow, and brown pixel ratios
6. Count compact dark spots with morphological opening
7. Rule-based label (thresholds live in `src/classify.py`)
8. Bad or unreadable files get `label=unknown` and do not crash the program

## Limitations

- Assumes **one leaf** that is the largest green/yellow object
- Plain background helps; busy grass or soil can confuse the mask
- Color rules depend on lighting; indoor yellow lamps can look like yellowing
- Spot counting can miss tiny speckles or count veins/shadows as spots
- Not a plant-pathology diagnosis; only a classroom color/spot demo
- Thresholds in `src/classify.py` should be retuned after you add real photos

## Project layout

```
.
  README.md
  PROJECT_REPORT.md
  requirements.txt
  sample_data/
  src/
  outputs/
  scripts/run_demo.sh
```
