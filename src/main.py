"""
Leaf Health Analyzer

CLI:
  python src/main.py --image sample_data/healthy/leaf1.jpg --output outputs
  python src/main.py --folder sample_data --output outputs
  python src/main.py --folder sample_data/test --output outputs
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Allow `python src/main.py` from the project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import classify
from src.features import extract_features
from src.preprocess import preprocess
from src.segment import segment_leaf
from src.utils import list_images, safe_imread, safe_imwrite
from src.visualize import annotate, mask_to_bgr


CSV_FIELDS = [
    "filename",
    "label",
    "green_pct",
    "yellow_pct",
    "brown_pct",
    "spots",
    "score",
]


def analyze_one(image_path: Path) -> dict:
    """Run the full pipeline on one file. Never raises for bad images."""
    image = safe_imread(image_path)
    if image is None:
        return _unknown_row(image_path.name), None, None

    try:
        prepared = preprocess(image)
        mask, contour = segment_leaf(prepared)
        feats = extract_features(prepared, mask)
        label, score = classify(feats)
        vis = annotate(prepared, contour, label, feats, score)
        row = {
            "filename": image_path.name,
            "label": label,
            "green_pct": f"{100.0 * feats['green_ratio']:.2f}",
            "yellow_pct": f"{100.0 * feats['yellow_ratio']:.2f}",
            "brown_pct": f"{100.0 * feats['brown_ratio']:.2f}",
            "spots": feats["spot_count"],
            "score": f"{score:.3f}",
        }
        return row, vis, mask
    except Exception:
        return _unknown_row(image_path.name), None, None


def _unknown_row(name: str) -> dict:
    return {
        "filename": name,
        "label": "unknown",
        "green_pct": "0.00",
        "yellow_pct": "0.00",
        "brown_pct": "0.00",
        "spots": 0,
        "score": "0.000",
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser(description="Classify a leaf photo as healthy, yellowing, or spotted.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Path to a single image")
    group.add_argument("--folder", type=Path, help="Folder of images (searched recursively)")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output
    vis_dir = out_dir / "annotated"
    mask_dir = out_dir / "masks"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    source = args.image if args.image is not None else args.folder
    images = list_images(source)
    if not images:
        print(f"No images found at {source}")
        write_csv(out_dir / "results.csv", [])
        return 0

    rows = []
    for img_path in images:
        row, vis, mask = analyze_one(img_path)
        rows.append(row)
        stem = img_path.stem
        if vis is not None:
            safe_imwrite(vis_dir / f"{stem}_annotated.jpg", vis)
        if mask is not None:
            safe_imwrite(mask_dir / f"{stem}_mask.png", mask_to_bgr(mask))
        print(f"{img_path.name:30s}  {row['label']:10s}  green={row['green_pct']}%  "
              f"yellow={row['yellow_pct']}%  brown={row['brown_pct']}%  spots={row['spots']}  score={row['score']}")

    write_csv(out_dir / "results.csv", rows)
    print(f"\nWrote {len(rows)} rows to {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
