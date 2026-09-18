"""Small helpers used by the rest of the pipeline."""

from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(path: Path) -> list[Path]:
    """Return image files from a file or folder (recursive for folders)."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not path.is_dir():
        return []
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    return files


def safe_imread(path: Path):
    """Read an image. Returns None if the file cannot be decoded."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def safe_imwrite(path: Path, image) -> bool:
    """Write an image without raising. Returns False on failure."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower() or ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
