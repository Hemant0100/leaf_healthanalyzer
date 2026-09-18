"""Create six simple synthetic leaf images for the offline demo."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def paper_background(w=640, h=480):
    rng = np.random.default_rng(0)
    bg = np.full((h, w, 3), 235, dtype=np.uint8)
    noise = rng.integers(0, 12, size=bg.shape, dtype=np.uint8)
    return cv2.subtract(bg, noise)


def draw_leaf(canvas, center, axes, angle, color):
    overlay = canvas.copy()
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
    # slight vein
    cv2.line(
        overlay,
        (center[0], center[1] - axes[1] + 8),
        (center[0], center[1] + axes[1] - 8),
        tuple(max(0, c - 35) for c in color),
        2,
    )
    return overlay


def save(path: Path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    buf.tofile(str(path))
    print("wrote", path)


def healthy(name, green, angle):
    img = paper_background()
    h, w = img.shape[:2]
    img = draw_leaf(img, (w // 2, h // 2), (90, 160), angle, green)
    save(ROOT / "sample_data" / "healthy" / name, img)


def yellowing(name, color, angle):
    img = paper_background()
    h, w = img.shape[:2]
    img = draw_leaf(img, (w // 2, h // 2 + 10), (95, 155), angle, color)
    save(ROOT / "sample_data" / "yellowing" / name, img)


def spotted(name, green, n_spots, seed):
    img = paper_background()
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    axes = (92, 158)
    img = draw_leaf(img, center, axes, 8, green)
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 8, 0, 360, 255, -1)
    ys, xs = np.where(mask > 0)
    idx = rng.choice(len(xs), size=n_spots, replace=False)
    for i in idx:
        x, y = int(xs[i]), int(ys[i])
        r = int(rng.integers(4, 9))
        cv2.circle(img, (x, y), r, (20, 40, 70), -1)
    save(ROOT / "sample_data" / "spotted" / name, img)


def main():
    healthy("leaf1.jpg", (40, 160, 45), -12)
    healthy("leaf2.jpg", (35, 175, 55), 18)
    yellowing("leaf1.jpg", (30, 210, 230), -8)
    yellowing("leaf2.jpg", (40, 190, 200), 15)
    spotted("leaf1.jpg", (40, 155, 50), 18, seed=1)
    spotted("leaf2.jpg", (50, 165, 40), 22, seed=2)
    # copies in test/ so the test folder command has files
    for src_rel, dst in [
        ("healthy/leaf1.jpg", "test/healthy_leaf.jpg"),
        ("yellowing/leaf1.jpg", "test/yellow_leaf.jpg"),
        ("spotted/leaf1.jpg", "test/spotted_leaf.jpg"),
    ]:
        src = ROOT / "sample_data" / src_rel
        dst_path = ROOT / "sample_data" / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(src.read_bytes())
        print("wrote", dst_path)


if __name__ == "__main__":
    main()
