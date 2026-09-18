"""Draw the contour, label, and color percentages on the image."""

import cv2
import numpy as np


LABEL_COLORS = {
    "healthy": (40, 180, 40),
    "yellowing": (0, 200, 255),
    "spotted": (0, 80, 220),
    "unknown": (128, 128, 128),
}


def annotate(image, contour, label: str, features: dict, score: float):
    """Return a BGR copy with contour and text overlay."""
    vis = image.copy()
    color = LABEL_COLORS.get(label, (200, 200, 200))

    if contour is not None:
        cv2.drawContours(vis, [contour], -1, color, 3)

    green_pct = 100.0 * float(features.get("green_ratio", 0.0))
    yellow_pct = 100.0 * float(features.get("yellow_ratio", 0.0))
    brown_pct = 100.0 * float(features.get("brown_ratio", 0.0))
    spots = int(features.get("spot_count", 0))

    lines = [
        f"label: {label}",
        f"score: {score:.2f}",
        f"green: {green_pct:.1f}%",
        f"yellow: {yellow_pct:.1f}%",
        f"brown: {brown_pct:.1f}%",
        f"spots: {spots}",
    ]

    x0, y0 = 12, 28
    pad = 8
    line_h = 24
    box_w = 230
    box_h = pad * 2 + line_h * len(lines)
    overlay = vis.copy()
    cv2.rectangle(overlay, (6, 6), (6 + box_w, 6 + box_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)

    for i, text in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(
            vis,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (x0, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            color if i == 0 else (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return vis


def mask_to_bgr(mask):
    """3-channel visualization of a binary mask."""
    if mask is None:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
