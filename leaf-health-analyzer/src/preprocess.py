"""Resize and blur so later HSV thresholds stay stable."""

import cv2

TARGET_LONG_SIDE = 800
BLUR_KERNEL = (5, 5)


def preprocess(image):
    """Resize so the long side is 800 pixels, then apply a light blur."""
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side == 0:
        return image
    scale = TARGET_LONG_SIDE / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, BLUR_KERNEL, 0)
    return blurred
