"""Color ratios and spot count inside the leaf mask."""

import cv2
import numpy as np


# HSV color bins used only for feature extraction (not segmentation).
GREEN_H = (35, 85)
YELLOW_H = (18, 35)
# Brown / dark necrotic pixels
BROWN_H = (0, 25)
DARK_V_MAX = 90
YELLOW_S_MIN = 50
GREEN_S_MIN = 40
BROWN_S_MIN = 20

# Morphological opening kernel for dark spots
SPOT_KERNEL_SIZE = 5
MIN_SPOT_AREA = 12
MAX_SPOT_AREA_RATIO = 0.08  # ignore huge dark regions (shadows)


def _ratio(count, total):
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def extract_features(image, mask):
    """
    Compute green / yellow / brown ratios and a dark-spot count.

    Returns a dict with keys:
      green_ratio, yellow_ratio, brown_ratio, spot_count, leaf_pixels
    """
    empty = {
        "green_ratio": 0.0,
        "yellow_ratio": 0.0,
        "brown_ratio": 0.0,
        "spot_count": 0,
        "leaf_pixels": 0,
    }
    if mask is None or int(cv2.countNonZero(mask)) == 0:
        return empty

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    leaf = mask > 0
    leaf_pixels = int(np.count_nonzero(leaf))
    if leaf_pixels == 0:
        return empty

    green = leaf & (h >= GREEN_H[0]) & (h <= GREEN_H[1]) & (s >= GREEN_S_MIN)
    yellow = leaf & (h >= YELLOW_H[0]) & (h < YELLOW_H[1]) & (s >= YELLOW_S_MIN)
    brown = leaf & (
        ((h >= BROWN_H[0]) & (h <= BROWN_H[1]) & (s >= BROWN_S_MIN) & (v <= 170))
        | (v <= DARK_V_MAX)
    )
    # Do not double-count green as brown
    brown = brown & ~green

    green_ratio = _ratio(int(np.count_nonzero(green)), leaf_pixels)
    yellow_ratio = _ratio(int(np.count_nonzero(yellow)), leaf_pixels)
    brown_ratio = _ratio(int(np.count_nonzero(brown)), leaf_pixels)

    spot_count = _count_spots(v, mask, brown)

    return {
        "green_ratio": green_ratio,
        "yellow_ratio": yellow_ratio,
        "brown_ratio": brown_ratio,
        "spot_count": spot_count,
        "leaf_pixels": leaf_pixels,
    }


def _count_spots(v_channel, mask, brown_bool):
    """Count compact dark blobs after morphological opening."""
    dark = np.zeros(mask.shape, dtype=np.uint8)
    dark[brown_bool] = 255
    dark = cv2.bitwise_and(dark, mask)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (SPOT_KERNEL_SIZE, SPOT_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_area = max(1, int(cv2.countNonZero(mask)))
    spots = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_SPOT_AREA:
            continue
        if area / leaf_area > MAX_SPOT_AREA_RATIO:
            continue
        spots += 1
    return spots
