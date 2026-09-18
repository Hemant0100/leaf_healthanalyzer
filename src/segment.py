"""Segment the leaf as the largest green/yellow blob."""

import cv2
import numpy as np


# HSV ranges (OpenCV H: 0-180, S/V: 0-255)
# Green leaves
LOWER_GREEN = np.array([25, 40, 40])
UPPER_GREEN = np.array([90, 255, 255])
# Yellow / chlorotic tissue
LOWER_YELLOW = np.array([15, 40, 40])
UPPER_YELLOW = np.array([35, 255, 255])
# Brown / necrotic tissue (low saturation, mid-low value, red-yellow hue)
LOWER_BROWN1 = np.array([0, 30, 20])
UPPER_BROWN1 = np.array([20, 255, 160])
LOWER_BROWN2 = np.array([10, 20, 15])
UPPER_BROWN2 = np.array([30, 180, 140])


def _color_mask(hsv):
    green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    brown1 = cv2.inRange(hsv, LOWER_BROWN1, UPPER_BROWN1)
    brown2 = cv2.inRange(hsv, LOWER_BROWN2, UPPER_BROWN2)
    mask = cv2.bitwise_or(green, yellow)
    mask = cv2.bitwise_or(mask, brown1)
    mask = cv2.bitwise_or(mask, brown2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def segment_leaf(image):
    """
    Return (mask, contour).
    mask is uint8 0/255. contour is the largest leaf contour or None.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = _color_mask(hsv)

    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        return empty, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        empty = np.zeros(image.shape[:2], dtype=np.uint8)
        return empty, None

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask, largest
