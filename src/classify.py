"""
Rule-based leaf health labels.

Tune these numbers after you collect real phone photos.
All decision thresholds live in this file only.
"""

from src.utils import clamp01


# --- thresholds (easy to retune) ---

# Spotted: many compact dark spots OR a large brown/necrotic area
SPOT_COUNT_HIGH = 8
BROWN_RATIO_HIGH = 0.08

# Yellowing: lots of yellow tissue, but not already called spotted
YELLOW_RATIO_HIGH = 0.22

# Healthy: mostly green, few spots, little brown/yellow
GREEN_RATIO_HEALTHY = 0.45
SPOT_COUNT_HEALTHY_MAX = 4
BROWN_RATIO_HEALTHY_MAX = 0.04

# Minimum leaf pixels; below this we cannot trust the mask
MIN_LEAF_PIXELS = 800


def classify(features: dict) -> tuple[str, float]:
    """
    Return (label, score).

    label is one of: healthy, yellowing, spotted, unknown
    score is a simple 0-1 confidence-like value (not a real probability).
    """
    leaf_pixels = int(features.get("leaf_pixels", 0))
    if leaf_pixels < MIN_LEAF_PIXELS:
        return "unknown", 0.0

    green = float(features.get("green_ratio", 0.0))
    yellow = float(features.get("yellow_ratio", 0.0))
    brown = float(features.get("brown_ratio", 0.0))
    spots = int(features.get("spot_count", 0))

    # 1) Disease spots / necrosis first
    if spots >= SPOT_COUNT_HIGH or brown >= BROWN_RATIO_HIGH:
        # More spots and more brown -> higher score
        spot_part = min(1.0, spots / float(SPOT_COUNT_HIGH + 10))
        brown_part = min(1.0, brown / (BROWN_RATIO_HIGH * 2.0))
        score = clamp01(0.55 + 0.25 * spot_part + 0.20 * brown_part)
        return "spotted", score

    # 2) Chlorosis / yellowing
    if yellow >= YELLOW_RATIO_HIGH:
        extra = (yellow - YELLOW_RATIO_HIGH) / max(1e-6, 1.0 - YELLOW_RATIO_HIGH)
        score = clamp01(0.55 + 0.45 * extra)
        return "yellowing", score

    # 3) Otherwise treat as healthy if it still looks mostly green
    if green >= GREEN_RATIO_HEALTHY and spots <= SPOT_COUNT_HEALTHY_MAX and brown <= BROWN_RATIO_HEALTHY_MAX:
        score = clamp01(0.50 + 0.50 * green)
        return "healthy", score

    # Ambiguous leftover (e.g. pale leaf, weak mask)
    # Prefer yellowing if yellow is the next strongest cue
    if yellow > green and yellow > 0.12:
        return "yellowing", 0.40
    if brown > 0.04 or spots >= 5:
        return "spotted", 0.40
    if green > 0.25:
        return "healthy", 0.40
    return "unknown", 0.20
