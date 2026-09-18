# Project Report: Leaf Health Analyzer

**Author:** ARNAV YADAV

## Abstract

TODO: 150–200 words. State the problem (leaf health from a phone photo), the method (OpenCV color + spots, no neural network), and placeholder results.

## Introduction

TODO: Why leaf appearance matters. Why a simple computer-vision pipeline is enough for this assignment. Mention that photos will be self-collected.

## Objectives

- Segment a single leaf from a plain background using HSV and the largest contour.
- Measure green, yellow, and brown pixel ratios inside the leaf mask.
- Count dark spots with morphological opening.
- Classify each image as healthy, yellowing, or spotted with explicit rules.
- Save annotated images, masks, and a CSV of features.

## Dataset (own collected leaves)

TODO: Replace synthetic demo files with real phone photos.

| Class | Planned count | Notes |
| --- | --- | --- |
| Healthy | TODO | Green, no obvious spots |
| Yellowing | TODO | Chlorosis / pale yellow tissue |
| Spotted | TODO | Dark dots or brown patches |
| Test (held-out) | TODO | Mix of the three classes |

Capture protocol: one leaf, plain background, daylight, top-down. See `sample_data/README.md`.

## Methodology

1. Resize long side to 800, Gaussian blur.
2. HSV inRange for green / yellow / brown; morphological close/open.
3. Largest external contour → binary leaf mask.
4. Features: green ratio, yellow ratio, brown ratio, spot count.
5. Rules in `src/classify.py` (tune after real photos).
6. Unknown label if the file cannot be read or the mask is too small.

## Implementation

Modules:

- `src/preprocess.py` — resize and blur
- `src/segment.py` — HSV + largest contour
- `src/features.py` — color ratios and spots
- `src/classify.py` — thresholds and labels
- `src/visualize.py` — overlay
- `src/main.py` — argparse CLI
- `src/utils.py` — safe I/O with pathlib

Stack: Python 3.10+, opencv-python, numpy, argparse.

## Results

TODO: Fill after running on **your** photos (not only synthetic demo leaves).

| Metric | Value |
| --- | --- |
| Number of images | TODO |
| Healthy correct | TODO |
| Yellowing correct | TODO |
| Spotted correct | TODO |
| Overall accuracy | TODO |
| Typical green% (healthy) | TODO |
| Typical yellow% (yellowing) | TODO |
| Typical spot count (spotted) | TODO |

Confusion notes: TODO.

Example rows from `outputs/results.csv`: TODO.

## Limitations

- Lighting and white-balance change HSV values.
- Overlapping leaves and cluttered backgrounds break the largest-contour assumption.
- Rule thresholds need retuning per camera and plant species.
- No disease identification beyond three visual classes.

## Conclusion

TODO: What worked, what failed on real photos, and one improvement (for example better spot filtering or a second color space).

## References

- OpenCV documentation: https://docs.opencv.org/
- Gonzalez, R. C. and Woods, R. E. *Digital Image Processing*.
- Course lab notes / instructor slides (TODO: add the exact title).
