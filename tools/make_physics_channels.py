"""Physics-derived input channels for SAR: replace the 3 identical grey channels by (raw, edge-preserving smoothed, edge magnitude).
  ch0 raw grey image (8-bit, as stored)
  ch1 bilateral filter (d 5, sigma colour 50, sigma space 5): smooths speckle while keeping edges (MDIS-DETR stacks Gaussian and bilateral copies)
  ch2 Sobel gradient magnitude of a Gaussian-smoothed copy (sigma 1), scaled per image so its 99th percentile maps to 255 (MSFA-style edge channel)
Written as lossless PNG with the same stem, so a YOLO dataset can reuse the existing label files. Deterministic, one image in, one image out.
Usage: python tools/make_physics_channels.py <yolo_dataset_in> <yolo_dataset_out>   (copies the split folders, links the labels)"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

src, dst = Path(sys.argv[1]).expanduser(), Path(sys.argv[2]).expanduser()


def physics(grey):
    smooth = cv2.bilateralFilter(grey, 5, 50, 5)
    g = cv2.GaussianBlur(grey.astype(np.float32), (0, 0), 1.0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    p99 = float(np.percentile(mag, 99)) or 1.0
    edge = np.clip(mag / p99 * 255, 0, 255).astype(np.uint8)
    return cv2.merge([grey, smooth, edge])


for split in ('train', 'val'):
    (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
    files = sorted((src / 'images' / split).iterdir())
    for f in files:
        grey = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(dst / 'images' / split / (f.stem + '.png')), physics(grey))
    print(split, len(files), 'images')
if not (dst / 'labels').exists():
    os.symlink(os.path.relpath((src / 'labels').resolve(), dst), dst / 'labels')
(dst / 'data.yaml').write_text((src / 'data.yaml').read_text().replace(f'path: {src.resolve()}', f'path: {dst.resolve()}'))
