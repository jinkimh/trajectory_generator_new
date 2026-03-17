#!/usr/bin/env python3
"""이미지의 픽셀 값 확인 스크립트"""
import cv2
import numpy as np
import sys

if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = "maps/Bexco.pgm"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Cannot read image {img_path}")
    sys.exit(1)

print(f"Image: {img_path}")
print(f"Shape: {img.shape}")
print(f"\nUnique pixel values:")
vals, counts = np.unique(img, return_counts=True)
for v, c in zip(vals, counts):
    percentage = (c / img.size) * 100
    print(f"  Value {v:3d}: {c:10d} pixels ({percentage:5.2f}%)")
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
