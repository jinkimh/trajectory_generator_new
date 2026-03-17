#!/usr/bin/env python3
"""
Split donut-style track boundaries from an edge-only map.

Assumes:
  - background is white (255)
  - boundaries are dark (0~1)
  - track is the white band between outer/inner boundaries

Outputs:
  - *_track.*  : filled drivable region (white=track)
  - *_outer.*  : outer boundary mask
  - *_inner.*  : inner boundary mask
  - *_outside.*: outside non-track region
  - *_hole.*   : inside hole (non-track) region
  - *_overlay.png : colored visualization
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
from collections import deque

import cv2
import numpy as np
import yaml


def _bfs_outside(background: np.ndarray) -> np.ndarray:
    """Flood-fill background from image borders to find outside non-track."""
    h, w = background.shape
    outside = np.zeros((h, w), dtype=np.bool_)
    q = deque()

    def push(y, x):
        outside[y, x] = True
        q.append((y, x))

    # Seed from borders
    for x in range(w):
        if background[0, x] and not outside[0, x]:
            push(0, x)
        if background[h - 1, x] and not outside[h - 1, x]:
            push(h - 1, x)
    for y in range(h):
        if background[y, 0] and not outside[y, 0]:
            push(y, 0)
        if background[y, w - 1] and not outside[y, w - 1]:
            push(y, w - 1)

    # 4-connected flood fill
    while q:
        y, x = q.popleft()
        if y > 0 and background[y - 1, x] and not outside[y - 1, x]:
            push(y - 1, x)
        if y + 1 < h and background[y + 1, x] and not outside[y + 1, x]:
            push(y + 1, x)
        if x > 0 and background[y, x - 1] and not outside[y, x - 1]:
            push(y, x - 1)
        if x + 1 < w and background[y, x + 1] and not outside[y, x + 1]:
            push(y, x + 1)

    return outside


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return mask
    keep = np.zeros_like(mask)
    for idx in range(1, num_labels):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == idx] = 255
    return keep


def main() -> int:
    ap = argparse.ArgumentParser(description="Split donut-style track boundaries")
    ap.add_argument("--map", type=str, required=True, help="Map name (without extension)")
    ap.add_argument("--ext", type=str, default=".png", help="Map image extension")
    ap.add_argument("--edge-method", type=str, default="threshold",
                    choices=["threshold", "canny"], help="Edge extraction method")
    ap.add_argument("--edge-thresh", type=int, default=5, help="Boundary threshold (<= thresh is edge)")
    ap.add_argument("--canny-low", type=int, default=50, help="Canny low threshold")
    ap.add_argument("--canny-high", type=int, default=150, help="Canny high threshold")
    ap.add_argument("--edge-dilate", type=int, default=0, help="Dilate edge mask (odd, 0 to skip)")
    ap.add_argument("--close-ksize", type=int, default=3, help="Morph close kernel size (odd, 0 to skip)")
    ap.add_argument("--open-ksize", type=int, default=0, help="Morph open kernel size (odd, 0 to skip)")
    ap.add_argument("--min-edge-area", type=int, default=0, help="Remove small edge components")
    ap.add_argument("--suffix", type=str, default="_tracksplit", help="Output suffix for track map")
    ap.add_argument("--save-yaml", action="store_true", help="Write YAML for track map if input YAML exists")
    ap.add_argument("--output-dir", type=str, default=None, help="Output directory (default: maps)")
    args = ap.parse_args()

    module = ROOT
    maps_dir = os.path.join(module, "maps")
    out_dir = args.output_dir or maps_dir
    os.makedirs(out_dir, exist_ok=True)

    img_path = os.path.join(maps_dir, args.map + args.ext)
    yaml_path = os.path.join(maps_dir, args.map + ".yaml")
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    # 1) Edge mask from dark boundary lines or Canny
    if args.edge_method == "canny":
        edge = cv2.Canny(img, int(args.canny_low), int(args.canny_high))
    else:
        edge = (img <= int(args.edge_thresh)).astype(np.uint8) * 255

    # 2) Morphological cleanup
    if args.edge_dilate and args.edge_dilate > 0:
        k = args.edge_dilate + (1 - args.edge_dilate % 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.dilate(edge, kernel)
    if args.close_ksize and args.close_ksize > 0:
        k = args.close_ksize + (1 - args.close_ksize % 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
    if args.open_ksize and args.open_ksize > 0:
        k = args.open_ksize + (1 - args.open_ksize % 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        edge = cv2.morphologyEx(edge, cv2.MORPH_OPEN, kernel)

    edge = _remove_small_components(edge, args.min_edge_area)

    # 3) Outside non-track via flood-fill on background
    background = edge == 0
    outside = _bfs_outside(background)
    inside = background & ~outside

    # 4) Classify boundary components by adjacency to outside
    edge_labels = cv2.connectedComponents(edge)[1]
    outside_dilate = cv2.dilate(outside.astype(np.uint8) * 255,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    outer_mask = np.zeros_like(edge)
    inner_mask = np.zeros_like(edge)
    for idx in range(1, edge_labels.max() + 1):
        comp = edge_labels == idx
        if np.any(outside_dilate[comp] > 0):
            outer_mask[comp] = 255
        else:
            inner_mask[comp] = 255

    # 5) Split inside regions into track vs hole based on adjacency to outer boundary
    inside_labels = cv2.connectedComponents((inside.astype(np.uint8) * 255))[1]
    outer_dilate = cv2.dilate(outer_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    track_mask = np.zeros_like(edge)
    hole_mask = np.zeros_like(edge)
    for idx in range(1, inside_labels.max() + 1):
        comp = inside_labels == idx
        if np.any(outer_dilate[comp] > 0):
            track_mask[comp] = 255
        else:
            hole_mask[comp] = 255

    # 6) Save outputs
    base = os.path.join(out_dir, args.map + args.suffix)
    ext = args.ext
    out_track = base + ext
    out_outer = base + "_outer" + ext
    out_inner = base + "_inner" + ext
    out_outside = base + "_outside" + ext
    out_hole = base + "_hole" + ext
    out_overlay = base + "_overlay.png"

    cv2.imwrite(out_track, track_mask)
    cv2.imwrite(out_outer, outer_mask)
    cv2.imwrite(out_inner, inner_mask)
    cv2.imwrite(out_outside, outside.astype(np.uint8) * 255)
    cv2.imwrite(out_hole, hole_mask)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay[track_mask == 255] = (0, 255, 0)
    overlay[outer_mask == 255] = (0, 0, 255)
    overlay[inner_mask == 255] = (255, 0, 0)
    overlay[hole_mask == 255] = (0, 255, 255)
    cv2.imwrite(out_overlay, overlay)

    # Optional YAML copy for the track mask
    if args.save_yaml and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh)
        yaml_data["image"] = os.path.basename(out_track)
        out_yaml = base + ".yaml"
        with open(out_yaml, "w", encoding="utf-8") as fh:
            yaml.dump(yaml_data, fh, default_flow_style=False)

    print("Track boundary split completed.")
    print(f"- track mask: {out_track}")
    print(f"- outer boundary: {out_outer}")
    print(f"- inner boundary: {out_inner}")
    print(f"- outside non-track: {out_outside}")
    print(f"- inside hole: {out_hole}")
    print(f"- overlay: {out_overlay}")
    if args.save_yaml and os.path.exists(yaml_path):
        print(f"- yaml: {base + '.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
