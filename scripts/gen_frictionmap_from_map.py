#!/usr/bin/env python3
"""Generate a simple friction map from a raster map (PGM/PNG).

Creates:
  - *_tpamap.csv : x,y grid points (meters)
  - *_tpadata.json : mu value per grid index
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _parse_rects(raw: List[List[str]]) -> List[Tuple[float, float, float, float, float]]:
    rects: List[Tuple[float, float, float, float, float]] = []
    if not raw:
        return rects
    for item in raw:
        if len(item) != 5:
            raise ValueError("--low-mu-rect requires 5 values: xmin ymin xmax ymax mu")
        rects.append(tuple(float(x) for x in item))
    return rects


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate frictionmap from raster map")
    ap.add_argument("--map-file", required=True, help="Path to map image (pgm/png)")
    ap.add_argument("--yaml-file", default=None, help="Path to map yaml (resolution/origin)")
    ap.add_argument("--out-dir", default=None, help="Output directory")
    ap.add_argument("--out-name", default=None, help="Output base name (without suffix)")
    ap.add_argument("--cellwidth", type=float, default=0.5, help="Grid spacing in meters")
    ap.add_argument("--track-thresh", type=int, default=250,
                    help="Track threshold in [0,255] (track pixels >= thresh)")
    ap.add_argument("--mu", type=float, default=1.0, help="Default friction coefficient")
    ap.add_argument("--low-mu-rect", nargs=5, action="append", metavar=("XMIN", "YMIN", "XMAX", "YMAX", "MU"),
                    help="Optional low-mu rectangle in world coords (repeatable)")

    args = ap.parse_args()

    map_file = args.map_file
    if not os.path.exists(map_file):
        raise FileNotFoundError(map_file)

    yaml_file = args.yaml_file
    if yaml_file is None:
        base = os.path.splitext(map_file)[0]
        yaml_file = base + ".yaml"
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(yaml_file)

    meta = _load_yaml(yaml_file)
    resolution = float(meta["resolution"])
    origin = meta["origin"]
    origin_x, origin_y = float(origin[0]), float(origin[1])
    negate = int(meta.get("negate", 0))

    img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read map image: {map_file}")
    if negate == 1:
        img = 255 - img

    track_thresh = int(args.track_thresh)
    track_mask = img >= track_thresh

    height, width = img.shape
    step_pix = max(1, int(round(args.cellwidth / resolution)))

    xs = np.arange(0, width, step_pix)
    ys = np.arange(0, height, step_pix)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pix = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    on_track = track_mask[pix[:, 1], pix[:, 0]]
    pix = pix[on_track]

    if pix.size == 0:
        raise RuntimeError("No grid points found on track. Check threshold or map.")

    # pixel -> world (ROS map convention; origin at bottom-left)
    x_pix = pix[:, 0].astype(np.float64)
    y_pix = pix[:, 1].astype(np.float64)
    x_world = origin_x + (x_pix + 0.5) * resolution
    y_world = origin_y + ((height - 1 - y_pix) + 0.5) * resolution
    coords = np.stack([x_world, y_world], axis=1)

    rects = _parse_rects(args.low_mu_rect)
    mu = np.full((coords.shape[0],), float(args.mu), dtype=np.float64)

    for xmin, ymin, xmax, ymax, mu_val in rects:
        mask = (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax) & \
               (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
        mu[mask] = mu_val

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(ROOT, "inputs", "frictionmaps")
    os.makedirs(out_dir, exist_ok=True)

    if args.out_name:
        out_name = args.out_name
    else:
        out_name = os.path.splitext(os.path.basename(map_file))[0]

    tpamap_path = os.path.join(out_dir, f"{out_name}_tpamap.csv")
    tpadata_path = os.path.join(out_dir, f"{out_name}_tpadata.json")

    np.savetxt(tpamap_path, coords, fmt="%0.4f", delimiter=";", header="x_m;y_m")

    data = {str(i): [float(mu_val)] for i, mu_val in enumerate(mu)}
    with open(tpadata_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, separators=(",", ":"))

    print(f"Saved {coords.shape[0]} points")
    print(f"tpamap: {tpamap_path}")
    print(f"tpadata: {tpadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
