#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import yaml

from trajectory_generator_new.config_utils import load_map_defaults


def load_yaml(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def map_to_pixel(x_m, y_m, height, res, origin):
    col = (x_m - origin[0]) / res
    row = height - (y_m - origin[1]) / res
    return int(round(col)), int(round(row))


def detect_delimiter(line):
    if ";" in line:
        return ";"
    return ","


def load_csv_points(csv_path):
    lines = []
    with open(csv_path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    if not lines:
        raise ValueError(f"No data rows found in {csv_path}")
    delim = detect_delimiter(lines[0])
    data = []
    for line in lines:
        parts = [p.strip() for p in line.split(delim)]
        if len(parts) < 2:
            continue
        data.append([float(p) for p in parts])
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"CSV format not recognized: {csv_path}")

    x = None
    y = None
    v = None
    if arr.shape[1] >= 6:
        # s, x, y, psi, kappa, vx, ...
        x = arr[:, 1]
        y = arr[:, 2]
        v = arr[:, 5]
    else:
        x = arr[:, 0]
        y = arr[:, 1]
        v = None
    return x, y, v


def color_from_speed(v, vmin, vmax):
    if vmax <= vmin:
        val = 0
    else:
        val = int(round(255 * (v - vmin) / (vmax - vmin)))
    val = max(0, min(255, val))
    color = cv2.applyColorMap(np.uint8([[val]]), cv2.COLORMAP_JET)[0][0]
    return int(color[0]), int(color[1]), int(color[2])


def resolve_original_map_name(name: str) -> str:
    for suffix in ("_tracksplit_processed", "_tracksplit", "_processed"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main():
    parser = argparse.ArgumentParser(description="Plot raceline CSV on map with optional speed gradient.")
    parser.add_argument("--map", type=str, default=None, help="Map name (without extension)")
    parser.add_argument("--map-ext", type=str, default=None, help="Map image extension (.png/.pgm).")
    parser.add_argument("--csv", type=str, required=True, help="Input raceline CSV path")
    parser.add_argument("--out", type=str, default=None, help="Output overlay image path (.png)")
    parser.add_argument("--stride", type=int, default=20, help="Label every N points")
    parser.add_argument("--line-width", type=int, default=2, help="Line thickness")
    parser.add_argument("--font-scale", type=float, default=0.4, help="Font scale for labels")
    parser.add_argument("--font-thickness", type=int, default=1, help="Font thickness for labels")
    parser.add_argument("--use-processed", action="store_true",
                        help="Use processed map name if provided (default: show original map).")
    args = parser.parse_args()

    module = ROOT
    map_name, map_ext = load_map_defaults(module)
    map_path = None
    if args.map:
        # Allow full path or name with extension
        if os.path.isfile(args.map):
            map_path = args.map
            base = os.path.basename(args.map)
            map_name = os.path.splitext(base)[0]
            map_ext = os.path.splitext(base)[1] or map_ext
        else:
            map_name = args.map
            name_no_ext, ext = os.path.splitext(map_name)
            if ext in (".pgm", ".png"):
                map_name = name_no_ext
                map_ext = ext
    if args.map_ext:
        map_ext = args.map_ext
    if not args.use_processed:
        map_name = resolve_original_map_name(map_name)

    # Resolve map image path
    if map_path is None:
        map_path = os.path.join(module, "maps", f"{map_name}{map_ext}")
        if not os.path.exists(map_path):
            alt_exts = [".png", ".pgm"]
            for ext in alt_exts:
                cand = os.path.join(module, "maps", f"{map_name}{ext}")
                if os.path.exists(cand):
                    map_path = cand
                    break
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Map image not found: {map_path}")

    yaml_path = os.path.join(module, "maps", f"{map_name}.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Map YAML not found: {yaml_path}")

    map_yaml = load_yaml(yaml_path)
    res = map_yaml["resolution"]
    origin = map_yaml["origin"]

    base = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if base is None:
        raise FileNotFoundError(f"Failed to read map image: {map_path}")
    canvas = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    x, y, v = load_csv_points(args.csv)
    h = canvas.shape[0]

    pts = [map_to_pixel(xi, yi, h, res, origin) for xi, yi in zip(x, y)]
    if v is not None:
        vmin = float(np.min(v))
        vmax = float(np.max(v))
    else:
        vmin = vmax = 0.0

    # Draw line segments
    for i in range(len(pts) - 1):
        if v is not None:
            color = color_from_speed(float(v[i]), vmin, vmax)
        else:
            color = (0, 0, 255)
        cv2.line(canvas, pts[i], pts[i + 1], color, args.line_width)

    # Labels for speed
    if v is not None:
        idxs = set(range(0, len(v), max(1, args.stride)))
        idxs.add(int(np.argmin(v)))
        idxs.add(int(np.argmax(v)))
        for idx in sorted(idxs):
            if idx < 0 or idx >= len(pts):
                continue
            col, row = pts[idx]
            label = f"{v[idx]:.2f}"
            cv2.putText(
                canvas,
                label,
                (col + 4, row - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                args.font_scale,
                (0, 0, 255),
                args.font_thickness,
                lineType=cv2.LINE_AA,
            )

    if args.out:
        out_path = args.out
    else:
        base_name = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join(os.path.dirname(args.csv), f"{base_name}_overlay.png")

    cv2.imwrite(out_path, canvas)
    print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()
