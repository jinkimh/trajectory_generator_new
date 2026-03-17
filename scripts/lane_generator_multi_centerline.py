import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import yaml
import argparse
from collections import deque

from trajectory_generator_new.config_utils import load_map_defaults

def show_images_grid(imgs, title, cols=3):
    if not imgs:
        return False
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    cv2.imshow(title, grid)
    print("Press any key to continue (or 'q' to quit)...")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise SystemExit("User requested exit.")
        if key != 255:
            cv2.destroyAllWindows()
            return True
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            return True


def save_csv(data, csv_name, header=None):
    with open(csv_name, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if header:
            csv_writer.writerow(header)
        for line in data:
            csv_writer.writerow(line.tolist())


def transform_coords(path, height, s, tx, ty):
    new_path_x = path[:, 0] * s + tx
    new_path_y = (height - path[:, 1]) * s + ty
    if path.shape[1] > 2:
        new_dist = path[:, 2] * s
        return np.vstack((new_path_x, new_path_y, new_dist, new_dist)).T
    return np.vstack((new_path_x, new_path_y)).T


def auto_invert_if_needed(
    img,
    negate=None,
    white_thresh=250,
    black_thresh=5,
    ratio_margin=0.05,
):
    """Invert only when metadata or pixel ratios strongly indicate white background."""
    if negate is not None:
        try:
            if int(negate) == 1:
                return 255 - img, True, f"negate={negate}"
            return img, False, f"negate={negate}"
        except (TypeError, ValueError):
            pass

    white_ratio = float((img >= white_thresh).mean())
    black_ratio = float((img <= black_thresh).mean())
    if (white_ratio - black_ratio) > ratio_margin:
        return 255 - img, True, f"white_ratio={white_ratio:.2%} black_ratio={black_ratio:.2%}"
    return img, False, f"white_ratio={white_ratio:.2%} black_ratio={black_ratio:.2%}"


def threshold_track(img, strict_white=True, track_value=254, track_tolerance=0):
    if strict_white:
        track_min = max(0, track_value - track_tolerance)
        return (img >= track_min).astype(np.uint8) * 255
    _, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bin_img


def remove_small_components(bin_img, min_area):
    if min_area <= 0:
        return bin_img
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img)
    if num_labels <= 1:
        return bin_img
    keep = np.zeros_like(bin_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def skeletonize(bin_img):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(bin_img)
    # Fallback morphology-based skeleton
    skel = np.zeros_like(bin_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = bin_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def prune_short_branches(skel, min_len=15):
    skel = skel.copy()
    h, w = skel.shape
    skel_bin = skel > 0

    def neighbors(y, x):
        pts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel_bin[ny, nx]:
                    pts.append((ny, nx))
        return pts

    changed = True
    while changed:
        changed = False
        endpoints = []
        for y in range(h):
            for x in range(w):
                if skel_bin[y, x] and len(neighbors(y, x)) == 1:
                    endpoints.append((y, x))

        for ep in endpoints:
            path = [ep]
            prev = None
            curr = ep
            while True:
                nbs = neighbors(*curr)
                nbs = [n for n in nbs if n != prev]
                if not nbs:
                    break
                if len(nbs) >= 2:
                    break
                nxt = nbs[0]
                path.append(nxt)
                prev, curr = curr, nxt
                if len(path) >= min_len:
                    break
            if len(path) < min_len:
                for y, x in path:
                    skel_bin[y, x] = False
                changed = True

    return (skel_bin.astype(np.uint8) * 255)


def build_skeleton_graph(skel):
    skel_bin = skel > 0
    h, w = skel.shape

    def neighbors(y, x):
        pts = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and skel_bin[ny, nx]:
                    pts.append((ny, nx))
        return pts

    degrees = {}
    for y in range(h):
        for x in range(w):
            if skel_bin[y, x]:
                degrees[(y, x)] = len(neighbors(y, x))

    nodes = {pt for pt, deg in degrees.items() if deg != 2}
    if not nodes and degrees:
        nodes = {next(iter(degrees.keys()))}

    edge_id = 0
    edges = []
    adj = {n: [] for n in nodes}
    visited = set()

    for n in nodes:
        for nb in neighbors(*n):
            edge_key = tuple(sorted([n, nb]))
            if edge_key in visited:
                continue
            path = [n]
            prev = n
            curr = nb
            while True:
                path.append(curr)
                if curr in nodes and curr != n:
                    break
                nbs = neighbors(*curr)
                if not nbs:
                    break
                next_candidates = [p for p in nbs if p != prev]
                if not next_candidates:
                    break
                prev, curr = curr, next_candidates[0]
            edge = {
                "id": edge_id,
                "n1": n,
                "n2": curr,
                "path": path,
            }
            edges.append(edge)
            if n in adj:
                adj[n].append(edge_id)
            if curr in adj:
                adj[curr].append(edge_id)
            for i in range(len(path) - 1):
                visited.add(tuple(sorted([path[i], path[i + 1]])))
            edge_id += 1

    return nodes, edges, adj


def build_edge_map(edges):
    edge_map = {}
    for edge in edges:
        key = tuple(sorted([edge["n1"], edge["n2"]]))
        edge_map.setdefault(key, []).append(edge["id"])
    return edge_map


def tree_path(parent, depth, u, v):
    path_u = []
    path_v = []
    uu, vv = u, v
    while depth[uu] > depth[vv]:
        path_u.append(uu)
        uu = parent[uu]
    while depth[vv] > depth[uu]:
        path_v.append(vv)
        vv = parent[vv]
    while uu != vv:
        path_u.append(uu)
        path_v.append(vv)
        uu = parent[uu]
        vv = parent[vv]
    lca = uu
    path_u.append(lca)
    return path_u + path_v[::-1]


def extract_cycle_paths(nodes, edges, adj, max_cycles=20, min_len=100, dedup_iou=0.6, mask_shape=None):
    if not nodes or not edges:
        return []
    edge_map = build_edge_map(edges)
    # Build DFS forest (handle disconnected components)
    parent = {}
    depth = {}
    comp = {}
    tree_edges = set()
    comp_id = 0
    for root in nodes:
        if root in parent:
            continue
        parent[root] = root
        depth[root] = 0
        comp[root] = comp_id
        stack = [root]
        while stack:
            u = stack.pop()
            for eid in adj[u]:
                edge = edges[eid]
                v = edge["n2"] if edge["n1"] == u else edge["n1"]
                if v not in parent:
                    parent[v] = u
                    depth[v] = depth[u] + 1
                    comp[v] = comp_id
                    tree_edges.add(eid)
                    stack.append(v)
        comp_id += 1

    cycles = []
    cycle_infos = []
    for edge in edges:
        if edge["id"] in tree_edges:
            continue
        u, v = edge["n1"], edge["n2"]
        if comp.get(u) != comp.get(v):
            continue
        node_cycle = tree_path(parent, depth, u, v)
        # close cycle with the non-tree edge
        node_cycle.append(u)
        # convert node cycle to pixel path
        pixel_path = []
        for i in range(len(node_cycle) - 1):
            a = node_cycle[i]
            b = node_cycle[i + 1]
            key = tuple(sorted([a, b]))
            if key not in edge_map:
                continue
            eid = edge_map[key][0]
            seg = edges[eid]["path"]
            if edges[eid]["n1"] != a:
                seg = list(reversed(seg))
            if pixel_path:
                seg = seg[1:]
            pixel_path.extend(seg)
        if len(pixel_path) > 2:
            path = np.array([(p[1], p[0]) for p in pixel_path], dtype=float)
            if path_length(path) >= min_len:
                cycle_infos.append((path_length(path), path))
        if len(cycle_infos) >= max_cycles * 3:
            break

    if not cycle_infos:
        return []

    cycle_infos.sort(key=lambda x: x[0], reverse=True)
    kept = []
    kept_masks = []
    for _, path in cycle_infos:
        if len(kept) >= max_cycles:
            break
        if mask_shape is None:
            kept.append(path)
            continue
        mask = path_mask(path, mask_shape)
        is_dup = False
        for km in kept_masks:
            inter = np.logical_and(mask, km).sum()
            union = np.logical_or(mask, km).sum()
            iou = inter / max(1, union)
            if iou >= dedup_iou:
                is_dup = True
                break
        if not is_dup:
            kept.append(path)
            kept_masks.append(mask)

    return kept


def enumerate_paths(nodes, edges, adj, max_paths=50, max_states=20000, max_edge_len=10000):
    node_list = list(nodes)
    node_index = {n: i for i, n in enumerate(node_list)}

    degrees = {n: len(adj[n]) for n in nodes}
    endpoints = [n for n in nodes if degrees[n] == 1]
    paths = []

    def edge_other(edge, node):
        return edge["n2"] if edge["n1"] == node else edge["n1"]

    if not endpoints:
        if not edges:
            return paths
        # Single loop: traverse edges in order
        start_edge = edges[0]
        start_node = start_edge["n1"]
        visited_edges = set()
        path_edges = []
        curr = start_node
        while True:
            next_edges = [eid for eid in adj[curr] if eid not in visited_edges]
            if not next_edges:
                break
            eid = next_edges[0]
            visited_edges.add(eid)
            path_edges.append(eid)
            curr = edge_other(edges[eid], curr)
            if curr == start_node:
                break
        if path_edges:
            paths.append(path_edges)
        return paths

    states = 0
    for start in endpoints:
        start_id = node_index[start]
        stack = deque()
        stack.append((start, [], set()))
        while stack:
            node, path_edges, used_edges = stack.pop()
            states += 1
            if states > max_states:
                return paths
            if len(paths) >= max_paths:
                return paths
            if node in endpoints and node != start and path_edges:
                if node_index[node] > start_id:
                    paths.append(path_edges)
                continue
            for eid in adj[node]:
                if eid in used_edges:
                    continue
                edge = edges[eid]
                if len(edge["path"]) > max_edge_len:
                    continue
                nxt = edge_other(edge, node)
                stack.append((nxt, path_edges + [eid], used_edges | {eid}))
    return paths


def stitch_path(edges, path_edge_ids, start_node):
    pts = []
    curr = start_node
    for eid in path_edge_ids:
        edge = edges[eid]
        if edge["n1"] == curr:
            seg = edge["path"]
            curr = edge["n2"]
        else:
            seg = list(reversed(edge["path"]))
            curr = edge["n1"]
        if pts:
            seg = seg[1:]
        pts.extend(seg)
    return np.array([(p[1], p[0]) for p in pts], dtype=np.int32)


def smooth_path(path, window=5):
    if window <= 2 or len(path) < window:
        return path
    kernel = np.ones(window) / window
    pad = window // 2
    pts = path.astype(float)
    # If the path is a loop, smooth with circular padding to avoid endpoint jumps.
    if np.linalg.norm(pts[0] - pts[-1]) < 2.0:
        xs_src = np.r_[pts[-pad:, 0], pts[:, 0], pts[:pad, 0]]
        ys_src = np.r_[pts[-pad:, 1], pts[:, 1], pts[:pad, 1]]
    else:
        xs_src = np.pad(pts[:, 0], (pad, pad), mode='edge')
        ys_src = np.pad(pts[:, 1], (pad, pad), mode='edge')
    xs = np.convolve(xs_src, kernel, mode='valid')
    ys = np.convolve(ys_src, kernel, mode='valid')
    return np.vstack((xs, ys)).T


def overlay_path(base, path, color):
    img = base.copy()
    pts = np.round(path).astype(int)
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, 1)
    return img


def extract_longest_contour_path(skel):
    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=len)
    pts = contour.squeeze()
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    return pts.astype(float)


def path_length(path):
    if len(path) < 2:
        return 0.0
    d = np.sqrt(np.sum(np.diff(path[:, :2], axis=0) ** 2, axis=1))
    return float(d.sum())


def path_curvature_score(path):
    pts = np.asarray(path, dtype=float)
    if pts.shape[0] < 3:
        return float("inf")
    closed = np.linalg.norm(pts[0] - pts[-1]) < 1e-6
    if closed:
        pts = pts[:-1]
    n = pts.shape[0]
    if n < 3:
        return float("inf")
    kappas = []
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        p0 = pts[im1]
        p1 = pts[i]
        p2 = pts[ip1]
        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            continue
        area2 = abs(np.cross(p1 - p0, p2 - p0))
        kappa = 2.0 * area2 / (a * b * c)
        kappas.append(kappa)
    if not kappas:
        return float("inf")
    return float(np.mean(kappas))


def max_jump(path):
    if len(path) < 2:
        return 0.0
    d = np.sqrt(np.sum(np.diff(path[:, :2], axis=0) ** 2, axis=1))
    return float(d.max())


def path_mask(path, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(path).astype(int)
    for i in range(len(pts) - 1):
        cv2.line(mask, tuple(pts[i]), tuple(pts[i + 1]), 255, 1)
    return mask > 0


def close_loop(path, threshold=2.0):
    if len(path) < 2:
        return path
    if np.linalg.norm(path[0] - path[-1]) <= threshold:
        return np.vstack([path, path[0]])
    return path


def draw_graph(base, nodes, edges, node_color=(0, 0, 255), edge_color=(255, 0, 0)):
    img = base.copy()
    for edge in edges:
        pts = [(p[1], p[0]) for p in edge["path"]]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], edge_color, 1)
    for (y, x) in nodes:
        cv2.circle(img, (x, y), 1, node_color, 1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-centerlines from a track map")
    parser.add_argument("--map", type=str, default=None, help="Map name (without extension)")
    parser.add_argument("--ext", type=str, default=None, help="Map image extension (e.g., .pgm)")
    parser.add_argument("--headless", action="store_true", help="Disable GUI windows and only save outputs")
    parser.add_argument("--max-paths", type=int, default=200, help="Maximum number of paths to enumerate")
    parser.add_argument("--max-states", type=int, default=50000, help="Maximum DFS states to explore")
    parser.add_argument("--max-edge-len", type=int, default=10000, help="Skip edges longer than this many pixels")
    parser.add_argument("--smooth-window", type=int, default=1, help="Smoothing window for centerlines (default: 1)")
    parser.add_argument("--prune-len", type=int, default=0, help="Prune skeleton branches shorter than this")
    parser.add_argument("--keep-top", type=int, default=2, help="Keep top-N longest distinct paths")
    parser.add_argument("--jump-threshold", type=float, default=5.0, help="Reject paths with large jumps in pixels")
    parser.add_argument("--dedup-iou", type=float, default=0.6, help="Drop near-duplicate paths by IoU")
    parser.add_argument("--no-dedup", action="store_true", help="Keep top-N paths even if they overlap")
    parser.add_argument("--max-cycles", type=int, default=20, help="Maximum cycles to extract for loop graphs")
    parser.add_argument("--min-cycle-len", type=int, default=0, help="Minimum cycle length in pixels (0 = auto)")
    args = parser.parse_args()

    module = ROOT
    config_file = os.path.join(module, "config", "params.yaml")
    with open(config_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)

    default_map, default_ext = load_map_defaults(module)
    input_map = args.map or default_map
    input_map_ext = args.ext or default_ext

    # Read map params
    yaml_file = os.path.join(module, "maps", input_map + ".yaml")
    with open(yaml_file, 'r') as stream:
        parsed_yaml = yaml.safe_load(stream)
    scale = parsed_yaml["resolution"]
    offset_x = parsed_yaml["origin"][0]
    offset_y = parsed_yaml["origin"][1]

    # Load map image
    img_path = os.path.join(module, "maps", input_map + input_map_ext)
    input_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        raise FileNotFoundError(f"Map image not found: {img_path}")

    negate = parsed_yaml.get("negate")
    input_img, inverted, invert_reason = auto_invert_if_needed(input_img, negate=negate)
    print(f"Auto invert: {'on' if inverted else 'off'} ({invert_reason})")
    track_mask = threshold_track(input_img, strict_white=True, track_value=254, track_tolerance=0)
    track_mask = remove_small_components(track_mask, min_area=70)

    # Skeletonize
    skel = skeletonize(track_mask)
    skel = prune_short_branches(skel, min_len=args.prune_len)
    if args.min_cycle_len == 0:
        skel_len = int(np.count_nonzero(skel))
        args.min_cycle_len = max(200, int(skel_len * 0.02))

    # Build graph and enumerate paths
    nodes, edges, adj = build_skeleton_graph(skel)
    degrees = {n: len(adj[n]) for n in nodes}
    branch_nodes = [n for n, d in degrees.items() if d >= 3]
    endpoints = [n for n, d in degrees.items() if d == 1]
    print(f"Graph: nodes={len(nodes)}, edges={len(edges)}, branches={len(branch_nodes)}, endpoints={len(endpoints)}")
    paths = enumerate_paths(
        nodes,
        edges,
        adj,
        max_paths=args.max_paths,
        max_states=args.max_states,
        max_edge_len=args.max_edge_len,
    )

    expected_count = None
    cycle_paths = extract_cycle_paths(
        nodes,
        edges,
        adj,
        max_cycles=args.max_cycles,
        min_len=args.min_cycle_len,
        dedup_iou=args.dedup_iou,
        mask_shape=track_mask.shape,
    )
    if cycle_paths:
        paths = [["__cycle__", i] for i in range(len(cycle_paths))]
        expected_count = len(cycle_paths)

    if not paths:
        print("No centerline paths found.")
        loop_path = extract_longest_contour_path(skel)
        if loop_path is None:
            raise SystemExit(1)
        paths = [["__loop__"]]
        expected_count = 1

    # Prepare images
    base = cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR)
    base[track_mask == 0] = (0, 0, 0)
    colors = [
        (0, 0, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255)
    ]

    # Decide target count from skeleton topology if available
    if expected_count is None:
        expected_count = len(paths)
    print(f"Expected centerlines from graph: {expected_count}")
    args.keep_top = expected_count

    # Default: avoid aggressive dedup for multi-lane output
    if args.keep_top > 1 and not args.no_dedup and args.dedup_iou >= 0.6:
        args.dedup_iou = 0.3

    # Convert paths to centerlines
    centerlines = []
    path_imgs = []
    path_infos = []
    path_infos_raw = []
    cycle_paths_cache = None
    if "cycle_paths" in locals():
        cycle_paths_cache = cycle_paths
    for edge_ids in paths:
        if edge_ids == ["__loop__"]:
            path = loop_path
        elif edge_ids[0] == "__cycle__":
            path = cycle_paths_cache[edge_ids[1]]
        else:
            start_node = edges[edge_ids[0]]["n1"]
            path = stitch_path(edges, edge_ids, start_node)
        path = smooth_path(path, window=args.smooth_window)
        path = close_loop(path, threshold=2.0)
        length = path_length(path)
        curv = path_curvature_score(path)
        path_infos_raw.append((curv, length, path))
        jump = max_jump(path)
        if jump <= args.jump_threshold:
            path_infos.append((curv, length, path))

    print(f"Candidate paths: {len(path_infos)} (filtered), {len(path_infos_raw)} (raw), dedup_iou={args.dedup_iou}")
    if not path_infos and path_infos_raw:
        print("No valid centerlines after filtering. Falling back to raw paths.")
        path_infos = list(path_infos_raw)
    if not path_infos:
        print("No valid centerlines after filtering.")
        raise SystemExit(1)
    if len(path_infos) < expected_count and path_infos_raw:
        print(f"Only {len(path_infos)} paths passed filters; relaxing to include raw paths up to {expected_count}.")
        path_infos = list(path_infos_raw)

    # Keep longest distinct paths
    path_infos.sort(key=lambda x: (x[0], -x[1]))
    kept = []
    kept_masks = []
    kept_scores = []
    for curv, length, path in path_infos:
        if len(kept) >= args.keep_top:
            break
        if args.no_dedup:
            kept.append(path)
            kept_scores.append((curv, length))
            continue
        mask = path_mask(path, track_mask.shape)
        is_dup = False
        for idx, km in enumerate(kept_masks):
            inter = np.logical_and(mask, km).sum()
            union = np.logical_or(mask, km).sum()
            iou = inter / max(1, union)
            if iou >= args.dedup_iou:
                # Keep the lower-curvature path.
                prev_curv, prev_len = kept_scores[idx]
                if curv < prev_curv or (curv == prev_curv and length > prev_len):
                    kept[idx] = path
                    kept_masks[idx] = mask
                    kept_scores[idx] = (curv, length)
                is_dup = True
                break
        if not is_dup:
            kept.append(path)
            kept_masks.append(mask)
            kept_scores.append((curv, length))

    # If dedup removes too much, fill remaining slots with longest raw paths.
    if len(kept) < args.keep_top:
        path_infos_raw.sort(key=lambda x: (x[0], -x[1]))
        for _, _, path in path_infos_raw:
            if len(kept) >= args.keep_top:
                break
            kept.append(path)

    # If still short, force-fill to expected_count even if duplicates overlap.
    if len(kept) < expected_count and path_infos_raw:
        path_infos_raw.sort(key=lambda x: (x[0], -x[1]))
        for _, _, path in path_infos_raw:
            if len(kept) >= expected_count:
                break
            kept.append(path)

    centerlines = kept
    for idx, path in enumerate(centerlines):
        color = colors[idx % len(colors)]
        path_imgs.append(overlay_path(base, path, color))

    # Save outputs
    output_dir = os.path.join(module, "outputs", input_map, "multi_centerlines")
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(output_dir):
        if filename.startswith("centerline_") and filename.endswith(".csv"):
            os.remove(os.path.join(output_dir, filename))
    for idx, path in enumerate(centerlines):
        path_m = transform_coords(path.astype(float), input_img.shape[0], scale, offset_x, offset_y)
        csv_path = os.path.join(output_dir, f"centerline_{idx:02d}.csv")
        save_csv(path_m, csv_path, header=["#x_m", "y_m"])
    print(f"Saved {len(centerlines)} centerlines to {output_dir}")

    # Save combined overlay (pixel space)
    combined = base.copy()
    for idx, path in enumerate(centerlines):
        combined = overlay_path(combined, path, colors[idx % len(colors)])
    cv2.imwrite(os.path.join(output_dir, "centerlines_all.png"), combined)

    graph_img = draw_graph(base, nodes, edges)
    cv2.imwrite(os.path.join(output_dir, "skeleton_graph.png"), graph_img)

    # Show grid
    if not args.headless and len(path_imgs) <= 12:
        show_images_grid(path_imgs, title="multi centerlines", cols=3)

    # Plot in meters
    # Save plot in meters separately (avoid overwriting pixel overlay)
    plt.figure(figsize=(10, 8))
    for path in centerlines:
        path_m = transform_coords(path.astype(float), input_img.shape[0], scale, offset_x, offset_y)
        plt.plot(path_m[:, 0], path_m[:, 1])
    plt.axis('equal')
    plt.savefig(os.path.join(output_dir, "centerlines_plot.png"), dpi=200)
    if not args.headless:
        plt.show()
