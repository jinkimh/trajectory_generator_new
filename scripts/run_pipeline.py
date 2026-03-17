import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import subprocess
import yaml


def load_yaml(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def resolve_placeholders(data, params):
    if isinstance(data, str):
        try:
            return data.format_map(params)
        except (KeyError, ValueError):
            return data
    if isinstance(data, list):
        return [resolve_placeholders(item, params) for item in data]
    if isinstance(data, dict):
        return {k: resolve_placeholders(v, params) for k, v in data.items()}
    return data


def add_args(cmd, args):
    for key, value in (args or {}).items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            # Support multi-arg flags and repeated flags (e.g., low_mu_rect)
            if all(isinstance(v, (list, tuple)) for v in value):
                for item in value:
                    if len(item) == 0:
                        continue
                    cmd.append(flag)
                    cmd.extend([str(v) for v in item])
            else:
                cmd.append(flag)
                cmd.extend([str(v) for v in value])
            continue
        cmd.extend([flag, str(value)])
    return cmd


def run_step(name, cmd, expected_paths=None):
    print(f"\n==> {name}")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(out)
        raise RuntimeError(explain_failure(name, out))
    if expected_paths:
        missing = [p for p in expected_paths if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                f"{name} finished but expected outputs are missing: {missing}\n"
                f"Check input map/params or logs above."
            )
    return out


def explain_failure(name, output):
    hints = []
    if "spline normals are crossed" in output or "pair of normals is crossed" in output:
        hints.append(
            "Normals crossing: increase smoothing (preprocess smooth_blur / contour), "
            "or set min_track_width to reduce tight curvature."
        )
    if "LBX" in output or "Ill-posed problem" in output:
        hints.append(
            "Infeasible bounds: track width likely smaller than width_opt. "
            "Use --min-track-width, reduce width_opt, or lower width_scale."
        )
    if "Maximum_Iterations_Exceeded" in output:
        hints.append(
            "Max iterations exceeded: try higher ipopt_max_iter, more smoothing, "
            "or adjust width_scale/min_track_width."
        )
    if "No grid points found on track" in output:
        hints.append(
            "Friction map has no points: lower track_thresh, use processed map, "
            "or increase cellwidth."
        )
    if "No module named 'casadi'" in output:
        hints.append(f"Activate the venv: source {ROOT}/.venv/bin/activate")
    if "dependency check skipped" in output:
        hints.append("Dependency warning is non-fatal; can be ignored if run completes.")
    hint_text = "\n- ".join(hints) if hints else "See logs above for details."
    return f"{name} failed.\n- {hint_text}"


def main():
    module = ROOT
    scripts_dir = os.path.join(module, "scripts")
    cfg_path = os.path.join(module, "config", "pipeline.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = load_yaml(cfg_path)
    map_name = cfg.get("map_name")
    map_ext = cfg.get("map_ext", ".pgm")

    py = sys.executable
    params = {"map_name": map_name, "map_ext": map_ext}
    tracksplit_cfg = resolve_placeholders(cfg.get("tracksplit", {}), params)
    if tracksplit_cfg.get("enabled", False):
        args = tracksplit_cfg.get("args", {})
        cmd = [py, os.path.join(scripts_dir, "track_boundary_split.py"), "--map", map_name, "--ext", map_ext]
        cmd = add_args(cmd, args)
        suffix = args.get("suffix", "_tracksplit")
        out_dir = args.get("output_dir", None)
        maps_dir = os.path.join(module, "maps")
        if out_dir and os.path.abspath(out_dir) != os.path.abspath(maps_dir):
            raise RuntimeError(
                "tracksplit output_dir must be the maps directory when running the pipeline. "
                f"Got: {out_dir}"
            )
        expected = [os.path.join(maps_dir, f"{map_name}{suffix}{map_ext}")]
        if args.get("save_yaml", False):
            expected.append(os.path.join(maps_dir, f"{map_name}{suffix}.yaml"))
        run_step("track_boundary_split", cmd, expected)
        map_name = f"{map_name}{suffix}"
        params = {"map_name": map_name, "map_ext": map_ext}

    preprocess_cfg = resolve_placeholders(cfg.get("preprocess", {}), params)
    if preprocess_cfg.get("enabled", True):
        args = preprocess_cfg.get("args", {})
        cmd = [py, os.path.join(scripts_dir, "preprocess_map.py"), "--map", map_name, "--ext", map_ext]
        cmd = add_args(cmd, args)
        suffix = args.get("suffix", "_processed")
        processed_img = os.path.join(module, "maps", f"{map_name}{suffix}{map_ext}")
        processed_yaml = os.path.join(module, "maps", f"{map_name}{suffix}.yaml")
        run_step("preprocess_map", cmd, [processed_img, processed_yaml])

    fric_cfg = resolve_placeholders(cfg.get("frictionmap", {}), params)
    if fric_cfg.get("enabled", False):
        fric_map = fric_cfg.get("map", f"{map_name}_processed")
        fric_ext = fric_cfg.get("ext", map_ext)
        args = fric_cfg.get("args", {})
        out_dir = args.get("out_dir", os.path.join(module, "inputs", "frictionmaps"))
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(module, out_dir)
        args["out_dir"] = out_dir
        map_file = os.path.join(module, "maps", f"{fric_map}{fric_ext}")
        yaml_file = os.path.join(module, "maps", f"{fric_map}.yaml")
        cmd = [py, os.path.join(scripts_dir, "gen_frictionmap_from_map.py"),
               "--map-file", map_file, "--yaml-file", yaml_file]
        cmd = add_args(cmd, args)
        out_name = args.get("out_name", map_name)
        tpamap = os.path.join(out_dir, f"{out_name}_tpamap.csv")
        tpadata = os.path.join(out_dir, f"{out_name}_tpadata.json")
        run_step("gen_frictionmap_from_map", cmd, [tpamap, tpadata])

    center_cfg = resolve_placeholders(cfg.get("centerlines", {}), params)
    if center_cfg.get("enabled", True):
        center_map = center_cfg.get("map", f"{map_name}_processed")
        center_ext = center_cfg.get("ext", map_ext)
        args = center_cfg.get("args", {})
        cmd = [py, os.path.join(scripts_dir, "lane_generator_multi_centerline.py"),
               "--map", center_map, "--ext", center_ext]
        cmd = add_args(cmd, args)
        center_out = os.path.join(module, "outputs", center_map, "multi_centerlines", "centerline_00.csv")
        run_step("lane_generator_multi_centerline", cmd, [center_out])

    mintime_cfg = resolve_placeholders(cfg.get("mintime", {}), params)
    if mintime_cfg.get("enabled", True):
        mint_map = mintime_cfg.get("map", map_name)
        center_map = mintime_cfg.get("centerline_map", f"{map_name}_processed")
        args = mintime_cfg.get("args", {})
        cmd = [py, os.path.join(scripts_dir, "main_globaltraj_multi.py"),
               "--map", mint_map, "--centerline-map", center_map]
        cmd = add_args(cmd, args)
        summary = os.path.join(module, "outputs", mint_map, "multi_mintime", "summary.json")
        run_step("main_globaltraj_multi", cmd, [summary])


if __name__ == "__main__":
    main()
