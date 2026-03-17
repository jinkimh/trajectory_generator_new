import os

import yaml


def read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_map_defaults(module_root, fallback_name="Bexco", fallback_ext=".pgm"):
    pipeline_cfg = read_yaml(os.path.join(module_root, "config", "pipeline.yaml"))
    params_cfg = read_yaml(os.path.join(module_root, "config", "params.yaml"))
    map_name = pipeline_cfg.get("map_name") or params_cfg.get("map_name") or fallback_name
    map_ext = pipeline_cfg.get("map_ext") or params_cfg.get("map_img_ext") or fallback_ext
    return map_name, map_ext
