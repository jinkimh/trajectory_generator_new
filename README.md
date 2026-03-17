# trajectory_generator_new

ROS2 `ament_python` package for map preprocessing, centerline extraction, friction map generation, and global
raceline optimization. This is a refactored layout of the original `trajectory_generator` with the same
`run_pipeline.py` behavior, but organized into `scripts/`, `config/`, `maps/`, `inputs/`, and `outputs/`.

## Layout

```
trajectory_generator_new/
  scripts/                # executable Python scripts (pipeline + tools)
  config/                 # pipeline.yaml and other configs
  maps/                   # map images + yaml
  inputs/                 # frictionmaps, veh_dyn_info, tracks
  outputs/                # generated centerlines/racelines/results
  helper_funcs_glob/      # shared helpers (unchanged)
  opt_mintime_traj/       # mintime optimizer (unchanged)
  frictionmap/            # frictionmap package (unchanged)
  params/                 # racecar params (unchanged)
  run_pipeline_and_plot.sh # run pipeline + plot result
  requirements.txt
  setup_venv.sh
  install_requirements.sh
  setup.py / setup.cfg / package.xml
```

## Quick Start

1) Create venv and install requirements:
```bash
cd /home/jin/ros2_prj/trajectory_generator_new
./setup_venv.sh
source .venv/bin/activate
```

2) Run the pipeline:
```bash
python3 /home/jin/ros2_prj/trajectory_generator_new/scripts/run_pipeline.py
```

3) Run pipeline + auto-plot:
```bash
./run_pipeline_and_plot.sh
```

## Architecture (high-level)

```
maps/<map>.(png|pgm)
  -> track_boundary_split.py (optional)
  -> preprocess_map.py
  -> gen_frictionmap_from_map.py (optional)
  -> lane_generator_multi_centerline.py
  -> main_globaltraj_multi.py (mintime/mincurv)
  -> plot_raceline_on_map.py (overlay)
```

## Pipeline Configuration

Edit `config/pipeline.yaml`:

- `map_name`: base map name without extension
- `map_ext`: `.png` or `.pgm`
- `tracksplit`: boundary-only map splitting (optional)
- `preprocess`: map preprocessing (binary drivable region)
- `frictionmap`, `centerlines`, `mintime`: downstream steps

Placeholders:
- Use `{map_name}` and `{map_ext}` in `pipeline.yaml`. They are resolved at runtime.
  Example:
  - `map: "{map_name}_processed"`
  - `ext: "{map_ext}"`

Notes:
- If `tracksplit.enabled: true`, the pipeline automatically switches `map_name` to
  `{map_name}{suffix}` for subsequent stages.
- `maps/` contains both images and yaml files used by downstream steps.
- RViz/launch files have been removed from this package; use `plot_raceline_on_map.py` for visualization.

## Scripts and Usage

All scripts live in `scripts/` and expect project paths relative to the package root.
Run them from anywhere (the scripts prepend the package root to `sys.path`).

### 1) `track_boundary_split.py`
Split donut-style boundary-only maps into outer/inner boundary and track region.
```bash
python3 scripts/track_boundary_split.py \
  --map icra --ext .png \
  --edge-method canny --canny-low 30 --canny-high 120 \
  --edge-dilate 3 --close-ksize 5 --min-edge-area 0 \
  --suffix _tracksplit --save-yaml
```
Outputs in `maps/`:
- `{map}_tracksplit.{ext}`
- `{map}_tracksplit_outer.{ext}`
- `{map}_tracksplit_inner.{ext}`
- `{map}_tracksplit_outside.{ext}`
- `{map}_tracksplit_hole.{ext}`
- `{map}_tracksplit_overlay.png`
- `{map}_tracksplit.yaml` (if `--save-yaml`)

### 2) `preprocess_map.py`
Binary preprocessing to extract drivable track region.
```bash
python3 scripts/preprocess_map.py --map levine_2nd_tracksplit --ext .pgm --preset icra_2
```
Outputs in `maps/`:
- `{map}_processed.{ext}`
- `{map}_processed.yaml`

### 3) `gen_frictionmap_from_map.py`
Generate friction map CSV/JSON from a map.
```bash
python3 scripts/gen_frictionmap_from_map.py \
  --map-file maps/levine_2nd_tracksplit_processed.pgm \
  --yaml-file maps/levine_2nd_tracksplit_processed.yaml \
  --out-dir inputs/frictionmaps --out-name levine_2nd
```
Outputs:
- `inputs/frictionmaps/{out_name}_tpamap.csv`
- `inputs/frictionmaps/{out_name}_tpadata.json`

### 4) `lane_generator_multi_centerline.py`
Extract centerlines from processed map.
```bash
python3 scripts/lane_generator_multi_centerline.py --map icra_2_processed --ext .pgm --headless
```
Outputs:
- `outputs/{map}/multi_centerlines/centerline_*.csv`
- `centerlines_all.png`, `skeleton_graph.png`, `centerlines_plot.png`

### 5) `main_globaltraj_multi.py`
Optimize raceline from centerlines (mintime/mincurv).
```bash
python3 scripts/main_globaltraj_multi.py --map icra_2 --centerline-map icra_2_processed --headless
```
Outputs:
- `outputs/{map}/multi_mintime/summary.json`
- `traj_race_cl_*.csv`, `racelines_overlay.png`, `mintime_ranking.csv`

### 6) `plot_raceline_on_map.py`
Overlay raceline CSV on the **original map** (default). Supports speed color + labels.
```bash
python3 scripts/plot_raceline_on_map.py \
  --csv outputs/icra_2_tracksplit/multi_mintime/traj_race_cl_00.csv \
  --map icra_2.pgm
```
Notes:
- `--map` can be a full path or a name with extension.
- Use `--use-processed` to force processed map.

### 7) `main_globaltraj.py`
Single-path optimization for a reference track in `inputs/tracks`.
```bash
python3 scripts/main_globaltraj.py
```
Outputs:
- `outputs/{map}/traj_race_cl.csv` (and optional exports)

### 8) `main_gen_frictionmap.py`
Legacy frictionmap generation from reference track files.
```bash
python3 scripts/main_gen_frictionmap.py
```

### 9) `check_image_values.py`
Quick histogram/value inspection for map images.
```bash
python3 scripts/check_image_values.py
```

### 10) `raceline_scripts.py`
Utility helpers for plotting/analysis. Use per function.

### 11) `run_pipeline_and_plot.sh`
Runs the full pipeline and plots the best raceline if mintime succeeds,
otherwise falls back to centerline overlay.
```bash
./run_pipeline_and_plot.sh
./run_pipeline_and_plot.sh icra_2
./run_pipeline_and_plot.sh --map ./maps/icra_2.pgm --csv ./outputs/icra_2_tracksplit/multi_mintime/traj_race_cl_00.csv --no-pipeline
```
Notes:
- If no arguments are given, it reads `map_name` from `config/pipeline.yaml`.

## Running as a ROS2 Package

Build:
```bash
cd /home/jin/ros2_prj
colcon build --packages-select trajectory_generator_new
```

You can still run scripts directly:
```bash
python3 /home/jin/ros2_prj/trajectory_generator_new/scripts/run_pipeline.py
```

## Troubleshooting

- `No module named 'casadi'`: activate venv
  ```bash
  source /home/jin/ros2_prj/trajectory_generator_new/.venv/bin/activate
  ```
- `matplotlib` backend errors: install `python3-tk`
  ```bash
  sudo apt install python3-tk
  ```
- `quadprog` build errors: install headers
  ```bash
  sudo apt install python3-dev gfortran build-essential
  ```
- `install_requirements.sh` rebuilds `quadprog` from source and applies a small
  compatibility patch for `trajectory_planning_helpers` (SciPy x0 shape).

## Notes

- All outputs are written under `outputs/` inside this package.
- `map_ext` controls `.png` vs `.pgm` everywhere (pipeline resolves `{map_ext}`).
- If a step seems to use the wrong map name, verify `map_name`, `map_ext`,
  and whether `tracksplit` is enabled.
