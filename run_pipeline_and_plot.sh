#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH="${ROOT_DIR}/config/pipeline.yaml"

MAP_NAME=""
MAP_PATH=""
MAP_EXT=""
CSV_OVERRIDE=""
SKIP_PIPELINE=0

PYTHON="python3"
if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT_DIR}/.venv/bin/python3"
fi

usage() {
  cat <<EOF
Usage:
  $0 <map_name>
  $0 --map <map_name_or_path> [--map-ext .pgm] [--csv /path/to.csv] [--no-pipeline]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --map)
      MAP_PATH="${2:-}"
      shift 2
      ;;
    --map-ext)
      MAP_EXT="${2:-}"
      shift 2
      ;;
    --csv)
      CSV_OVERRIDE="${2:-}"
      shift 2
      ;;
    --no-pipeline|--skip-pipeline)
      SKIP_PIPELINE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${MAP_NAME}" ]]; then
        MAP_NAME="$1"
        shift
      else
        echo "Unknown argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

if [[ -n "${MAP_PATH}" ]]; then
  if [[ -f "${MAP_PATH}" ]]; then
    base="$(basename "${MAP_PATH}")"
    MAP_NAME="${base%.*}"
    if [[ -z "${MAP_EXT}" ]]; then
      MAP_EXT=".${base##*.}"
    fi
  else
    MAP_NAME="${MAP_PATH}"
  fi
fi

if [[ -z "${MAP_NAME}" ]]; then
  MAP_NAME="$(
    CFG_PATH="${CFG_PATH}" "${PYTHON}" - <<'PY'
import os
import yaml
from pathlib import Path

cfg = Path(os.environ["CFG_PATH"])
data = yaml.safe_load(cfg.read_text()) or {}
print(data.get("map_name", ""))
PY
  )"
fi

if [[ -z "${MAP_NAME}" ]]; then
  usage
  exit 1
fi

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "Missing config: ${CFG_PATH}"
  exit 1
fi

BACKUP="${CFG_PATH}.bak.$(date +%s)"
cp "${CFG_PATH}" "${BACKUP}"
cleanup() {
  mv "${BACKUP}" "${CFG_PATH}"
}
trap cleanup EXIT

MAP_EXT="$(
  CFG_PATH="${CFG_PATH}" MAP_NAME="${MAP_NAME}" MAP_EXT="${MAP_EXT}" "${PYTHON}" - <<'PY'
import os
import yaml
from pathlib import Path

cfg = Path(os.environ["CFG_PATH"])
data = yaml.safe_load(cfg.read_text()) or {}
data["map_name"] = os.environ["MAP_NAME"]
map_ext_env = os.environ.get("MAP_EXT", "").strip()
if map_ext_env:
    data["map_ext"] = map_ext_env
cfg.write_text(yaml.safe_dump(data, sort_keys=False))
print(data.get("map_ext", ".pgm"))
PY
)"

if [[ "${SKIP_PIPELINE}" -eq 0 ]]; then
  echo "Running pipeline for map_name=${MAP_NAME} (map_ext=${MAP_EXT})"
  "${PYTHON}" "${ROOT_DIR}/scripts/run_pipeline.py"
else
  echo "Skipping pipeline run."
fi

MINT_DIR="${ROOT_DIR}/outputs/${MAP_NAME}_tracksplit/multi_mintime"
SUMMARY="${MINT_DIR}/summary.json"
MINT_CSV="${MINT_DIR}/traj_race_cl_00.csv"
CENTER_CSV="${ROOT_DIR}/outputs/${MAP_NAME}_tracksplit_processed/multi_centerlines/centerline_00.csv"
MAP_FOR_PLOT="${MAP_NAME}_tracksplit_processed"

STATUS="$(
  SUMMARY="${SUMMARY}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["SUMMARY"])
if not p.exists():
    print("")
    raise SystemExit(0)
data = json.loads(p.read_text())
if isinstance(data, list) and data:
    print(str(data[0].get("status", "")))
elif isinstance(data, dict):
    print(str(data.get("status", "")))
else:
    print("")
PY
)"

CSV_TO_PLOT=""
OUT_PATH=""
if [[ "${STATUS}" == "ok" && -f "${MINT_CSV}" ]]; then
  CSV_TO_PLOT="${MINT_CSV}"
  OUT_PATH="${MINT_DIR}/raceline_overlay_latest.png"
  echo "Mintime raceline found. Plotting ${CSV_TO_PLOT}"
elif [[ -n "${CSV_OVERRIDE}" && -f "${CSV_OVERRIDE}" ]]; then
  CSV_TO_PLOT="${CSV_OVERRIDE}"
  OUT_PATH="${MINT_DIR}/raceline_overlay_latest.png"
  echo "Using provided CSV: ${CSV_TO_PLOT}"
else
  CSV_TO_PLOT="${CENTER_CSV}"
  OUT_PATH="${MINT_DIR}/mincurv_overlay_latest.png"
  echo "Mintime raceline not found. Falling back to centerline plot: ${CSV_TO_PLOT}"
fi

if [[ ! -f "${CSV_TO_PLOT}" ]]; then
  echo "CSV not found: ${CSV_TO_PLOT}"
  exit 1
fi

"${PYTHON}" "${ROOT_DIR}/scripts/plot_raceline_on_map.py" \
  --csv "${CSV_TO_PLOT}" \
  --map "${MAP_FOR_PLOT}" \
  --map-ext "${MAP_EXT}" \
  --out "${OUT_PATH}"

echo "Saved overlay: ${OUT_PATH}"
