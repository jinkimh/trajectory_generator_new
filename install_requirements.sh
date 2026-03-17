#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQ_FILE="${ROOT_DIR}/requirements.txt"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Venv not found at ${VENV_DIR}. Run ./setup_venv.sh first."
  exit 1
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

# quadprog wheels can break on some setups; rebuild from source to match local libs
python - <<'PY'
import subprocess
import sys
import importlib.util
from pathlib import Path

def run(cmd):
    return subprocess.check_call(cmd)

try:
    # quadprog 0.1.7 needs Cython < 3; force a compatible build toolchain
    run([sys.executable, "-m", "pip", "install", "Cython==0.29.36"])
    run([sys.executable, "-m", "pip", "uninstall", "-y", "quadprog"])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-binary=quadprog",
        "--no-build-isolation",
        "quadprog==0.1.7",
    ])
except subprocess.CalledProcessError:
    print("WARNING: quadprog source build failed. You may need build tools (e.g., gfortran).")
    print("Try: sudo apt install -y gfortran build-essential python3-dev")

# Patch trajectory_planning_helpers for SciPy x0 shape regression if needed
try:
    spec = importlib.util.find_spec("trajectory_planning_helpers")
    if spec and spec.origin:
        base = Path(spec.origin).resolve().parent
        target = base / "spline_approximation.py"
        if target.exists():
            text = target.read_text()
            needle = "x0=t_glob_guess_cl[i]"
            repl = "x0=np.array([t_glob_guess_cl[i]])"
            updated = text
            if needle in updated and repl not in updated:
                updated = updated.replace(needle, repl)
                print(f"Patched {target} x0 shape.")
            dist_old = (
                "def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):\\n"
                "    s = interpolate.splev(t_glob, path)\\n"
                "    return spatial.distance.euclidean(p, s)\\n"
            )
            dist_new = (
                "def dist_to_p(t_glob: np.ndarray, path: list, p: np.ndarray):\\n"
                "    t = float(np.asarray(t_glob).ravel()[0])\\n"
                "    s = interpolate.splev(t, path)\\n"
                "    s = np.asarray(s).reshape(-1)\\n"
                "    return spatial.distance.euclidean(p, s)\\n"
            )
            if dist_old in updated:
                updated = updated.replace(dist_old, dist_new)
                print(f"Patched {target} dist_to_p scalar handling.")
            if updated != text:
                target.write_text(updated)
except Exception as exc:
    print(f"WARNING: spline_approximation patch failed: {exc}")
PY

echo "Installed requirements from ${REQ_FILE}"
