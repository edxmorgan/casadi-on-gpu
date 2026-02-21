#!/usr/bin/env bash
# Author: edward morgan
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "[clean] repo: ${REPO_ROOT}"

remove_dir_if_exists() {
  local d="$1"
  if [ -d "${d}" ]; then
    rm -rf "${d}"
    echo "[clean] removed dir: ${d}"
  else
    echo "[clean] dir not found (skip): ${d}"
  fi
}

remove_file_if_exists() {
  local f="$1"
  if [ -f "${f}" ]; then
    rm -f "${f}"
    echo "[clean] removed file: ${f}"
  else
    echo "[clean] file not found (skip): ${f}"
  fi
}

# 1) Build artifacts
remove_dir_if_exists "${REPO_ROOT}/build"
remove_dir_if_exists "${REPO_ROOT}/_skbuild"

# 2) Generated artifacts
if [ -d "${REPO_ROOT}/src/generated" ]; then
  while IFS= read -r -d '' f; do
    rm -f "${f}"
    echo "[clean] removed generated: ${f}"
  done < <(
    find "${REPO_ROOT}/src/generated" -maxdepth 1 -type f \
      \( -name "*.cu" -o -name "*.cuh" -o -name "kernels_manifest.json" \) -print0
  )
else
  echo "[clean] dir not found (skip): ${REPO_ROOT}/src/generated"
fi

remove_file_if_exists "${REPO_ROOT}/src/python/casadi_on_gpu_kernel_registry.cu"

# 3) Installed Python module (casadi_on_gpu / casadi-on-gpu)
if command -v python3 >/dev/null 2>&1; then
  python3 -m pip uninstall -y casadi-on-gpu --break-system-packages >/dev/null 2>&1 || \
    python3 -m pip uninstall -y casadi-on-gpu >/dev/null 2>&1 || true
  python3 -m pip uninstall -y casadi_on_gpu --break-system-packages >/dev/null 2>&1 || \
    python3 -m pip uninstall -y casadi_on_gpu >/dev/null 2>&1 || true

  python3 - <<'PY'
import glob
import os
import shutil
import site
import sysconfig

paths = set()
try:
    paths.add(site.getusersitepackages())
except Exception:
    pass

for key in ("platlib", "purelib"):
    p = sysconfig.get_path(key)
    if p:
        paths.add(p)

try:
    for p in site.getsitepackages():
        paths.add(p)
except Exception:
    pass

patterns = [
    "casadi_on_gpu*.so",
    "casadi_on_gpu*.pyd",
    "casadi_on_gpu*.dll",
    "casadi_on_gpu-*.dist-info",
    "casadi_on_gpu-*.egg-info",
    "kernels_manifest.json",
]

removed = 0
for base in sorted(p for p in paths if p and os.path.isdir(p)):
    for pat in patterns:
        for path in glob.glob(os.path.join(base, pat)):
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue
            removed += 1
            print(f"[clean] removed python artifact: {path}")

print(f"[clean] python artifact removals: {removed}")
PY

  python3 - <<'PY'
import importlib.util
for name in ("cog", "casadi_on_gpu"):
    spec = importlib.util.find_spec(name)
    where = spec.origin if spec else "not_found"
    print(f"[clean] import check {name}: {where}")
PY
else
  echo "[clean] python3 not found (skip python uninstall/purge)"
fi

echo "[clean] done"
