#!/usr/bin/env bash
# install.sh — build / refresh the shared CPU TokEye install on Omega.
#
#   /fusion/projects/codes/tokeye/
#     src/          git checkout (branch diiid), installed editable into env/
#     env/          conda env: Python 3.13 + MDSplus + Qt (conda-forge),
#                   CPU-only torch + OpenVINO + tokeye[app] (pip)
#     models/       pre-staged weights + cached OpenVINO IR (read-only for users)
#     bin/tokeye    launcher (the only thing the module puts on PATH)
#     modulefiles/tokeye/default.lua
#
# Usage (from a login/somega node with internet; members of group tokeye):
#   /fusion/projects/codes/tokeye/src/deploy/omega/fusion/install.sh [--pull] [--rebuild]
#     --pull     git pull --ff-only the checkout first
#     --rebuild  delete and recreate env/ from scratch
# Idempotent: re-running only refreshes what changed.
set -euo pipefail

pull=0
rebuild=0
for arg in "$@"; do
  case "$arg" in
    --pull) pull=1 ;;
    --rebuild) rebuild=1 ;;
    -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
    *) echo "install.sh: unknown argument $arg" >&2; exit 2 ;;
  esac
done

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SRC="$(cd "$here/../../.." && pwd)"
ROOT="${TOKEYE_ROOT:-$(dirname "$SRC")}"
ENV="$ROOT/env"
MODELS="$ROOT/models"
MAMBA="${MAMBA:-/fusion/projects/codes/conda/mambaforge/bin/mamba}"
# Fresh download/build cache per run, deleted on exit: /cscratch is swept daily
# (files go, folders stay), and a half-deleted conda/pip cache can yield a
# broken env. Nothing in the finished install lives here.
if [[ -n "${TOKEYE_BUILD_DIR:-}" ]]; then
  BUILD="$TOKEYE_BUILD_DIR"            # caller's own dir: never deleted
else
  BUILD="$(mktemp -d "/cscratch/$USER/tokeye-build.XXXXXX" 2>/dev/null || mktemp -d)"
  trap 'rm -rf "$BUILD"' EXIT          # only the temp dir we just made
fi

umask 002
unset PYTHONPATH PYTHONHOME
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$BUILD/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$BUILD/pip}"
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$BUILD"

say() { printf '\n== %s\n' "$*"; }

if [[ "$pull" -eq 1 ]]; then
  say "updating $SRC"
  git -C "$SRC" pull --ff-only
fi

# 1. conda env (Python, MDSplus, Qt) ------------------------------------------
if [[ "$rebuild" -eq 1 && -d "$ENV" ]]; then
  say "removing $ENV"
  rm -rf "$ENV"
fi
if [[ ! -x "$ENV/bin/python" ]]; then
  say "creating conda env at $ENV"
  "$MAMBA" env create -p "$ENV" -f "$SRC/deploy/omega/environment-omega.yml"
fi

# 2. pip: CPU torch first (so the [app] extra never drags in CUDA wheels) -----
say "installing CPU torch + OpenVINO + tokeye (editable)"
"$ENV/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch<2.11" torchvision
"$ENV/bin/pip" install "openvino>=2025.1"
"$ENV/bin/pip" install -e "$SRC[app]"
if "$ENV/bin/python" -c 'import torch,sys; sys.exit(0 if torch.version.cuda else 1)'; then
  echo "install.sh: WARNING: a CUDA torch build got installed; reinstalling the CPU wheel" >&2
  "$ENV/bin/pip" install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cpu "torch<2.11" torchvision
fi

# 3. publish launcher + modulefile + docs -------------------------------------
say "publishing launcher and modulefile"
mkdir -p "$ROOT/bin" "$ROOT/modulefiles/tokeye" "$MODELS/openvino"
install -m 775 "$here/bin/tokeye" "$ROOT/bin/tokeye"
install -m 775 "$SRC/deploy/omega/tokeye-app.sh" "$ROOT/bin/tokeye-app"
install -m 664 "$here/modulefiles/tokeye/default.lua" "$ROOT/modulefiles/tokeye/default.lua"
install -m 664 "$here/README.md" "$ROOT/README.md"
install -m 664 "$here/about.txt" "$ROOT/about.txt"

# 4. weights + OpenVINO IR (so users never touch Hugging Face) -----------------
say "staging model weights in $MODELS"
"$ENV/bin/python" - "$MODELS" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tokeye.hub import DEFAULT_REPO_ID, MODEL_REGISTRY

models = Path(sys.argv[1])
for spec in MODEL_REGISTRY.values():
    dst = models / spec.filename
    if dst.is_file():
        print(f"  have {dst.name}")
        continue
    try:
        src = hf_hub_download(spec.repo_id or DEFAULT_REPO_ID, spec.filename)
    except Exception as exc:  # optional models may be private / missing
        print(f"  skip {spec.name}: {exc.__class__.__name__}: {exc}")
        continue
    shutil.copyfile(src, dst)
    dst.chmod(0o664)
    print(f"  staged {dst.name}")
PY
say "converting the default model to OpenVINO IR"
# IRs are keyed on the weights' content; drop any stale/foreign ones first.
find "$MODELS/openvino" -maxdepth 1 -type f \( -name '*.xml' -o -name '*.bin' \) -delete
TOKEYE_MODEL_DIR="$MODELS" TOKEYE_OV_DIR="$MODELS/openvino" "$ENV/bin/python" -c '
from tokeye import accel
r = accel.load_runner(backend="openvino")
print("  OpenVINO runner ready:", r.model_name, r.precision)'

# 5. permissions: group tokeye maintains, everyone reads ------------------------
say "fixing permissions"
chmod -R g+rwX,o+rX "$ROOT/bin" "$ROOT/modulefiles" "$MODELS" 2>/dev/null || true
chmod -R o+rX "$ENV" 2>/dev/null || true

# 6. smoke test -----------------------------------------------------------------
say "smoke test"
"$ROOT/bin/tokeye" --version
# exactly what users get: the staged weights + shared IR, no Hugging Face
TOKEYE_MODEL_DIR="$MODELS" TOKEYE_OV_DIR="$MODELS/openvino" HF_HUB_OFFLINE=1 \
  "$ENV/bin/python" "$here/selftest.py"

cat <<EOF

TokEye installed in $ROOT
    module use $ROOT/modulefiles
    module load tokeye
    tokeye 190904 MPI66M067D
EOF
