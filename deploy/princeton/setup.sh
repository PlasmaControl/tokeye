#!/usr/bin/env bash
# One-shot (idempotent) setup of the shared TokEye install on stellar.
#
#   ./deploy/princeton/setup.sh [--bashrc] [--skip-download]
#
# Run it from any checkout by a member of the `kolemen` group, on a node with
# internet (login or stellar-vis; compute nodes are offline). It:
#   1. syncs the uv venv (app + gui + princeton extras) inside the repo,
#   2. prefetches the model weights into a shared HF cache,
#   3. renders the Tcl group modulefile into the group Modules tree,
#   4. opens group read permissions (setgid dir + umask 002 + g+rX sweep).
#
# Everything is overridable for testing / relocation:
#   TOKEYE_DIR           repo root            (default /projects/EKOLEMEN/tokeye)
#   TOKEYE_MODULES_ROOT  group modulefiles    (default /projects/EKOLEMEN/Modules/modulefiles-shared)
#   TOKEYE_HF_HOME       shared HF cache      (default $TOKEYE_DIR/.cache/huggingface)
#   TOKEYE_FOUNDATION_DIR shot archive        (default /scratch/gpfs/EKOLEMEN/foundation_model)
#
# --bashrc  append the one-time `module use` line to ~/.bashrc (once).
#           Needed because stellar's auto group-module discovery probes
#           /projects/KOLEMEN (the uppercased unix group `kolemen`) but the
#           group space is /projects/EKOLEMEN — so it never fires here.
set -euo pipefail
umask 002  # keep everything group-writable-dir / group-readable for `kolemen`

ROOT="${TOKEYE_DIR:-/projects/EKOLEMEN/tokeye}"
MODULES_ROOT="${TOKEYE_MODULES_ROOT:-/projects/EKOLEMEN/Modules/modulefiles-shared}"
HF_HOME_DIR="${TOKEYE_HF_HOME:-$ROOT/.cache/huggingface}"
FOUNDATION_DIR="${TOKEYE_FOUNDATION_DIR:-/scratch/gpfs/EKOLEMEN/foundation_model}"
TEMPLATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/modulefiles/tokeye.tcl"

DO_BASHRC=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
    case "$arg" in
        --bashrc) DO_BASHRC=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        *) echo "usage: setup.sh [--bashrc] [--skip-download]" >&2; exit 2 ;;
    esac
done

say() { printf '\n==> %s\n' "$*"; }

# ── 0. sanity ─────────────────────────────────────────────────────────────────
[ -f "$ROOT/pyproject.toml" ] || {
    echo "error: $ROOT does not look like the tokeye repo (no pyproject.toml)" >&2
    echo "       set TOKEYE_DIR to the checkout you want to install" >&2
    exit 1
}
branch="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [ "$branch" != "princeton" ]; then
    echo "warning: $ROOT is on branch '$branch', not 'princeton' — the" >&2
    echo "         foundation source / Princeton tab may be missing" >&2
fi
command -v uv >/dev/null 2>&1 || {
    echo "error: uv not found. Install it first:" >&2
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
}

# ── 1. venv ───────────────────────────────────────────────────────────────────
say "Syncing the uv environment in $ROOT (app + gui + princeton extras)"
(cd "$ROOT" && uv sync --extra app --extra gui --extra princeton)

# ── 2. model weights (compute nodes are offline — prefetch into a shared cache) ─
if [ "$SKIP_DOWNLOAD" = 1 ]; then
    say "Skipping weight download (--skip-download)"
else
    say "Prefetching model weights into $HF_HOME_DIR"
    mkdir -p "$HF_HOME_DIR"
    (cd "$ROOT" && HF_HOME="$HF_HOME_DIR" uv run tokeye download)
fi

# ── 3. group modulefile ───────────────────────────────────────────────────────
version="$(cd "$ROOT" && uv run tokeye --version 2>/dev/null | tail -n1)"
[ -n "$version" ] || version="0.0.0"
module_dir="$MODULES_ROOT/tokeye"
module_file="$module_dir/$version"
say "Rendering the Tcl modulefile -> $module_file"
mkdir -p "$module_dir"
sed -e "s|@ROOT@|$ROOT|g" \
    -e "s|@MODULES_ROOT@|$MODULES_ROOT|g" \
    -e "s|@FOUNDATION_DIR@|$FOUNDATION_DIR|g" \
    -e "s|@HF_HOME@|$HF_HOME_DIR|g" \
    "$TEMPLATE" > "$module_file"

# ── 4. group permissions ──────────────────────────────────────────────────────
say "Opening group read permissions (g+rX)"
chmod -R g+rX "$ROOT/.venv" "$MODULES_ROOT" 2>/dev/null || true
[ -d "$HF_HOME_DIR" ] && chmod -R g+rX "$HF_HOME_DIR" 2>/dev/null || true

# ── 5. self-check ─────────────────────────────────────────────────────────────
say "Self-check"
(cd "$ROOT" && uv run python -c "import h5py, tokeye; print('imports ok')")
if [ ! -d "$FOUNDATION_DIR" ]; then
    echo "warning: foundation archive not found at $FOUNDATION_DIR" >&2
    echo "         (fine off-cluster; on stellar check /scratch/gpfs mounts)" >&2
fi
if command -v module >/dev/null 2>&1; then
    module use --append "$MODULES_ROOT" 2>/dev/null || true
    module show tokeye >/dev/null 2>&1 \
        && echo "module show tokeye: ok" \
        || echo "note: 'module show tokeye' failed in this shell (harmless here)"
fi

# ── 6. user instructions ──────────────────────────────────────────────────────
BASHRC_LINE="module use --append $MODULES_ROOT"
if [ "$DO_BASHRC" = 1 ]; then
    rc="$HOME/.bashrc"
    if [ -f "$rc" ] && grep -qxF "$BASHRC_LINE" "$rc"; then
        say "~/.bashrc already has the module-use line"
    else
        say "Appending the module-use line to ~/.bashrc"
        printf '\n# TokEye group module (EKOLEMEN is not auto-discovered: the\n# group-space probe looks for /projects/KOLEMEN)\n%s\n' \
            "$BASHRC_LINE" >> "$rc"
    fi
fi

cat <<EOF

────────────────────────────────────────────────────────────────────────────
TokEye is installed. Every kolemen member can now use it:

  1. One-time (or pass --bashrc to this script): add to ~/.bashrc
         $BASHRC_LINE
     Why: stellar auto-discovers group modules by probing /projects/<GROUP>
     with the group name uppercased — 'kolemen' -> /projects/KOLEMEN — but
     this group's space is /projects/EKOLEMEN, so discovery never fires.

  2. Then, on stellar-vis1 or stellar-vis2 (GPU + X11):
         module load tokeye
         tokeye              # native GUI (needs ssh -X)
         tokeye app          # web app -> ssh -N -L 7860:localhost:7860 ...
         tokeye princeton-batch --shots 190000-190010 \\
             --outdir /scratch/gpfs/\$USER/tokeye/run1

  Shots come from $FOUNDATION_DIR
  Weights cache:  $HF_HOME_DIR (HF_HOME, set by the module)
  Modulefile:     $module_file
────────────────────────────────────────────────────────────────────────────
EOF
