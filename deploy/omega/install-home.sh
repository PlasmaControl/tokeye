#!/usr/bin/env bash
# install-home.sh — install TokEye into a DURABLE home, publish to the shared
# (swept) deployment root, and self-heal the env. Idempotent; safe to re-run.
#
# WHY: today the whole working deployment lives on /cscratch/share/tokeye, which
# is swept (files >32 days old are deleted) — env, modulefile, launchers all
# vanish. This gives TokEye two deliberately-split roots:
#   * durable root (dest, default $HOME/tokeye): a full git checkout of the diiid
#     branch — repo, scripts, modulefile all survive the sweep.
#   * deployment root ($TOKEYE_DIR, default /cscratch/share/tokeye): the multi-GB
#     conda env + shot cache, plus PUBLISHED copies of the modulefile / launchers
#     / shim so teammates can `module use` a shared path. All rebuildable, so the
#     sweep is survivable (re-run this script, or let the shim self-heal).
#
# Usage: install-home.sh [dest] [--yes]
#   dest    durable checkout location (default $HOME/tokeye)
#   --yes   forwarded to ensure-env.sh (rebuild the env without prompting)
#
# Relocate to /fusion later with NO edits (env root + durable home together):
#   TOKEYE_DIR=/fusion/projects/codes/tokeye install-home.sh /fusion/projects/codes/tokeye
set -euo pipefail

dest=""
yes=0
for arg in "$@"; do
  case "$arg" in
    --yes) yes=1 ;;
    -*) echo "install-home.sh: unknown option: $arg" >&2
        echo "  usage: install-home.sh [dest] [--yes]" >&2
        exit 2 ;;
    *) if [[ -z "$dest" ]]; then dest="$arg"
       else echo "install-home.sh: unexpected extra argument: $arg" >&2; exit 2; fi ;;
  esac
done
dest="${dest:-$HOME/tokeye}"

TOKEYE_DIR="${TOKEYE_DIR:-/cscratch/share/tokeye}"
BRANCH="diiid"
DEFAULT_URL="https://github.com/PlasmaControl/tokeye.git"

# The checkout that owns THIS script — the offline clone source and the remote
# URL source (its `origin`).
self="$(readlink -f "${BASH_SOURCE[0]}")"
src_repo="$(cd "$(dirname "$self")" && git rev-parse --show-toplevel 2>/dev/null || true)"
url="$DEFAULT_URL"
if [[ -n "$src_repo" ]]; then
  url="$(git -C "$src_repo" remote get-url origin 2>/dev/null || echo "$DEFAULT_URL")"
fi

# --- 1. Obtain / refresh the durable checkout at dest ------------------------
if [[ -d "$dest/.git" ]]; then
  echo "install-home.sh: refreshing existing checkout at $dest"
  git -C "$dest" pull --ff-only
  cur="$(git -C "$dest" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  if [[ "$cur" != "$BRANCH" ]]; then
    echo "install-home.sh: WARNING: $dest is on '$cur', not '$BRANCH'." >&2
  fi
elif git ls-remote --heads "$url" "$BRANCH" >/dev/null 2>&1; then
  echo "install-home.sh: cloning $url ($BRANCH) → $dest"
  git clone -b "$BRANCH" "$url" "$dest"
elif [[ -n "$src_repo" ]]; then
  # GitHub unreachable (offline compute node). A LOCAL git clone from the source
  # checkout is the same offline story as an rsync but with real clone semantics
  # and no hand-kept exclude list.
  echo "install-home.sh: GitHub unreachable — local clone from $src_repo → $dest"
  git clone -b "$BRANCH" "$src_repo" "$dest"
else
  echo "install-home.sh: cannot obtain the repo — no $dest/.git, GitHub" >&2
  echo "  unreachable, and this script is not inside a checkout." >&2
  exit 1
fi

# --- 2. Publish into the shared (swept) deployment root ----------------------
# All copies come from the DURABLE checkout, so a re-publish always matches the
# installed repo. Everything here is rebuildable → the sweep is survivable.
srcdir="$dest/deploy/omega"
mkdir -p "$TOKEYE_DIR/modulefiles" "$TOKEYE_DIR/bin"

cp "$srcdir/modulefiles/tokeye.lua" "$TOKEYE_DIR/modulefiles/"

cp "$srcdir/tokeye-connect.sh" "$TOKEYE_DIR/"
chmod 755 "$TOKEYE_DIR/tokeye-connect.sh"

cp "$srcdir/about.txt" "$TOKEYE_DIR/"

cp "$srcdir/bin/tokeye" "$TOKEYE_DIR/bin/"
cp "$srcdir/ensure-env.sh" "$TOKEYE_DIR/bin/"
chmod 755 "$TOKEYE_DIR/bin/tokeye" "$TOKEYE_DIR/bin/ensure-env.sh"

# Sidecar: how the PUBLISHED ensure-env.sh (now living outside the repo) finds
# the durable checkout when it needs to rebuild the env.
printf '%s\n' "$dest" > "$TOKEYE_DIR/bin/.tokeye-repo"

# --- 3. Build / self-heal the env (from the durable checkout) ----------------
if [[ "$yes" -eq 1 ]]; then
  "$srcdir/ensure-env.sh" --yes
else
  "$srcdir/ensure-env.sh"
fi

# --- 4. Summary --------------------------------------------------------------
cat <<EOF

TokEye installed at $dest (env root: $TOKEYE_DIR).

Load the module — DURABLE path (survives the /cscratch sweep):
    module use $dest/deploy/omega/modulefiles
    module load tokeye

Or the SHARED published path (on $TOKEYE_DIR; re-run this script to re-publish
if it gets swept):
    module use $TOKEYE_DIR/modulefiles
    module load tokeye
EOF
