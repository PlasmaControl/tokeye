#!/usr/bin/env bash
# ensure-env.sh — self-healing env guard for the Omega (DIII-D) deployment.
#
# Project rule (MEMORY / CLAUDE.md): a broken or missing env is REBUILT, not
# diagnosed. /cscratch/share/tokeye is swept (files >32 days old are deleted),
# so the multi-GB conda env can disappear; this script detects that and rebuilds
# it in place. It is both the supported "build the env" path AND the self-heal
# the launcher shim (bin/tokeye) hands off to when the env binary is gone.
#
# Usage: ensure-env.sh [--yes]
#   --yes   rebuild without prompting (scripted / non-interactive rebuilds)
set -euo pipefail

yes=0
for arg in "$@"; do
  case "$arg" in
    --yes) yes=1 ;;
    *) echo "ensure-env.sh: unknown argument: $arg (usage: ensure-env.sh [--yes])" >&2
       exit 2 ;;
  esac
done

# --- Resolve REPO (the checkout that owns deploy/omega + pyproject.toml) -----
# Order: $TOKEYE_REPO → .tokeye-repo sidecar next to us → walk up from our own
# resolved location. This lets the script run from any checkout (/home,
# /cscratch, /fusion) or as the published copy in $TOKEYE_DIR/bin.
self="$(readlink -f "${BASH_SOURCE[0]}")"
selfdir="$(dirname "$self")"

_has_pyproject() { [[ -f "$1/pyproject.toml" ]]; }

REPO=""
if [[ -n "${TOKEYE_REPO:-}" ]]; then
  if _has_pyproject "$TOKEYE_REPO"; then
    REPO="$TOKEYE_REPO"
  else
    echo "ensure-env.sh: TOKEYE_REPO=$TOKEYE_REPO has no pyproject.toml." >&2
    exit 1
  fi
elif [[ -f "$selfdir/.tokeye-repo" ]]; then
  # Sidecar written by install-home.sh when it publishes this script outside the
  # repo (into $TOKEYE_DIR/bin). One line: the durable checkout path.
  REPO="$(head -n1 "$selfdir/.tokeye-repo")"
  if ! _has_pyproject "$REPO"; then
    echo "ensure-env.sh: .tokeye-repo points at '$REPO' (no pyproject.toml)." >&2
    exit 1
  fi
else
  # Walk up from our own resolved location until we find pyproject.toml.
  d="$selfdir"
  while [[ "$d" != "/" ]]; do
    if _has_pyproject "$d"; then REPO="$d"; break; fi
    d="$(dirname "$d")"
  done
fi

if [[ -z "$REPO" ]]; then
  echo "ensure-env.sh: cannot locate the TokEye repo." >&2
  echo "  Remedy: set TOKEYE_REPO=<checkout>, or run this script from inside a checkout." >&2
  exit 1
fi

# --- Env path (arch mapping identical to tokeye.lua / the shim) --------------
TOKEYE_DIR="${TOKEYE_DIR:-/cscratch/share/tokeye}"
case "$(uname -m)" in
  aarch64) arch_env="env-aarch64" ;;
  *)       arch_env="env-x86_64" ;;
esac
ENV="$TOKEYE_DIR/$arch_env"

# --- Sanitize our own environment (mirror tokeye.lua) ------------------------
# When this runs from a bare Omega login shell (the documented first-install
# flow via install-home.sh, and the shim's self-heal), the inherited env carries
# two poisons the modulefile deliberately strips — but the modulefile isn't
# loaded here yet. Reproduce exactly what tokeye.lua does, or a HEALTHY env
# false-negatives the health check below and we rebuild in a loop:
#   1. PYTHONPATH points at the cluster's py3.7 MDSplus
#      (/fusion/usc/.../opt/mdsplus/...), which SHADOWS this env's own
#      conda-forge MDSplus and crashes under numpy 2.x — cf. tokeye.lua's
#      `unsetenv("PYTHONPATH")`. Observed live on omega14 importing the py3.7
#      MDSplus instead of the env's copy.
#   2. The login LD_LIBRARY_PATH puts the system /lib64 (old libstdc++, no
#      GLIBCXX_3.4.29) AHEAD of the env's lib, which can break torch/numpy —
#      cf. tokeye.lua's `prepend_path("LD_LIBRARY_PATH", <env>/lib)`.
# Doing it here (not just in the shim) covers the health check, pip, AND the
# post-rebuild re-check in one place. `$ENV/lib` may not exist yet before the
# first build — harmless as a leading LD_LIBRARY_PATH entry.
unset PYTHONPATH
export LD_LIBRARY_PATH="$ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- Health check ------------------------------------------------------------
# The model + both UIs must import cleanly (native GUI: PySide6/pyqtgraph; web
# app: gradio/plotly; data: MDSplus). This is the standardization step's canary.
HEALTH_IMPORTS='import tokeye, MDSplus, PySide6, pyqtgraph, gradio, plotly'

_check_health() {
  "$ENV/bin/python" -c "$HEALTH_IMPORTS" >/dev/null 2>&1
}

if _check_health; then
  # Cheap idempotent repair of a known gotcha: `mamba env create` recreates
  # env/bin from scratch, so the manually-copied tokeye-app launcher can go
  # missing even when the env itself is healthy. Re-copy it if absent.
  if [[ ! -e "$ENV/bin/tokeye-app" ]]; then
    cp "$REPO/deploy/omega/tokeye-app.sh" "$ENV/bin/tokeye-app"
    chmod +x "$ENV/bin/tokeye-app"
    echo "ensure-env.sh: restored missing launcher $ENV/bin/tokeye-app"
  fi
  echo "ensure-env.sh: env OK at $ENV"
  exit 0
fi

# --- Unhealthy → rebuild flow ------------------------------------------------
echo "ensure-env.sh: env unhealthy at $ENV" >&2
echo "  (\"$ENV/bin/python -c '$HEALTH_IMPORTS'\" failed / python missing)" >&2

# The exact rebuild command, kept in one place so the prompt, the non-tty
# refusal message, and the code below can't drift apart.
REBUILD_CMD="rm -rf \"$ENV\" && mamba env create -p \"$ENV\" -f \"$REPO/deploy/omega/environment-omega.yml\" && \"$ENV/bin/pip\" install -e \"$REPO[app]\""

if [[ "$yes" -ne 1 ]]; then
  if [[ -t 0 ]]; then
    printf 'Rebuild env at %s? [y/N] ' "$ENV" >&2
    read -r reply
    case "$reply" in
      y | Y | yes | YES) ;;
      *) echo "ensure-env.sh: aborted — no rebuild." >&2; exit 1 ;;
    esac
  else
    # Non-interactive and no --yes: never hang a caller. Print the exact command
    # and bail so a human / script runs the rebuild deliberately.
    echo "ensure-env.sh: not a tty and --yes not given — refusing to rebuild." >&2
    echo "  Re-run with --yes, or run the rebuild yourself:" >&2
    echo "    $REBUILD_CMD" >&2
    exit 1
  fi
fi

# --- Rebuild (consent given via prompt or --yes) -----------------------------
# Locate mamba FIRST (before deleting anything): PATH → `module load conda`
# (Lmod exports `module` as a shell function to subshells) → absolute
# mambaforge fallback. Error out if none work.
mamba_bin="$(command -v mamba || true)"
if [[ -z "$mamba_bin" ]] && declare -F module >/dev/null 2>&1; then
  module load conda >/dev/null 2>&1 || true
  mamba_bin="$(command -v mamba || true)"
fi
if [[ -z "$mamba_bin" ]]; then
  fallback="/fusion/projects/codes/conda/mambaforge/bin/mamba"
  [[ -x "$fallback" ]] && mamba_bin="$fallback"
fi
if [[ -z "$mamba_bin" ]]; then
  echo "ensure-env.sh: cannot find mamba (tried PATH, 'module load conda', and" >&2
  echo "  /fusion/projects/codes/conda/mambaforge/bin/mamba)." >&2
  exit 1
fi

echo "ensure-env.sh: rebuilding env at $ENV using $mamba_bin …" >&2
# A half-swept env makes `mamba env create` fail; the user just confirmed a
# rebuild, so clear it. Scoped strictly to $ENV — nothing else is ever removed.
rm -rf "$ENV"
# NO -y here: `env create` never prompts (non-interactive by design), and old
# conda/mamba — e.g. the 2022-mambaforge absolute fallback above — rejects the
# flag outright (`mamba: error: unrecognized arguments: -y`, seen live on
# omega14). Consent is handled by our own prompt/--yes gating above.
"$mamba_bin" env create -p "$ENV" -f "$REPO/deploy/omega/environment-omega.yml"

# TokEye (editable) + its wheels (torch/torchvision/gradio/plotly). No manual
# gradio pin: since ab3245f the [app] extra pins gradio>=5.49,<6 itself, so pip
# resolves the tested gradio without a second, drift-prone argument here.
"$ENV/bin/pip" install -e "$REPO[app]"

# Re-copy the tunnel-printing launcher (mamba env create wiped env/bin).
cp "$REPO/deploy/omega/tokeye-app.sh" "$ENV/bin/tokeye-app"
chmod +x "$ENV/bin/tokeye-app"

# Re-run the health check; report and exit with its status.
if _check_health; then
  echo "ensure-env.sh: rebuild OK at $ENV"
  exit 0
fi
echo "ensure-env.sh: rebuild completed but the health check still fails at $ENV" >&2
exit 1
