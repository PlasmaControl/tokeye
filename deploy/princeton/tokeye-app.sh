#!/usr/bin/env bash
# Launch the TokEye web app and print the exact SSH tunnel one-liner.
#
# Run this on stellar-vis1/stellar-vis2 after `module load tokeye`. The app
# binds 127.0.0.1 (localhost only) — a per-user SSH tunnel is the access
# model; do not use --share (it egresses data through gradio's relay).
set -euo pipefail

port="${TOKEYE_PORT:-7860}"
host="$(hostname -f 2>/dev/null || hostname)"

cat <<EOF
────────────────────────────────────────────────────────────────────────────
TokEye web app → http://localhost:${port}   (binding 127.0.0.1:${port} on ${host})

From your laptop, open the tunnel, then browse to the URL above:

    ssh -N -L ${port}:localhost:${port} ${USER}@${host}

(stellar-vis1/stellar-vis2 are directly reachable from the campus
network/VPN — no jump host needed.)
────────────────────────────────────────────────────────────────────────────
EOF

exec tokeye app --port "${port}" "$@"
