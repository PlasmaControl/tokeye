#!/usr/bin/env bash
# Launch the TokEye web app and print the exact SSH tunnel one-liner.
#
# Run this on somega (e.g. omega14) after `module load tokeye`. The app binds
# 127.0.0.1 (localhost only) — a per-user SSH tunnel is the security model, so
# do not add --share on GA (it egresses lab data off-network).
set -euo pipefail

port="${TOKEYE_PORT:-7860}"
host="$(hostname -f 2>/dev/null || hostname)"

cat <<EOF
────────────────────────────────────────────────────────────────────────────
TokEye web app → http://localhost:${port}   (binding 127.0.0.1:${port} on ${host})

From your laptop, open the tunnel, then browse to the URL above:

    ssh -N -L ${port}:localhost:${port} ${USER}@${host}

If that node isn't reachable directly, hop through the gateway:

    ssh -N -L ${port}:localhost:${port} -J ${USER}@omega.gat.com ${USER}@${host%%.*}
────────────────────────────────────────────────────────────────────────────
EOF

exec tokeye app --port "${port}" "$@"
