#!/usr/bin/env bash
# tokeye-connect — run this on your LAPTOP (not the cluster). One command:
# it opens the SSH tunnel, launches `tokeye app` on the cluster over that same
# connection, and opens your local browser once the app is up. Ctrl-C tears
# down both the app and the tunnel.
#
#   ./tokeye-connect.sh                      # defaults to chenn@somega.gat.com:7860
#   ./tokeye-connect.sh you@somega.gat.com   # different user/host
#   TOKEYE_PORT=7870 ./tokeye-connect.sh     # different port
#
# Why laptop-side: the cluster app binds 127.0.0.1 and cannot reach your local
# browser, so the tunnel + browser-open must be orchestrated from here. The
# LocalForward rides the *same* SSH session that runs the app, so it always
# points at the node the app actually landed on (even via the somega round-robin).
set -euo pipefail

HOST="${1:-chenn@somega.gat.com}"
PORT="${TOKEYE_PORT:-7860}"
URL="http://localhost:${PORT}"

# Open the browser once the forwarded port answers (runs in the background).
(
  for _ in $(seq 1 90); do
    if curl -fsS "${URL}/" >/dev/null 2>&1; then
      echo "tokeye-connect: app is up → opening ${URL}"
      open "${URL}" 2>/dev/null || xdg-open "${URL}" 2>/dev/null || \
        echo "tokeye-connect: open ${URL} in your browser"
      exit 0
    fi
    sleep 1
  done
  echo "tokeye-connect: timed out waiting for ${URL} (is the app failing to start?)"
) &

echo "tokeye-connect: tunneling ${PORT} and launching the app on ${HOST} …"
echo "tokeye-connect: (Ctrl-C here stops the app and closes the tunnel)"
exec ssh -t -L "${PORT}:localhost:${PORT}" "${HOST}" \
  "bash -lc 'module use /cscratch/share/tokeye/modulefiles && module load tokeye && tokeye app --port ${PORT}'"
