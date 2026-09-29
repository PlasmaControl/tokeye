#!/usr/bin/env bash
# Reject files over 2 MB: every blob that the commits in a range add, including
# one a later commit deletes (it still lands in the history), and every file in
# the tree at HEAD.
#
# CI (the large-files job in workflows/test.yml) passes the range in the
# environment: BASE is the PR's base sha or the push's `before`, and TIP is the
# PR head or the pushed sha. Locally, pass a rev-list range, e.g.
# `bash .github/scripts/large-files.sh origin/main..`, or nothing to scan the
# whole history of HEAD.
set -euo pipefail

limit=$((2 * 1024 * 1024))
zero=0000000000000000000000000000000000000000
tip=${TIP:-HEAD}

if [ "$#" -gt 0 ]; then
  range=("$@")
elif [ -n "${BASE:-}" ] && [ "$BASE" != "$zero" ] &&
  git cat-file -e "$BASE^{commit}" 2>/dev/null; then
  range=("$BASE..$tip")
else
  # No usable BASE: a new branch or tag (the all-zeros `before`), a `before`
  # that a force-push left unreachable, or a local run. Scan TIP's whole
  # history. `TIP --not --remotes` would miss commits: with fetch-depth 0 the
  # pushed branch is itself a remote-tracking ref, and any other branch that
  # holds a commit would hide it, checked or not.
  range=("$tip")
fi

echo "Range: ${range[*]} ($(git rev-list --count "${range[@]}") commits)"
fail=0

# rev-list names each object once, with the path it was first seen at.
added=$(git rev-list --objects "${range[@]}" |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  awk -v lim="$limit" '$1 == "blob" && $3 > lim {
    path = $0
    sub(/^[^ ]+ [^ ]+ [^ ]+ ?/, "", path)
    printf "  %.1f MB  %s  (blob %s)\n", $3 / 1048576, path, substr($2, 1, 12)
  }')
if [ -n "$added" ]; then
  echo "::error::Commits in this range add files larger than 2 MB. Deleting a file in a later commit leaves it in the history: rewrite the commits that add it, and host large files on Hugging Face or a Release."
  echo "$added"
  fail=1
fi

tree=$(git ls-tree -r -l HEAD |
  awk -F '\t' -v lim="$limit" '{ split($1, f, " ") }
    f[2] == "blob" && f[4] + 0 > lim { printf "  %.1f MB  %s\n", f[4] / 1048576, $2 }')
if [ -n "$tree" ]; then
  echo "::error::Files larger than 2 MB are not allowed (host them on Hugging Face or a Release):"
  echo "$tree"
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "OK: no file over 2 MB in the range or in the tree at HEAD."
fi
exit "$fail"
