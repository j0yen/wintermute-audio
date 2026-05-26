#!/usr/bin/env bash
# Materialize sha256sums in PKGBUILD from the actual upstream files.
# Requires `updpkgsums` from the pacman-contrib package.
#
# Usage:
#   cd contrib/wm-models && ./update-hashes.sh

set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

if ! command -v updpkgsums >/dev/null 2>&1; then
  echo "error: updpkgsums not found; install pacman-contrib" >&2
  exit 1
fi

if [[ ! -f PKGBUILD ]]; then
  echo "error: PKGBUILD not found in $here" >&2
  exit 1
fi

# Backup, then update.
cp PKGBUILD PKGBUILD.bak.$(date -u +%Y%m%dT%H%M%SZ)
updpkgsums
echo "sha256sums refreshed in $here/PKGBUILD"
