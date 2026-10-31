#!/usr/bin/env bash
# Convenience wrapper so you can run ./setup_amd.sh from the repo root.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$here/scripts/setup_amd.sh" "$@"
