#!/usr/bin/env bash
# Run the plugin's tests inside the FiestaBoard dev container.
# The host has no pytest, and FiestaBoard's CLAUDE.md forbids local installs.
set -euo pipefail
cd "$(dirname "$0")/.."
docker exec fiestaboard-dev rm -rf /tmp/gd
docker cp . fiestaboard-dev:/tmp/gd >/dev/null
docker exec -e PYTHONPATH=/app -w /tmp/gd fiestaboard-dev \
    python -m pytest "${@:-tests/}" -q
