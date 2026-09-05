#!/usr/bin/env bash
# Run the plugin's tests inside the FiestaBoard dev container.
# The host has no pytest, and FiestaBoard's CLAUDE.md forbids local installs.
set -euo pipefail
cd "$(dirname "$0")/.."
docker exec fiestaboard-dev rm -rf /tmp/gd
docker cp . fiestaboard-dev:/tmp/gd >/dev/null
# Core's `plugins` package extends its path into /app/data/external_plugins,
# so imports resolve THERE, not through /tmp/gd's symlink. Sync it too, or
# the suite silently tests whatever was deployed last.
docker exec fiestaboard-dev sh -c 'rm -rf /app/data/external_plugins/generative_dashboard \
  && cp -r /tmp/gd /app/data/external_plugins/generative_dashboard \
  && rm -rf /app/data/external_plugins/generative_dashboard/plugins' 
docker exec -e PYTHONPATH=/app -w /tmp/gd fiestaboard-dev \
    python -m pytest "${@:-tests/}" -q
