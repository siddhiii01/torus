#!/bin/sh
# Use the active config bundle when one is installed; otherwise the config baked
# into the image. Switching bundles = activate/rollback + restart; no rebuild.
set -eu
ROOT="${TORUS_BUNDLE_ROOT:-/var/lib/torus/bundles}"
if [ -f "$ROOT/state.json" ]; then
    TORUS_CONFIG_DIR="$(python -m torus.service bundle path --root "$ROOT")"
    export TORUS_CONFIG_DIR
fi
exec python -m torus.service "$@"
