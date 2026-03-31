#!/usr/bin/env bash
set -euo pipefail

SPEC="${NEDIS_SPEC:-baked}"

case "$SPEC" in
  baked)
    echo "Using baked-in nedis (from /workspace)"
    ;;

  workspace|.)
    echo "Installing nedis in editable mode from /workspace"
    micromamba run -n nedis python -m pip install --no-deps -e /workspace
    ;;

  *)
    echo "Installing nedis from spec: $SPEC"
    micromamba run -n nedis python -m pip install --no-deps --upgrade "$SPEC"
    ;;
esac

exec micromamba run -n nedis "$@"
