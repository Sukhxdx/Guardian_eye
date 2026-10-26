#!/bin/sh
# Bind to the platform PORT when present (Render, Railway, Fly, etc.).
set -eu
export GUARDIANEYE_HOST="${GUARDIANEYE_HOST:-0.0.0.0}"
export GUARDIANEYE_PORT="${PORT:-${GUARDIANEYE_PORT:-8742}}"
exec python -m guardianeye.serve.api
