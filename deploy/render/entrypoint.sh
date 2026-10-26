#!/bin/sh
# Bind to the platform PORT when present (Render, Railway, Fly, etc.).
set -eu
export GUARDIANEYE_HOST="${GUARDIANEYE_HOST:-0.0.0.0}"
export GUARDIANEYE_PORT="${PORT:-${GUARDIANEYE_PORT:-8742}}"
# Faster dashboard on small Render instances: skip mel preview without weights,
# and do not phone home to Gradio analytics.
export GUARDIANEYE_FAST_UI="${GUARDIANEYE_FAST_UI:-1}"
export GUARDIANEYE_DEMO_LABELS="${GUARDIANEYE_DEMO_LABELS:-1}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"
exec python -m guardianeye.serve.api
