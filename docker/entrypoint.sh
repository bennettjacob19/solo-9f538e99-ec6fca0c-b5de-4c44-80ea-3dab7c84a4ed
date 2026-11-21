#!/bin/sh
# Start the demo writers (same PID namespace, same UID) and then the web app.
set -eu

LOG_ROOT="${LOG_ROOT:-/data/logs}"
STATE_DIR="${STATE_DIR:-/data/state}"
mkdir -p "$LOG_ROOT" "$STATE_DIR"

if [ "${DEMO_WRITERS:-1}" != "0" ]; then
  # Also seed one generation of deleted logs so the page has work to do.
  python /app/demo/seed.py \
    --log-root "$LOG_ROOT" \
    --state-dir "$STATE_DIR" \
    --count "${DEMO_WRITERS_COUNT:-8}" \
    --chunk "${WRITE_CHUNK:-8192}" \
    --interval "${WRITE_INTERVAL:-0.25}" \
    --max-mb "${WRITE_MAX_MB:-64}" \
    --fill-seconds 1.5 || true
fi

# One worker: plans are held in process memory and must be validated by the
# same process that issued them.  Threads keep the UI responsive.
exec gunicorn \
  --bind "0.0.0.0:${PORT:-8080}" \
  --workers 1 \
  --threads 8 \
  --timeout 60 \
  --access-logfile - \
  --error-logfile - \
  "logreclaim.wsgi:app"
