#!/bin/sh
set -eu

umask 027

# Database readiness is part of starting the application server, not operator
# utilities. In particular, emergency Modal cleanup must still run while
# PostgreSQL is unavailable.
if [ "${1:-}" = "uvicorn" ]; then
  if [ -n "${DATABASE_URL:-}" ]; then
    echo "ctrl-pi: applying PostgreSQL migrations"
    alembic -c /app/alembic.ini upgrade head
  else
    echo "ctrl-pi: DATABASE_URL is not configured; starting in first-run mode" >&2
  fi
fi

exec "$@"
