#!/bin/sh
set -eu

umask 027

if [ -n "${DATABASE_URL:-}" ]; then
  echo "ctrl-pi: applying PostgreSQL migrations"
  alembic -c /app/alembic.ini upgrade head
else
  echo "ctrl-pi: DATABASE_URL is not configured; starting in first-run mode" >&2
fi

exec "$@"
