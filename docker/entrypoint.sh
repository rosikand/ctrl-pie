#!/bin/sh
set -eu

umask 027

# Database readiness is part of starting the application server, not operator
# utilities. In particular, emergency Modal cleanup must still run while
# PostgreSQL is unavailable.
if [ "${1:-}" = "uvicorn" ]; then
  listen_port="${CTRL_PI_LISTEN_PORT:-8000}"
  case "${listen_port}" in
    ''|*[!0-9]*)
      echo "ctrl-pi: CTRL_PI_LISTEN_PORT must be an integer from 1 through 65535" >&2
      exit 2
      ;;
  esac
  if [ "${listen_port}" -lt 1 ] || [ "${listen_port}" -gt 65535 ]; then
    echo "ctrl-pi: CTRL_PI_LISTEN_PORT must be an integer from 1 through 65535" >&2
    exit 2
  fi
  export CTRL_PI_LISTEN_PORT="${listen_port}"
  export UVICORN_PORT="${listen_port}"

  if [ -n "${DATABASE_URL:-}" ]; then
    echo "ctrl-pi: applying PostgreSQL migrations"
    alembic -c /app/alembic.ini upgrade head
  else
    echo "ctrl-pi: DATABASE_URL is not configured; starting in first-run mode" >&2
  fi
fi

exec "$@"
