#!/bin/sh
set -e
mkdir -p /app/data/index
chown -R appuser:appuser /app/data/index
exec gosu appuser python /app/scripts/docker_entrypoint.py "$@"
