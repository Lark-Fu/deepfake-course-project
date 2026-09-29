#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /data/conda_envs/fq/data/bin/python app/app.py "$@"
