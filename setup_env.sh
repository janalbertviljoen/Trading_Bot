#!/usr/bin/env bash
# Create the project virtual environment and install every dependency group
# (pyproject.toml) at the pinned versions (package_versions.txt).
#
# Usage, from the repository root:
#   ./setup_env.sh              # creates .venv with python3
#   PYTHON=python3.13 ./setup_env.sh
#   VENV_DIR=/path/to/venv ./setup_env.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYTHON" -m venv "$VENV_DIR"
fi

# pip >= 25.1 is needed for --group.
"$VENV_DIR/bin/python" -m pip install --upgrade "pip>=25.1"
(cd "$REPO_ROOT" && "$VENV_DIR/bin/python" -m pip install \
    -c package_versions.txt --group dev)

"$VENV_DIR/bin/python" -m pip check
echo "Environment ready: $VENV_DIR ($("$VENV_DIR/bin/python" --version))"
