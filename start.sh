#!/bin/bash

set -e

git pull

SCRIPT_PATH="main.py"

echo "--- Executing $SCRIPT_PATH with arguments: $@ ---"

python3 "$SCRIPT_PATH" "$@"
