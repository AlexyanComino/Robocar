#!/bin/bash

set -e

git pull

SCRIPT_PATH="main.py"

python3 "$SCRIPT_PATH" "$@"
