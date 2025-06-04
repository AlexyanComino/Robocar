#!/bin/bash

BASE_DIR="/home/robocar/Robocar"
LOGFILE="$BASE_DIR/logs/init.log"
mkdir -p "$(dirname "$LOGFILE")"

{
    echo -e "$[INFO] Pulling latest code from Git..."
    git pull origin main

    echo "[INFO] Running venv setup..."
    $BASE_DIR/scripts/setup_venv.sh
} > "$LOGFILE" 2>&1
