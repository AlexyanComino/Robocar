#!/bin/bash

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo -e "$[INFO] Pulling latest code from Git..."
git pull origin main

./setup_venv.sh
