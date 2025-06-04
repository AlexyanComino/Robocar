#!/bin/bash

BASE_DIR="/home/robocar/Robocar"

echo -e "$[INFO] Pulling latest code from Git..."
git pull origin main

$BASE_DIR/scripts/setup_venv.sh
