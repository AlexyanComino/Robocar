#!/bin/bash

{
    echo "[INFO] Pulling latest code from Git..."
    git pull origin main

    echo "[INFO] Setting up Python environment..."
    ./setup_venv.sh

    echo "[INFO] Activating venv..."
    source venv/bin/activate
} >> robocar.log 2>&1

python3 pyvesc_input.py
