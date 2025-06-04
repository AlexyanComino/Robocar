#!/bin/bash

GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

BASE_DIR="/home/robocar/Robocar"
VENV_NAME="$BASE_DIR/.venv"

echo -e "${YELLOW}Setting up Python virtual environment...${RESET}"

if [ ! -d "$VENV_NAME" ]; then
    python3 -m venv "$VENV_NAME"
    echo -e "${GREEN}[OK] Virtual environment created: $VENV_NAME${RESET}"
else
    echo -e "${CYAN}[INFO] Virtual environment already exists: $VENV_NAME${RESET}"
fi

echo -e "${CYAN}[INFO] Activating virtual environment...${RESET}"
source "$VENV_NAME/bin/activate"

if [ -f "requirements.txt" ]; then
    echo -e "${CYAN}[INFO] Installing dependencies...${RESET}"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    echo -e "${GREEN}[OK] Dependencies installed successfully!${RESET}"
else
    echo -e "${RED}[WARNING] requirements.txt not found. Skipping dependency installation.${RESET}"
fi

echo -e "${GREEN}[DONE] Setup completed successfully!${RESET}"
