#!/bin/bash

BASE_DIR="/home/robocar/Robocar"
LOGFILE="$BASE_DIR/logs/run_gamepad_if_idle.log"
SERVICE_NAME="run_gamepad.service"

YELLOW='\033[0;33m'
RESET='\033[0m'

mkdir -p "$(dirname "$LOGFILE")"

{
    echo -e "$YELLOW$(date) Checking SSH session status...$RESET"

    SSH_USERS=$(who | grep -E "pts|ssh" | wc -l)

    if [ "$SSH_USERS" -eq 0 ]; then
        if ! systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "[INFO] No SSH sessions detected. Starting $SERVICE_NAME service..."
            sudo systemctl start "$SERVICE_NAME"
        else
            echo "[INFO] No SSH users detected, but $SERVICE_NAME is already running."
        fi
    else
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            echo "[INFO] SSH users detected. Stopping $SERVICE_NAME service..."
            sudo systemctl stop "$SERVICE_NAME"
        else
            echo "[INFO] SSH users detected, but $SERVICE_NAME is not running."
        fi
    fi

} >> "$LOGFILE" 2>&1

exit 0
