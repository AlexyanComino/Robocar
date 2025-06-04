#!/bin/bash

BASE_DIR="/home/robocar/Robocar"
LOGFILE="$BASE_DIR/logs/run_gamepad_if_idle.log"
SCRIPT_PATH="$BASE_DIR/pyvesc_input.py"
VENV_NAME="$BASE_DIR/.venv"
PYTHON_BIN="$VENV_NAME/bin/python"

mkdir -p "$(dirname "$LOGFILE")"

if [ ! -f "$SCRIPT_NAME" ]; then
    echo "$(date) [ERROR] Script $SCRIPT_NAME not found!" >> "$LOGFILE"
    exit 1
fi

if [ ! -d "$VENV_NAME" ]; then
    echo "$(date) [ERROR] Virtual environment $VENV_NAME not found! Please run setup_venv.sh first." >> "$LOGFILE"
    exit 1
fi

source "$VENV_NAME/bin/activate"

SSH_USERS=$(who | grep -E "pts|ssh" | wc -l)

RUNNING_PID=$(pgrep -f "$PYTHON_BIN $SCRIPT_NAME")

if [ "$SSH_USERS" -eq 0 ]; then
    if [ -z "$RUNNING_PID" ]; then
        echo "$(date) [INFO] No SSH users detected and script is not running. Starting $SCRIPT_NAME..." >> "$LOGFILE"
        nohup "$PYTHON_BIN" "$SCRIPT_NAME" >> "$LOGFILE" 2>&1 &
    else
        echo "$(date) [INFO] No SSH users detected, but script is already running with PID $RUNNING_PID." >> "$LOGFILE"
    fi
else
    if [ -n "$RUNNING_PID" ]; then
        echo "$(date) [INFO] SSH users detected, stopping script with PID $RUNNING_PID." >> "$LOGFILE"
        kill "$RUNNING_PID"
    else
        echo "$(date) [INFO] SSH users detected, but script is not running." >> "$LOGFILE"
    fi
fi
