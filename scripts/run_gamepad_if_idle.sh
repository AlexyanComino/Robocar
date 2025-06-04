#!/bin/bash

BASE_DIR="/home/robocar/Robocar"
LOGFILE="$BASE_DIR/logs/run_gamepad_if_idle.log"
SCRIPT_NAME="$BASE_DIR/pyvesc_input.py"
VENV_NAME="$BASE_DIR/.venv"

mkdir -p "$(dirname "$LOGFILE")"

{
    echo "$(date) [INFO] Starting run_gamepad_if_idle.sh script..."

    if [ ! -f "$SCRIPT_NAME" ]; then
        echo "$(date) [ERROR] Script $SCRIPT_NAME not found!"
        exit 1
    fi

    echo "$(date) [INFO] Checking for Python virtual environment..."

    if [ ! -d "$VENV_NAME" ]; then
        echo "$(date) [ERROR] Virtual environment $VENV_NAME not found! Please run setup_venv.sh first."
        exit 1
    fi

    echo "$(date) [INFO] Activating virtual environment $VENV_NAME..."
    source "$VENV_NAME/bin/activate"

    SSH_USERS=$(who | grep -E "pts|ssh" | wc -l)

    RUNNING_PID=$(pgrep -f "$SCRIPT_NAME")

    echo "$(date) [DEBUG] Running PIDs: $(pgrep -f "pyvesc_input.py" | tr '\n' ' ')" >> "$LOGFILE"

    if [ "$SSH_USERS" -eq 0 ]; then
        if [ -z "$RUNNING_PID" ]; then
            echo "$(date) [INFO] No SSH users detected and script is not running. Starting $SCRIPT_NAME..."
            nohup python "$SCRIPT_NAME" >> "$LOGFILE" 2>&1 &
            echo "$(date) [INFO] Script started with PID $!"
        else
            echo "$(date) [INFO] No SSH users detected, but script is already running with PID $RUNNING_PID."
        fi
    else
        if [ -n "$RUNNING_PID" ]; then
            echo "$(date) [INFO] SSH users detected, stopping script with PID $RUNNING_PID."
            kill "$RUNNING_PID"
        else
            echo "$(date) [INFO] SSH users detected, but script is not running."
        fi
    fi
} >> "$LOGFILE" 2>&1
