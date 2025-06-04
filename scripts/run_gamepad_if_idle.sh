#!/bin/bash

BASE_DIR="/home/robocar/Robocar"
LOGFILE="$BASE_DIR/logs/run_gamepad_if_idle.log"
SCRIPT_NAME="$BASE_DIR/pyvesc_input.py"
VENV_NAME="$BASE_DIR/.venv"
PIDFILE="$BASE_DIR/pyvesc_input.pid"

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

    SSH_USERS=$(who | grep -E "pts|ssh" | wc -l)
    SSH_USERS="0"

    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            if ps -p "$PID" -o args= | grep -q "$SCRIPT_NAME"; then
                RUNNING_PID="$PID"
            else
                echo "$(date) [WARN] PID $PID does not match expected script. Cleaning up stale PID file."
                rm -f "$PIDFILE"
                RUNNING_PID=""
            fi
        else
            RUNNING_PID=""
            rm -f "$PIDFILE"
            echo "$(date) [WARN] PID $PID is not running. Cleaning up stale PID file."
        fi
    else
        RUNNING_PID=""
    fi

    echo "$(date) [DEBUG] Current running PID: $RUNNING_PID"

    if [ "$SSH_USERS" -eq 0 ]; then
        if [ -z "$RUNNING_PID" ]; then
            echo "$(date) [INFO] Activating virtual environment $VENV_NAME..."
            echo "$(date) [INFO] No SSH users detected and script is not running. Starting $SCRIPT_NAME..."
            source "$VENV_NAME/bin/activate"

            nohup python "$SCRIPT_NAME" >> "$BASE_DIR/python.log" 2>&1 &
            echo $! > "$PIDFILE"
            echo "$(date) [INFO] Script started with PID $!"
        else
            echo "$(date) [INFO] No SSH users detected, but script is already running with PID $RUNNING_PID."
        fi
    else
        if [ -n "$RUNNING_PID" ]; then
            echo "$(date) [INFO] SSH users detected, stopping script with PID $RUNNING_PID."
            kill "$RUNNING_PID"
            rm -f "$PIDFILE"
        else
            echo "$(date) [INFO] SSH users detected, but script is not running."
        fi
    fi
} >> "$LOGFILE" 2>&1
