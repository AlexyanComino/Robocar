#!/bin/bash

# Usage: ./onnx_to_engine.sh model.onnx WIDTH HEIGHT

set -e

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <model.onnx> <WIDTH> <HEIGHT>"
    exit 1
fi

ONNX_PATH="$1"
WIDTH="$2"
HEIGHT="$3"

# Check ONNX file
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "❌ ONNX file '$ONNX_PATH' does not exist."
    exit 1
fi

# Check if WIDTH and HEIGHT are integers
if ! [[ "$WIDTH" =~ ^[0-9]+$ ]] || ! [[ "$HEIGHT" =~ ^[0-9]+$ ]]; then
    echo "❌ WIDTH and HEIGHT must be positive integers."
    exit 1
fi

ENGINE_PATH="${ONNX_PATH%.onnx}_fp16_${HEIGHT}x${WIDTH}.engine"

# Check if the engine already exists
if [[ ! -f "$ENGINE_PATH"]]; then
    read -p "⚠️ Engine '$ENGINE_PATH' already exists. Overwrite? (y/N): " confirm
    case "$confirm" in
        [yY][eE][sS]|[yY])
            echo "🔁 Overwriting existing engine..."
            ;;
        *)
            echo "❌ Aborted."
            exit 1
            ;;
    esac
fi

CMD=(
    trtexec
    --onnx="$ONNX_PATH"
    --saveEngine="$ENGINE_PATH"
    --minShapes=input:1x3x${HEIGHT}x${WIDTH}
    --optShapes=input:1x3x${HEIGHT}x${WIDTH}
    --maxShapes=input:1x3x${HEIGHT}x${WIDTH}
    --explicitBatch
    --buildOnly
    --workspace=1024
    --fp16
    --verbose
)

echo "🔧 Building engine..."
"${CMD[@]}"

# Check result
if [[ $? -eq 0 ]]; then
    echo "✅ Engine saved to $ENGINE_PATH"
else
    echo "❌ Failed to build engine."
    exit 1
fi
