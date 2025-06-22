#!/bin/bash

# Usage: ./onnx_to_engine.sh model.onnx

ONNX_PATH="$1"

# Check ONNX file
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "❌ ONNX file '$ONNX_PATH' does not exist."
    exit 1
fi

HEIGHT=256
WIDTH=783

ENGINE_PATH="${ONNX_PATH%.onnx}_fp16_${HEIGHT}x${WIDTH}.engine"

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
