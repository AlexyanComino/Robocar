#!/bin/bash

# Usage: ./build_engine.sh model.onnx

ONNX_PATH="$1"

# Check ONNX file
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "❌ ONNX file '$ONNX_PATH' does not exist."
    exit 1
fi

ENGINE_PATH="${ONNX_PATH%.onnx}_fp16.engine"

CMD=(
    trtexec
    --onnx="$ONNX_PATH"
    --saveEngine="$ENGINE_PATH"
    --minShapes=input:1x3x256x320
    --optShapes=input:1x3x256x320
    --maxShapes=input:1x3x256x320
    --explicitBatch
    --buildOnly
    --workspace=1024
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
