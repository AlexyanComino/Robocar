#!/bin/bash

# Usage: ./build_engine.sh model.onnx [fp16|int8]

ONNX_PATH="$1"
PRECISION="$2"

# Check ONNX file
if [[ ! -f "$ONNX_PATH" ]]; then
    echo "❌ ONNX file '$ONNX_PATH' does not exist."
    exit 1
fi

# Check precision
if [[ "$PRECISION" != "fp16" && "$PRECISION" != "int8" ]]; then
    echo "❌ Invalid precision: '$PRECISION'"
    echo "Usage: ./build_engine.sh model.onnx [fp16|int8]"
    exit 1
fi

# Derive output path
ENGINE_PATH="${ONNX_PATH%.onnx}_${PRECISION}.engine"

# Build command
CMD=(
    trtexec
    --onnx="$ONNX_PATH"
    --saveEngine="$ENGINE_PATH"
    --explicitBatch
    --buildOnly
)

# Add precision flag
if [[ "$PRECISION" == "fp16" ]]; then
    CMD+=(--fp16)
elif [[ "$PRECISION" == "int8" ]]; then
    CMD+=(--int8)
fi

# Run
echo "🔧 Building engine with precision: $PRECISION"
"${CMD[@]}"

# Check result
if [[ $? -eq 0 ]]; then
    echo "✅ Engine saved to $ENGINE_PATH"
else
    echo "❌ Failed to build engine."
    exit 1
fi
