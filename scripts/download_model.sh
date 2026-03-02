#!/bin/bash
set -e
MODEL_DIR="/home/ubuntu/joe/OpenCapBench/models/synthpose-hrnet-48"
MODEL_FILE="${MODEL_DIR}/hrnet-w48_dark.pth"
MODEL_URL="https://huggingface.co/stanfordmimi/synthpose-hrnet-48-mmpose/resolve/main/hrnet-w48_dark.pth"
mkdir -p "$MODEL_DIR"
echo "Downloading SynthPose HRNet48 checkpoint..."
curl -L --progress-bar -o "$MODEL_FILE" "$MODEL_URL"
echo "Download complete!"
ls -lh "$MODEL_FILE"
