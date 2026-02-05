#!/bin/bash
# Quick test run script - evaluates a single subject to verify setup

set -e

echo "================================================"
echo "OpenCapBench Quick Test Run"
echo "================================================"
echo ""
echo "This will run a quick evaluation on subject2, Session0 only"
echo "to verify your setup is working correctly."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate environment
source .venv/bin/activate

# Source GPU config if available
if [ -f "dev.env" ]; then
    source dev.env
fi

# Run quick test
python benchmarking/benchmark.py \
    --dataDir "${DATA_DIR:-/home/ubuntu/joe/OpenCapBench/data}" \
    --dataName "quick_test_$(date +%Y%m%d_%H%M%S)" \
    --subjects "subject2" \
    --sessions "Session0" \
    --cameraSetups "2-cameras" \
    --marker_set "Anatomical"

echo ""
echo "Quick test completed!"
echo "If this worked, you're ready to run the full evaluation."
