#!/bin/bash
# Example script to run OpenCapBench evaluation
# This script demonstrates how to run the benchmark with proper configuration

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;94m'
NC='\033[0m' # No Color

echo "================================================"
echo "OpenCapBench Evaluation Script"
echo "================================================"
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run setup.sh first:"
    echo "  ./setup.sh"
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}[1/4] Activating virtual environment...${NC}"
source .venv/bin/activate

# Source GPU configuration if available
if [ -f "dev.env" ]; then
    echo -e "${BLUE}[2/4] Loading GPU configuration...${NC}"
    source dev.env
    echo "  GPU device: $CUDA_VISIBLE_DEVICES"
else
    echo -e "${YELLOW}[2/4] No dev.env found, using default GPU configuration${NC}"
fi

# Configuration (modify these for your setup)
DATA_DIR="${DATA_DIR:-/home/ubuntu/joe/OpenCapBench/data}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py}"
MODEL_CKPT="${MODEL_CKPT:-/home/ubuntu/joe/OpenCapBench/models/synthpose-vitpose-base/pytorch_model.pth}"
DATA_NAME="${DATA_NAME:-evaluation_run_$(date +%Y%m%d_%H%M%S)}"
SUBJECTS="${SUBJECTS:-all}"
SESSIONS="${SESSIONS:-all}"
CAMERA_SETUP="${CAMERA_SETUP:-2-cameras}"
MARKER_SET="${MARKER_SET:-Anatomical}"

echo -e "${BLUE}[3/4] Configuration:${NC}"
echo "  Data directory: $DATA_DIR"
echo "  Model config: $MODEL_CONFIG"
echo "  Model checkpoint: $MODEL_CKPT"
echo "  Output name: $DATA_NAME"
echo "  Subjects: $SUBJECTS"
echo "  Sessions: $SESSIONS"
echo "  Camera setup: $CAMERA_SETUP"
echo "  Marker set: $MARKER_SET"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory not found: $DATA_DIR${NC}"
    echo "Please download the OpenCap dataset first (see docs/DATA_DOWNLOAD.md)"
    exit 1
fi

# Check if model exists (if not a URL)
if [[ ! "$MODEL_CKPT" =~ ^https?:// ]] && [ ! -f "$MODEL_CKPT" ]; then
    echo -e "${RED}Error: Model checkpoint not found: $MODEL_CKPT${NC}"
    echo "Please download model weights first (see docs/MODEL_DOWNLOAD.md)"
    exit 1
fi

# Run the benchmark
echo -e "${BLUE}[4/4] Running benchmark...${NC}"
echo ""
echo "This may take several hours depending on:"
echo "  - Number of subjects/sessions"
echo "  - GPU speed"
echo "  - Camera setup"
echo ""
echo "Press Ctrl+C to cancel within 5 seconds..."
sleep 5

python benchmarking/benchmark.py \
    --model_config_pose "$MODEL_CONFIG" \
    --model_ckpt_pose "$MODEL_CKPT" \
    --dataDir "$DATA_DIR" \
    --dataName "$DATA_NAME" \
    --subjects "$SUBJECTS" \
    --sessions "$SESSIONS" \
    --cameraSetups "$CAMERA_SETUP" \
    --marker_set "$MARKER_SET"

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo -e "${GREEN}Evaluation completed successfully!${NC}"
    echo "================================================"
    echo ""
    echo "Results saved to:"
    echo "  $DATA_DIR/$DATA_NAME/"
    echo ""
    echo "Key output files:"
    echo "  - MarkerData/*.trc (3D marker trajectories)"
    echo "  - OpenSimData/Kinematics/*.mot (joint angles)"
    echo "  - ${DATA_NAME}mean_rmses.csv (metrics with temporal alignment)"
    echo "  - ${DATA_NAME}mean_rmses_no_shift.csv (metrics without alignment)"
    echo "  - shifts.json (temporal alignment offsets)"
    echo ""
else
    echo ""
    echo "================================================"
    echo -e "${RED}Evaluation failed!${NC}"
    echo "================================================"
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - Missing dependencies (run: python scripts/test_imports.py)"
    echo "  - Incorrect paths in benchmarking/constants.py"
    echo "  - GPU out of memory (reduce batch_size_pose in constants.py)"
    echo "  - Missing data files"
    echo ""
    exit 1
fi
