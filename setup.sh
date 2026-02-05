#!/bin/bash
# OpenCapBench Environment Setup Script
# This script automates the installation of all dependencies for OpenCapBench

set -e  # Exit on error

echo "================================================"
echo "OpenCapBench Environment Setup"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if uv is installed
echo -e "${YELLOW}[1/9] Checking for uv package manager...${NC}"
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo -e "${GREEN}✓ uv installed successfully${NC}"
else
    echo -e "${GREEN}✓ uv is already installed${NC}"
fi

# Create Python 3.11 virtual environment
echo ""
echo -e "${YELLOW}[2/9] Creating Python 3.11 virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping...${NC}"
else
    uv venv --python 3.11
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo -e "${YELLOW}[3/9] Activating virtual environment...${NC}"
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Source dev.env if it exists
if [ -f "dev.env" ]; then
    echo ""
    echo -e "${YELLOW}[4/9] Sourcing dev.env for GPU configuration...${NC}"
    source dev.env
    echo -e "${GREEN}✓ GPU environment configured (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)${NC}"
else
    echo ""
    echo -e "${YELLOW}[4/9] No dev.env found, skipping GPU configuration...${NC}"
fi

# Install PyTorch with CUDA 12.8
echo ""
echo -e "${YELLOW}[5/9] Installing PyTorch with CUDA 12.8 support...${NC}"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Verify CUDA availability
echo ""
echo -e "${YELLOW}[6/9] Verifying CUDA availability...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install base dependencies from pyproject.toml
echo ""
echo -e "${YELLOW}[7/9] Installing base dependencies...${NC}"
uv pip install -e .
echo -e "${GREEN}✓ Base dependencies installed${NC}"

# Install MMPose ecosystem
echo ""
echo -e "${YELLOW}[8/9] Installing MMPose ecosystem (mmcv, mmdet, mmpose)...${NC}"
echo "This may take several minutes..."

# Install mmcv
echo "  - Installing mmcv..."
mim install "mmcv>=2.0.0"

# Install mmdet
echo "  - Installing mmdet..."
mim install "mmdet>=3.0.0"

# Install mmpose
echo "  - Installing mmpose..."
mim install "mmpose>=1.0.0"

echo -e "${GREEN}✓ MMPose ecosystem installed${NC}"

# Run verification test
echo ""
echo -e "${YELLOW}[9/9] Running verification tests...${NC}"
if [ -f "scripts/test_imports.py" ]; then
    python scripts/test_imports.py
else
    echo -e "${YELLOW}Test script not found, skipping verification...${NC}"
fi

# Create necessary directories
echo ""
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p data
mkdir -p models
mkdir -p results
echo -e "${GREEN}✓ Directories created${NC}"

# Summary
echo ""
echo "================================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Download the OpenCap dataset (see docs/DATA_DOWNLOAD.md)"
echo ""
echo "  3. Download pre-trained models:"
echo "     huggingface-cli download yonigozlan/synthpose-vitpose-base-hf --local-dir ./models/synthpose-vitpose-base"
echo ""
echo "  4. Configure benchmarking/constants.py with your paths"
echo ""
echo "  5. Run the benchmark:"
echo "     source dev.env"
echo "     python benchmarking/benchmark.py --help"
echo ""
echo "For more information, see README.md"
echo ""
