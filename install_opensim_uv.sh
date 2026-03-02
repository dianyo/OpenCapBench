#!/bin/bash
# Install OpenSim into a uv virtual environment
#
# OpenSim is not on PyPI, so we use a temporary conda environment
# to download it, then copy the package into the uv venv.
# The temp conda env is deleted afterward.
#
# Tested: OpenSim 4.5.2 with Python 3.11 on Ubuntu

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "================================================"
echo "OpenSim Installation for UV Environment"
echo "================================================"
echo ""

if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: No virtual environment activated${NC}"
    echo "  source .venv/bin/activate"
    exit 1
fi

if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found${NC}"
    echo "Install miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

PY_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
UV_SP="$VIRTUAL_ENV/lib/python${PY_VERSION}/site-packages"
TEMP_ENV="_opensim_tmp_$$"

echo "UV venv: $VIRTUAL_ENV"
echo "Python:  $PY_VERSION"
echo ""

# Step 1: Create temporary conda environment
echo -e "${YELLOW}[1/4] Creating temporary conda environment (python=$PY_VERSION)...${NC}"
conda create -n "$TEMP_ENV" "python=$PY_VERSION" -y > /dev/null 2>&1
CONDA_SP="$(conda info --envs | grep "$TEMP_ENV" | awk '{print $NF}')/lib/python${PY_VERSION}/site-packages"

# Step 2: Install opensim via conda
echo -e "${YELLOW}[2/4] Installing OpenSim via conda...${NC}"
conda install -n "$TEMP_ENV" -c opensim-org opensim -y > /dev/null 2>&1
CONDA_LIB="$(conda info --envs | grep "$TEMP_ENV" | awk '{print $NF}')/lib"

# Step 3: Copy opensim package + shared libraries into uv venv
echo -e "${YELLOW}[3/4] Copying OpenSim into uv venv...${NC}"
cp -r "$CONDA_SP/opensim" "$UV_SP/"
cp "$CONDA_LIB"/libosim* "$UV_SP/opensim/" 2>/dev/null || true
cp "$CONDA_LIB"/libcasadi* "$UV_SP/opensim/" 2>/dev/null || true

# Step 4: Clean up
echo -e "${YELLOW}[4/4] Cleaning up temporary conda environment...${NC}"
conda env remove -n "$TEMP_ENV" -y > /dev/null 2>&1

# Verify
echo ""
if python -c "import opensim; print('OpenSim', opensim.GetVersionAndDate())" 2>/dev/null; then
    echo -e "${GREEN}OpenSim installed successfully into uv venv!${NC}"
else
    echo -e "${RED}OpenSim import failed. Check error above.${NC}"
    exit 1
fi
