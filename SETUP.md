# OpenCapBench Environment Setup - Quick Start Guide

This guide helps you set up a complete runnable environment for OpenCapBench to reproduce the evaluation results from the WACV 2025 paper.

## 📋 Prerequisites

- **Python 3.11** (recommended by OpenSim)
- **GPU with CUDA 12.8** support
- **150+ GB free disk space** (dataset + outputs)
- **Linux/Ubuntu** (tested environment)

## 🚀 Quick Setup (Automated)

The easiest way to set up the environment:

```bash
# 1. Navigate to the repository
cd /home/ubuntu/joe/OpenCapBench

# 2. Run the automated setup script
./setup.sh
```

This will:
- Install `uv` package manager (if needed)
- Create a Python 3.11 virtual environment
- Install PyTorch with CUDA 12.8 support
- Install all dependencies (OpenSim, MMPose, etc.)
- Set up project directories

**Time required**: 15-30 minutes

## 📦 Manual Setup (Step-by-Step)

If you prefer manual installation or the automated script fails:

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or ~/.zshrc
```

### 2. Create Virtual Environment

```bash
uv venv --python 3.11
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
# Source GPU configuration
source dev.env

# Install PyTorch with CUDA 12.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install base dependencies
uv pip install -e .

# Install MMPose ecosystem
uv pip install openmim
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
mim install "mmpose>=1.0.0"
```

### 4. Verify Installation

```bash
python scripts/test_imports.py
```

You should see all green checkmarks ✓ for successful imports.

## 📥 Download Data and Models

### Download OpenCap Dataset

See detailed instructions in **[docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md)**

Quick summary:
1. Visit: https://simtk.org/projects/opencap
2. Download subjects 2-11 (10 subjects total)
3. Place in: `/home/ubuntu/joe/OpenCapBench/data/`

### Download Model Weights

See detailed instructions in **[docs/MODEL_DOWNLOAD.md](docs/MODEL_DOWNLOAD.md)**

**For SynthPose models (recommended)**:
```bash
# Create models directory
mkdir -p models

# Download base model
huggingface-cli download yonigozlan/synthpose-vitpose-base-hf \
    --local-dir ./models/synthpose-vitpose-base
```

## ⚙️ Configuration

Edit `benchmarking/constants.py` to match your setup:

```python
config_global = "local"  # Already set

config_base_local = {
    # Update these paths:
    "dataDir": "/home/ubuntu/joe/OpenCapBench/data",
    "model_ckpt_pose": "/home/ubuntu/joe/OpenCapBench/models/synthpose-vitpose-base/pytorch_model.pth",
    
    # Leave other settings as default for initial testing
    ...
}
```

## 🧪 Testing Your Setup

### Quick Test (Single Subject)

```bash
./scripts/quick_test.sh
```

Runs evaluation on subject2, Session0 only (~30-60 minutes).

### Full Evaluation

```bash
./scripts/run_evaluation.sh
```

Runs full evaluation on all subjects and sessions (~8-12 hours).

## 📊 Expected Results

After successful evaluation, you should see:

```
data/evaluation_run_XXXXXXXX/
├── subject2_Session0/
│   ├── MarkerData/*.trc          # 3D trajectories
│   ├── OpenSimData/Kinematics/*.mot  # Joint angles
├── ...
├── evaluation_run_XXXXXXXXmean_rmses.csv  # Main results
├── evaluation_run_XXXXXXXXmean_rmses_no_shift.csv
└── shifts.json
```

**Paper results** (SynthPose models):
- Mean joint angle RMSE: **5-8 degrees**
- Metrics for: pelvis, hip, knee, ankle, lumbar, shoulder, elbow

## 📁 Project Structure

```
OpenCapBench/
├── benchmarking/          # Main evaluation code
│   ├── benchmark.py       # Entry point
│   ├── constants.py       # Configuration
│   └── ...
├── docs/                  # Documentation
│   ├── DATA_DOWNLOAD.md   # Dataset guide
│   └── MODEL_DOWNLOAD.md  # Model guide
├── scripts/               # Helper scripts
│   ├── test_imports.py    # Verify setup
│   ├── run_evaluation.sh  # Full evaluation
│   └── quick_test.sh      # Quick test
├── setup.sh               # Automated setup
├── pyproject.toml         # Dependencies
└── dev.env                # GPU configuration
```

## 🔧 Troubleshooting

### Import Errors

```bash
# Re-run setup
./setup.sh

# Or install missing package manually
uv pip install <package-name>
```

### GPU Not Detected

```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Verify dev.env is sourced
source dev.env
echo $CUDA_VISIBLE_DEVICES
```

### Out of Memory

Edit `benchmarking/constants.py`:
```python
"batch_size_pose": 16,  # Reduce from 32
```

### Model Not Found

Verify paths in `benchmarking/constants.py` match your downloaded models:
```bash
ls -lh models/synthpose-vitpose-base/
```

## 📚 Additional Resources

- **Paper**: https://arxiv.org/abs/2406.09788
- **Dataset**: https://simtk.org/projects/opencap
- **MMPose Docs**: https://mmpose.readthedocs.io/
- **OpenSim Docs**: https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/

## 🆘 Getting Help

1. Check the error message and troubleshooting section above
2. Run verification: `python scripts/test_imports.py`
3. Check configuration: `benchmarking/constants.py`
4. See detailed guides: `docs/DATA_DOWNLOAD.md`, `docs/MODEL_DOWNLOAD.md`

## ✅ Checklist

Before running evaluation, ensure:

- [ ] Virtual environment created and activated
- [ ] All dependencies installed (green ✓ from test_imports.py)
- [ ] GPU detected and CUDA available
- [ ] OpenCap dataset downloaded and organized
- [ ] Model weights downloaded
- [ ] `constants.py` configured with correct paths
- [ ] Quick test runs successfully

## 🎯 Next Steps

Once setup is complete:

1. **Run quick test**: `./scripts/quick_test.sh`
2. **Verify results**: Check output CSV files
3. **Run full evaluation**: `./scripts/run_evaluation.sh`
4. **Compare with paper**: Results should match ~5-8° RMSE

Good luck with your experiments! 🚀
