# Model Weights Download Guide

This guide explains how to download pre-trained model weights for use with OpenCapBench.

## Overview

OpenCapBench supports two types of models:
1. **SynthPose models** (recommended): Fine-tuned for anatomical markers - best results
2. **Standard COCO models**: Baseline models for comparison

## SynthPose Models (Recommended)

### Available Models

Two SynthPose models are available on Hugging Face:

1. **synthpose-vitpose-base-hf**: Faster, good accuracy
   - URL: https://huggingface.co/yonigozlan/synthpose-vitpose-base-hf
   - Paper results: ~5-8° joint angle RMSE

2. **synthpose-vitpose-huge-hf**: Slower, best accuracy
   - URL: https://huggingface.co/yonigozlan/synthpose-vitpose-huge-hf
   - Paper results: ~4-7° joint angle RMSE

### Download Instructions

#### Method 1: Using Hugging Face CLI (Recommended)

```bash
# Ensure you're in the OpenCapBench directory
cd /home/ubuntu/joe/OpenCapBench

# Create models directory
mkdir -p models

# Activate your environment
source .venv/bin/activate

# Download base model
huggingface-cli download yonigozlan/synthpose-vitpose-base-hf \
    --local-dir ./models/synthpose-vitpose-base

# OR download huge model (better accuracy, slower)
huggingface-cli download yonigozlan/synthpose-vitpose-huge-hf \
    --local-dir ./models/synthpose-vitpose-huge
```

#### Method 2: Manual Download

1. Visit the model page:
   - Base: https://huggingface.co/yonigozlan/synthpose-vitpose-base-hf
   - Huge: https://huggingface.co/yonigozlan/synthpose-vitpose-huge-hf

2. Download these files:
   - `pytorch_model.pth` (main weights)
   - `config.json` (model configuration)
   - `model_config.py` (MMPose config)

3. Place in: `models/synthpose-vitpose-base/` or `models/synthpose-vitpose-huge/`

### Model Size

- **Base model**: ~300-400 MB
- **Huge model**: ~1.5-2 GB

## Standard COCO Models

For baseline comparison or if you don't have SynthPose models:

### HRNet-W48 (COCO)

This model is automatically downloaded by MMPose:

```python
# In benchmarking/constants.py
config_base_local = {
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py",
    "model_ckpt_pose": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth",
    "marker_set": "Coco",  # Important: Use "Coco" for COCO-trained models
}
```

No manual download needed - MMPose will download automatically on first run.

## MMPose Model Configurations

### Option 1: Using System-Wide MMPose

If you installed MMPose via pip/uv:

```python
# In benchmarking/constants.py
config_base_local = {
    "mmposeDirectory": "",  # Leave empty
    "model_config_pose": "configs/body_2d_keypoint/...",  # Relative to mmpose install
    ...
}
```

MMPose configs are in your Python environment's site-packages.

### Option 2: Using Cloned MMPose Fork

For SynthPose models with custom "Infinity" dataset configs:

```bash
# Clone the author's MMPose fork
cd /home/ubuntu/joe
git clone https://github.com/yonigozlan/mmpose.git
cd mmpose
source /home/ubuntu/joe/OpenCapBench/.venv/bin/activate
uv pip install -v -e .
```

Then update constants.py:
```python
config_base_local = {
    "mmposeDirectory": "/home/ubuntu/joe/mmpose",
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py",
    ...
}
```

## Person Detector Models

Person detection models are auto-downloaded:

### Faster R-CNN (Default)
- Fast, good enough for most cases
- Auto-downloaded from MMDetection model zoo

### ConvNeXt (Better Performance)
```python
# In benchmarking/constants.py - uncomment these lines:
"model_config_person": "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
"model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
```

## Verify Model Setup

After downloading models:

```bash
# Check model files
ls -lh models/synthpose-vitpose-base/
# Should show: pytorch_model.pth, config.json, etc.

# Verify configuration
python -c "
from benchmarking.constants import config
print(f'Model config: {config[\"model_config_pose\"]}')
print(f'Model checkpoint: {config[\"model_ckpt_pose\"]}')
print(f'Marker set: {config[\"marker_set\"]}')
"
```

## Configuration Summary

### For SynthPose Models:

```python
# benchmarking/constants.py
config_global = "local"

config_base_local = {
    "mmposeDirectory": "/home/ubuntu/joe/mmpose",  # If using cloned fork
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py",
    "model_ckpt_pose": "/home/ubuntu/joe/OpenCapBench/models/synthpose-vitpose-base/pytorch_model.pth",
    "marker_set": "Anatomical",  # Important!
    ...
}
```

### For COCO Models:

```python
# benchmarking/constants.py
config_global = "local"

config_base_local = {
    "mmposeDirectory": "",  # System-wide install
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py",
    "model_ckpt_pose": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth",
    "marker_set": "Coco",  # Important!
    ...
}
```

## Troubleshooting

### Download Fails
```bash
# Try with git-lfs if clone method fails
git lfs install
huggingface-cli download --resume-download yonigozlan/synthpose-vitpose-base-hf \
    --local-dir ./models/synthpose-vitpose-base
```

### Model Not Found
- Verify file paths in constants.py
- Check that model files exist: `ls models/*/`
- Ensure marker_set matches model type ("Anatomical" for SynthPose, "Coco" for COCO)

### Config File Not Found
- For SynthPose: Clone the MMPose fork and set mmposeDirectory
- For COCO: Leave mmposeDirectory empty and use relative paths

### Out of Memory
- Use base model instead of huge
- Reduce batch_size_pose in constants.py (e.g., 16 instead of 32)

## Expected Results

With properly configured models:

| Model Type | Marker Set | Mean RMSE (degrees) |
|-----------|-----------|---------------------|
| SynthPose Base | Anatomical | 5-8° |
| SynthPose Huge | Anatomical | 4-7° |
| HRNet-W48 COCO | Coco | 10-15° |

## Next Steps

After downloading models:

1. **Test imports** (see scripts/test_imports.py)
2. **Run a quick test**:
   ```bash
   python benchmarking/benchmark.py \
       --subjects subject2 \
       --sessions Session0 \
       --dataName test_run
   ```
3. **Run full evaluation** (see scripts/run_evaluation.sh)

## References

- SynthPose HuggingFace: https://huggingface.co/yonigozlan
- MMPose Model Zoo: https://mmpose.readthedocs.io/en/latest/model_zoo.html
- MMDetection Models: https://mmdetection.readthedocs.io/en/latest/model_zoo.html
