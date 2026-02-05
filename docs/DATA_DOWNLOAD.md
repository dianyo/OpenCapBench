# OpenCap Dataset Download and Setup Guide

This guide walks you through downloading and organizing the OpenCap dataset for use with OpenCapBench.

## Dataset Overview

The OpenCapBench evaluation uses the OpenCap dataset, which includes:
- **10 subjects** (subject2 through subject11)
- **2 sessions per subject** (Session0 and Session1)
- **Multiple trials per session**: static pose, sit-to-stand, squat, drop jump, walking
- **5-camera setup** with calibration data
- **Motion capture ground truth** for joint angles

## Download Instructions

### 1. Access the Dataset

Visit the OpenCap dataset page:
**https://simtk.org/projects/opencap**

You may need to:
1. Create a free SimTK account
2. Request access to the dataset (usually approved quickly)
3. Download the dataset files

### 2. Download Required Files

The dataset is organized by subject. For each subject (2-11), download:

```
subject<N>/
├── VideoData/
│   ├── Session0/
│   │   ├── Cam0/, Cam1/, Cam2/, Cam3/, Cam4/
│   │   │   ├── cameraIntrinsicsExtrinsics.pickle
│   │   │   └── <trial_name>/
│   │   │       └── <trial_name>_syncdWithMocap.avi (or .avi for extrinsics)
│   └── Session1/
│       └── (same structure as Session0)
├── OpenSimData/
│   └── Mocap/
│       ├── IK/
│       │   └── *.mot (ground truth joint angles)
│       └── Model/
│           └── LaiArnoldModified2017_poly_withArms_weldHand_scaled_adjusted.osim
└── sessionMetadata.yaml
```

### 3. Organize the Dataset

Place the downloaded data in your chosen data directory:

```bash
# Default location (from constants.py)
mkdir -p /home/ubuntu/joe/OpenCapBench/data

# Your structure should look like:
OpenCapBench/data/
├── subject2/
│   ├── VideoData/
│   ├── OpenSimData/
│   └── sessionMetadata.yaml
├── subject3/
├── subject4/
├── subject5/
├── subject6/
├── subject7/
├── subject8/
├── subject9/
├── subject10/
└── subject11/
```

### 4. Verify Download

Check that you have all required components:

```bash
# Navigate to your data directory
cd /home/ubuntu/joe/OpenCapBench/data

# Check subjects
ls -d subject*
# Should show: subject2 subject3 ... subject11

# Check a sample subject structure
tree -L 3 subject2
# Should show VideoData, OpenSimData, and sessionMetadata.yaml

# Count video files (should be ~200+ total across all subjects)
find . -name "*.avi" | wc -l

# Count ground truth motion files
find . -name "*.mot" -path "*/Mocap/IK/*" | wc -l
# Should be ~160+ files (2 sessions × ~8 trials × 10 subjects)
```

## Expected Trials per Session

### Session 0 (subject<N>_Session0):
- `extrinsics`: Camera calibration trial
- `static`: Neutral pose for model scaling
- `STS` (Sit-to-Stand): Functional movement
- `squats`: Squatting motion
- `DJ` (Drop Jump): Dynamic movement

### Session 1 (subject<N>_Session1):
- `extrinsics`: Camera calibration trial
- `walk`: Walking trials at different speeds

Note: Not all subjects have all trials. The benchmark script handles missing trials automatically.

## Camera Configuration

The dataset includes 5 cameras positioned at different angles:
- **Cam0**: -70° (left)
- **Cam1**: -45°
- **Cam2**: 0° (center, facing participant)
- **Cam3**: 45°
- **Cam4**: 70° (right)

The benchmark can evaluate with:
- **2-cameras**: Cam1 + Cam3 (default for paper)
- **3-cameras**: Cam1 + Cam2 + Cam3
- **5-cameras**: All cameras

## Important Files

### Camera Calibration
Each camera directory contains `cameraIntrinsicsExtrinsics.pickle` with:
- Camera intrinsic parameters (focal length, principal point, distortion)
- Extrinsic parameters (rotation, translation)

The repository includes pre-computed calibrations in `benchmarking/CameraIntrinsics/`.

### Ground Truth Data
Motion capture ground truth is in:
```
subject<N>/OpenSimData/Mocap/IK/<trial_name>.mot
```

These `.mot` files contain joint angles computed from marker-based motion capture, serving as ground truth for evaluation.

### Session Metadata
Each subject has `sessionMetadata.yaml` containing:
```yaml
mass_kg: <subject mass in kg>
height_m: <subject height in meters>
openSimModel: "LaiUhlrich2022"
```

## Dataset Size

**Total dataset size**: ~50-100 GB depending on compression
- Videos: ~40-80 GB
- OpenSim data: ~1-5 GB
- Metadata: <100 MB

**Recommended**: Ensure you have at least 150 GB free space (dataset + processing outputs)

## Troubleshooting

### Missing Files
If certain trials are missing, the benchmark will skip them automatically. However, ensure you have:
- At least 1 session per subject
- Ground truth `.mot` files for evaluation
- Camera calibration `.pickle` files

### Download Issues
- **Slow downloads**: SimTK servers may be slow; consider downloading in batches
- **Access denied**: Ensure you've requested and been granted dataset access
- **Corrupted files**: Verify file sizes match expected values; re-download if needed

### Storage Issues
If storage is limited:
1. Download subjects sequentially and process them individually
2. Delete processed subject videos after computing results
3. Keep only the OpenSimData for ground truth comparison

## Next Steps

After downloading the data:

1. **Update constants.py**:
   ```python
   "dataDir": "/home/ubuntu/joe/OpenCapBench/data"
   ```

2. **Download model weights** (see MODEL_DOWNLOAD.md)

3. **Run the benchmark**:
   ```bash
   source dev.env
   python benchmarking/benchmark.py --dataDir /home/ubuntu/joe/OpenCapBench/data --dataName test_run
   ```

## References

- OpenCap Project: https://www.opencap.ai/
- Dataset Paper: https://simtk.org/projects/opencap
- OpenCapBench Paper: https://arxiv.org/abs/2406.09788

## Support

For dataset-specific issues:
- Visit: https://simtk.org/projects/opencap
- Email: OpenCap team via SimTK

For OpenCapBench issues:
- GitHub: https://github.com/StanfordMIMI/OpenCapBench
