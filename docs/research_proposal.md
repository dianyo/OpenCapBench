# Research Proposal: What Makes a Token Matter for Biomechanics? Efficient Anatomical Pose Estimation via Token Analysis and State Space Models

## 1. Problem Statement

Markerless motion capture is poised to replace expensive marker-based systems in
clinical biomechanics. The SynthPose + OpenCapBench pipeline (Gozlan et al., 2024)
demonstrated that fine-tuning 2D pose models to predict **52 anatomical keypoints**
(rather than 17 COCO keypoints) yields significantly better kinematic accuracy when
fed through OpenSim inverse kinematics.

However, the best-performing models — ViTPose-Huge (637M params, 2.4GB) and
ViTPose-Base (90M params) — are too large for real-time clinical edge deployment
(mobile devices, embedded GPUs in clinics, wearable systems). HRNet-48 (~64M params)
is smaller but still heavy, and all these models require the full SynthPose
fine-tuning procedure on large synthetic datasets (BEDLAM, Infinity).

**The gap:** No existing work explores whether modern efficient architectures can
achieve competitive biomechanical accuracy at a fraction of the compute — making
markerless motion capture truly deployable.

A deeper question also remains unanswered: ViT-based models like ViTPose process
all image tokens uniformly, yet biomechanical accuracy depends heavily on a
*subset* of anatomical keypoints (e.g., pelvis markers drive hip kinematics).
Popular ViT efficiency methods (ToMe, EViT, DynamicViT) reduce compute by
pruning or merging "unimportant" tokens — but importance is defined by
classification loss, not biomechanical fidelity. **Do these methods preserve the
tokens that matter for joint angle accuracy? Or does naive token reduction
disproportionately degrade biomechanical metrics?** This question has never been
studied.

## 2. Existing Work & Positioning

### Mamba/SSM for Pose (2024-2025)

| Method | Venue | Task | Key Difference from Ours |
|---|---|---|---|
| PoseMamba (2024) | AAAI 2025 | 2D-to-3D lifting (Human3.6M) | Lifts existing 2D keypoints to 3D; NOT a 2D pose backbone |
| SAMA (2025) | arXiv | 2D-to-3D lifting with structure-awareness | Same — post-hoc 3D lifting, not 2D detection |
| SasMamba (2025) | WACV 2026 | 2D-to-3D lifting with skeleton-aware scanning | Same — no biomechanical evaluation |
| Mamba2D (2024) | arXiv | General 2D SSM formulation for vision | Vision backbone, not applied to pose |

**Critical gap:** All existing Mamba-for-pose works are 2D-to-3D *lifting* networks
that take pre-detected 2D keypoints as input. None of them:
- Serve as the **2D pose estimation backbone** itself (image to heatmaps)
- Predict **anatomical/biomechanical markers** (52 keypoints, not 17 COCO)
- Evaluate on **kinematic metrics** (joint angle RMSE via OpenSim)
- Target **edge deployment** for clinical biomechanics

### ViT Efficiency Methods (not yet applied to biomechanical pose)

| Method | Venue | Technique | Applied to Pose? |
|---|---|---|---|
| ToMe (Bolya et al.) | ICLR 2023 | Token merging via bipartite matching, training-free | No |
| EViT (Liang et al.) | ICLR 2022 | Attentive token pruning, fuse dropped tokens | No |
| DynamicViT (Rao et al.) | NeurIPS 2021 | Learned token sparsification via prediction module | No |
| PITOME (2024) | NeurIPS 2024 | Energy-score token merging preserving spectral properties | No |
| SDPose (2024) | arXiv | Self-distillation for small pose transformers (4.4M, 1.8G) | Yes (COCO only) |
| PoseSynViT (2025) | CVPR-W 2025 | Lightweight ViT with knowledge token for pose | Yes (COCO only) |

**Critical gap:** Token pruning/merging methods have never been evaluated on
**biomechanical metrics**. SDPose and PoseSynViT target COCO AP — a metric where
all 17 keypoints contribute roughly equally. In biomechanics, a handful of
keypoints (pelvis, spine) drive the majority of kinematic accuracy. Standard
token reduction may discard the very spatial regions these keypoints occupy.

### SynthPose / OpenCapBench (Gozlan et al., 2024)

Evaluated architectures: HRNet-32, HRNet-48, ViTPose-Base, ViTPose-Huge.
All are CNN or ViT-based. **No SSM backbone or ViT efficiency method has been
evaluated on OpenCapBench.**

## 3. Proposed Approach

### Core Idea

Replace the 2D pose estimation backbone (HRNet/ViTPose) with a Mamba-based vision
backbone, fine-tune using the SynthPose pipeline for 52 anatomical keypoints, and
evaluate on OpenCapBench — with a focus on the Pareto frontier of
**accuracy vs. efficiency**.

### Architecture

Use an existing vision SSM backbone (candidates below) as the encoder in a
top-down heatmap regression framework (same as OpenCapBench's setup):

| Candidate Backbone | Params | FLOPs | Notes |
|---|---|---|---|
| VMamba-Tiny | ~31M | ~4.9G | Hierarchical, good for dense prediction |
| PlainMamba-L1 | ~7M | ~3.0G | Minimal, aggressive efficiency |
| PlainMamba-L3 | ~50M | ~12G | Larger, closer to ViTPose-Base |
| LocalVMamba-S | ~50M | ~11G | Local attention + SSM hybrid |
| EfficientVMamba-S | ~11M | ~1.3G | Ultra-lightweight |

For comparison, the baselines:

| Baseline | Params | FLOPs |
|---|---|---|
| HRNet-W32 | ~29M | ~7.7G |
| HRNet-W48 | ~64M | ~14.6G |
| ViTPose-Base | ~90M | ~17.1G |
| ViTPose-Huge | ~637M | ~140G |

The SSM backbones offer **2-10x parameter reduction** and **3-100x FLOPs reduction**
compared to ViTPose-Huge, while operating in the same parameter range as HRNet-32.

### Research Angle: Token Importance for Biomechanics

Before proposing an efficient architecture, we first ask: **what visual
information does a pose model actually need for biomechanical accuracy?**

We use ViTPose's attention maps as an analytical tool:

1. **Attention-to-keypoint mapping:** For each of the 52 anatomical keypoints,
   extract the attention rollout from ViTPose-Base/Huge to identify which image
   tokens contribute most to each keypoint's heatmap prediction. This produces
   a per-keypoint "token importance map."

2. **Keypoint-to-kinematics sensitivity:** Not all keypoints contribute equally
   to joint angle accuracy. Perturb each keypoint independently (add noise or
   drop it) and measure the resulting RMSE change per joint DOF through the full
   OpenSim IK pipeline. This yields a **keypoint importance ranking for
   biomechanics** — which has never been established.

3. **Token pruning under biomechanical evaluation:** Apply ToMe and EViT to
   ViTPose at varying reduction rates (10%, 30%, 50%, 70% tokens removed).
   Measure not just COCO AP (standard) but joint angle RMSE (biomechanical).
   **Hypothesis:** Biomechanical accuracy degrades faster than COCO AP because
   token pruning disproportionately discards regions around torso/pelvis markers.

4. **The SSM alternative:** SSMs process all tokens in linear time — no need to
   decide which to drop. If token pruning hurts biomechanics but SSMs process
   everything cheaply, that is the architectural argument for SSM in this domain.

### Training Pipeline

Same as SynthPose — this is intentional. We keep the training pipeline identical
to isolate the backbone contribution:

1. Start with ImageNet-pretrained SSM backbone
2. Attach top-down heatmap head (same as HRNet/ViTPose setup in mmpose)
3. Fine-tune on SynthPose synthetic data (BEDLAM + Infinity + COCO + 3DPW)
4. Evaluate on OpenCapBench (joint angle RMSE via OpenSim IK)

## 4. Research Questions & Experiments

### RQ1: Which tokens and keypoints actually matter for biomechanics? (Analysis)

This is the core scientific question that elevates the paper beyond an
architecture comparison.

**Exp 1a — Keypoint importance for kinematics:**
- Systematically drop each of the 52 keypoints (set to zero confidence) and
  re-run the full pipeline (triangulation + OpenSim IK) using existing ViTPose
  predictions
- Measure per-joint RMSE change: which keypoints are critical for which DOFs?
- Produce a **52x12 sensitivity matrix** (keypoints x joint DOFs)
- Expected finding: pelvis markers (r/l_ASIS, r/l_PSIS) dominate hip/pelvis
  kinematics; medial/lateral markers dominate knee/ankle — but some keypoints
  may be near-redundant
- **This artifact alone is a contribution** — the biomechanics community has no
  such analysis for predicted (vs. physical) markers

**Exp 1b — ViT attention analysis:**
- Extract attention rollout from ViTPose-Base on OpenCap test frames
- For each anatomical keypoint, compute the average attention mass over the
  image spatial grid — identifying which image regions the model relies on
- Visualize: do torso/pelvis keypoints draw attention from a wide spatial
  context? (Suggesting they need global information that token pruning removes)

**Exp 1c — Token pruning degrades biomechanics disproportionately:**
- Apply ToMe to ViTPose-Base at r={5, 10, 20, 30, 40} (tokens merged per layer)
- Apply EViT-style top-k token keeping at {90%, 70%, 50%, 30%} retention
- Measure both COCO AP (if available) AND OpenCapBench joint angle RMSE
- Plot: token reduction rate vs. RMSE degradation curve
- **Hypothesis:** At 50% token reduction, COCO AP drops ~1-2% but pelvis/spine
  RMSE may jump >20% because these keypoints occupy few tokens in a torso region
  that pruning algorithms consider "background-like"
- If confirmed, this finding motivates the need for architectures that do NOT
  discard tokens — i.e., SSMs

### RQ2: Can SSM backbones match CNN/ViT accuracy on biomechanical metrics?

- Train VMamba-Tiny, PlainMamba-L1/L3 with full SynthPose pipeline
- Evaluate on OpenCapBench (joint angle RMSE per DOF)
- Compare against HRNet-32, HRNet-48, ViTPose-Base, ViTPose-Huge
- Compare against ViTPose-Base + ToMe (the "efficient ViT" baseline)
- **Success criterion:** Match HRNet-48 accuracy at <=50% of its FLOPs,
  AND match or beat token-pruned ViTPose at equivalent FLOPs

### RQ3: Are SSMs more data-efficient for synthetic-to-real transfer?

This is the most interesting scientific question. SSMs have structured state
representations that might encode human body priors more efficiently than
attention or convolutions.

- Ablation: train all backbones with {10%, 25%, 50%, 100%} of synthetic data
- Plot accuracy vs. training data curves for each architecture
- **Hypothesis:** SSM reaches HRNet-48 full-data accuracy with <=50% data
- **Why this matters:** Generating synthetic data (SynthPose pipeline) requires
  SMPL-X fitting + rendering — reducing this need is practically valuable

### RQ4: Model efficiency profiling (GPU / server-side)

Before edge deployment, we need to establish that SSMs are genuinely faster
on standard hardware — this is the core efficiency claim.

**Exp 3a — Throughput & latency (single GPU, A100/4090):**
- Measure at input 384x288, batch sizes {1, 4, 16, 32, 64}
- Metrics: images/sec, latency-per-image (ms), latency p50/p95/p99
- All models: HRNet-32, HRNet-48, ViTPose-Base, ViTPose-Huge, VMamba-Tiny, PlainMamba-L1/L3
- Use PyTorch (fp32), PyTorch (fp16/bf16), and torch.compile()
- **Expected table in paper: Accuracy (RMSE) vs. Throughput (img/s) Pareto plot**

**Exp 3b — Computational cost breakdown:**
- FLOPs (MACs) via fvcore or ptflops
- Parameter count (total, backbone-only, head-only)
- Peak GPU memory at batch=1 and batch=32
- Training cost: GPU-hours to convergence on SynthPose data (A100)
- **Expected figure: scatter plot — x=GFLOPs, y=RMSE, bubble size=params**

**Exp 3c — Inference scaling behavior:**
- How does latency scale with input resolution? Test {256x192, 384x288, 512x384}
- SSM should scale linearly (O(n)) vs. ViT quadratic (O(n^2)) — demonstrate this
- Plot: resolution vs. latency curves for SSM/ViT/CNN
- This is the key theoretical advantage — must show it empirically

**Exp 3d — Full pipeline wall-clock time:**
- OpenCapBench processes a full trial (multi-camera video, hundreds of frames)
- Measure end-to-end time per trial for each model (detection + pose + triangulation + IK)
- Pose model is the bottleneck — show how much SSM reduces total pipeline time
- Metric: minutes per subject (10 subjects x 2 sessions x ~5 trials each)

### RQ5: Edge deployment feasibility

- Export to ONNX (opset 17+), then TensorRT (FP16) for Jetson Orin Nano
- Also test: ONNX Runtime on x86 CPU (simulating clinic laptop without GPU)
- Measure: FPS, peak RAM/VRAM, power draw (watts), energy per frame (mJ)
- Compare full pipeline: person detector (fixed) + pose model (variable)
- **Target:** >=30 FPS pose inference at 384x288 on Jetson Orin Nano
- **Stretch target:** >=15 FPS on mobile CPU (ONNX Runtime, no GPU)
- Note: If SSM custom CUDA kernels don't export cleanly to ONNX/TRT, this is a
  known risk. Mitigation: use the pure-PyTorch SSM implementations or approximate
  the selective scan with standard ops for export.

### RQ6: Which biomechanical joints benefit most from SSMs?

- Per-joint breakdown: hip, knee, ankle, pelvis, lumbar spine
- Analysis: Do SSMs better capture pelvis/spine markers (spatially distant from
  extremities, where global context matters)?
- This provides the **"why SSMs for biomechanics"** narrative

### Experiment Summary

| ID | Experiment | Key Metric | Expected Paper Artifact |
|---|---|---|---|
| **1a** | Keypoint importance for kinematics | per-DOF RMSE delta when keypoint dropped | 52x12 sensitivity matrix (main contribution) |
| **1b** | ViT attention analysis | spatial attention mass per keypoint | Attention heatmaps (qualitative figure) |
| **1c** | Token pruning vs biomechanics | RMSE at {10-70}% token reduction | Plot: token reduction vs RMSE degradation |
| **2** | SSM backbone accuracy | joint angle RMSE (deg) | Main results table (SSM vs CNN vs ViT vs pruned-ViT) |
| **3** | Data efficiency | RMSE at {10,25,50,100}% data | Plot: data scaling curves |
| **4a** | GPU throughput & latency | img/sec, latency ms (p50/p95) | Table: all models x batch sizes |
| **4b** | Compute cost breakdown | GFLOPs, params, GPU memory | Scatter: GFLOPs vs RMSE (main figure) |
| **4c** | Resolution scaling | latency vs resolution curve | Plot: O(n) SSM vs O(n^2) ViT |
| **4d** | Full pipeline wall-clock | minutes per subject | Table: end-to-end time comparison |
| **5a** | Jetson Orin Nano | FPS, watts, mJ/frame | Table: edge deployment benchmarks |
| **5b** | CPU-only (ONNX Runtime) | FPS on x86 laptop | Feasibility for clinics without GPUs |
| **6** | Per-joint SSM analysis | per-DOF RMSE breakdown | Heatmap or bar chart |

## 5. Proposed Story / Narrative

> "We ask: what does a pose model need to see for accurate biomechanics?
> Through attention analysis and systematic keypoint perturbation, we
> establish that (1) a small set of anatomical keypoints (pelvis, spine)
> disproportionately drives kinematic accuracy, and (2) popular ViT token
> efficiency methods (ToMe, EViT) degrade these keypoints first, because
> they occupy spatially sparse, 'background-like' image regions. This
> motivates architectures that process all tokens cheaply rather than
> selectively discarding them. We show that state space model backbones
> (VMamba, PlainMamba) achieve this: competitive biomechanical accuracy
> to ViTPose-Base at 1/3 the compute, with stronger robustness under
> limited synthetic training data. The resulting models run in real-time
> on edge GPUs, enabling deployable markerless motion capture."

The paper follows a **question-driven arc**, not just a model comparison:

1. **Analysis** (RQ1): What tokens/keypoints matter for biomechanics? Why does
   naive efficiency fail?
2. **Architecture** (RQ2): SSMs as a principled alternative — process everything
   in O(n) rather than prune at O(n^2)
3. **Data** (RQ3): SSMs transfer more efficiently from synthetic to real
4. **Systems** (RQ4-5): From theory to deployable clinical tool
5. **Understanding** (RQ6): Per-joint breakdown — where and why SSMs help

## 6. Honest Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| SSM doesn't match HRNet-48 accuracy | High | Try multiple SSM variants; hybrid SSM+CNN; accept efficiency-accuracy tradeoff if gap is small |
| Token pruning does NOT hurt biomechanics (hypothesis fails) | Medium | Still interesting finding — report it honestly; pivot story to "standard efficiency methods transfer to biomechanics" |
| Data efficiency hypothesis fails | Medium | Drop RQ3, strengthen RQ1+RQ2; the token analysis alone is a contribution |
| Existing Mamba-pose work published targeting biomechanics | Low (currently no work) | Our token analysis angle is distinct regardless of backbone choice |
| Reviewers see it as "backbone swap" | Low (mitigated) | RQ1 (token/keypoint analysis) is architecture-agnostic research; stands on its own |
| Edge deployment numbers unimpressive | Low | SSMs are inherently efficient; TensorRT should help significantly |
| ToMe/EViT integration is non-trivial for mmpose | Medium | Use HuggingFace ViTPose (already downloaded) where ToMe is plug-and-play |

## 7. What We Already Have

- [x] OpenCapBench benchmark code and evaluation pipeline
- [x] OpenCap dataset (video + ground truth motion capture)
- [x] SynthPose HRNet-48 checkpoint (763MB, mmpose format) — baseline
- [x] SynthPose ViTPose-Base checkpoint (344MB, safetensors) — baseline
- [x] SynthPose ViTPose-Huge checkpoint (2.4GB, safetensors) — baseline
- [x] yonigozlan/mmpose fork with Infinity dataset configs — training infrastructure
- [x] Full environment (mmpose, mmdet, OpenSim, CUDA)
- [ ] Mamba backbone integrated into mmpose
- [ ] Training on SynthPose synthetic data
- [ ] OpenCapBench evaluation runs
- [ ] Edge deployment benchmarks
- [ ] Paper writing

## 8. Timeline

**ECCV 2026 deadline: March 5, 2026 (~1 month from now)**

This is NOT achievable for ECCV 2026 given the experiments needed.

### Realistic targets

| Venue | Deadline | Feasibility |
|---|---|---|
| **ECCV 2026** | Mar 5, 2026 | Nearly impossible (1 month) |
| **MICCAI 2026** | ~Apr 2026 (TBD) | Tight but possible; biomechanics angle fits perfectly |
| **NeurIPS 2026** | ~May 2026 | Feasible; good venue for efficiency + scientific findings |
| **CVPR 2027** | ~Nov 2026 | Very comfortable; strongest paper possible |
| **WACV 2027** | ~Sep 2026 | Comfortable; deployment focus fits well |

### Suggested plan (targeting MICCAI or NeurIPS 2026)

**Weeks 1-2:** Reproduce baselines. Run HRNet-48 on OpenCapBench, confirm
published numbers. Set up HF ViTPose-Base inference pipeline for token analysis.

**Weeks 3-4:** Token & keypoint analysis (RQ1). Run keypoint perturbation study
(Exp 1a). Extract ViT attention maps (Exp 1b). Apply ToMe/EViT at varying rates
and measure biomechanical degradation (Exp 1c). This is the paper's analytical
foundation — do it first.

**Weeks 5-7:** Integrate VMamba-Tiny + PlainMamba into mmpose fork. Train with
full SynthPose pipeline. Run OpenCapBench evaluation. Compare against baselines
AND token-pruned ViTPose (RQ2).

**Weeks 8-9:** Data efficiency ablations (RQ3). Per-joint analysis (RQ6).

**Weeks 10-11:** Efficiency profiling (RQ4) and edge deployment benchmarks (RQ5).
ONNX export, TensorRT, Jetson.

**Weeks 12-14:** Paper writing, figures, revision.

## 9. Baselines to Reproduce

Using OpenCapBench, we need these runs as baselines:

| Model | Config | Checkpoint | Marker Set | Status |
|---|---|---|---|---|
| HRNet-48 (SynthPose) | infinity/hrnet48 config | hrnet-w48_dark.pth | Anatomical (52 kpts) | Ready |
| HRNet-48 (COCO) | coco/td-hm config | OpenMMLab URL | Coco (17 kpts) | Config available |
| ViTPose-Base (SynthPose) | Needs HF-to-mmpose conversion | model.safetensors | Anatomical (52 kpts) | Downloaded (HF format) |
| ViTPose-Huge (SynthPose) | Needs HF-to-mmpose conversion | model.safetensors | Anatomical (52 kpts) | Downloaded (HF format) |

Note: The ViTPose models are in HuggingFace Transformers format (safetensors),
not MMPose format (.pth). To use them as OpenCapBench baselines, we either need
to convert them to MMPose format or modify the benchmark pipeline to support
the HF inference API.

**Additional baselines for token efficiency analysis (RQ1c):**

| Method | Base Model | Implementation | Notes |
|---|---|---|---|
| ToMe (r=5,10,20,30,40) | ViTPose-Base (HF) | tomesd / facebookresearch/ToMe | Training-free, easy to plug in |
| EViT (top-k keep) | ViTPose-Base (HF) | Custom attention masking | Requires minor model surgery |
| SDPose-T | Custom (4.4M) | Retrain from scratch | Smallest known pose ViT baseline |
| No token reduction | ViTPose-Base (HF) | Identity | Control |

## 10. Key References

1. Gozlan et al., "OpenCapBench: A Benchmark to Bridge Pose Estimation and Biomechanics," arXiv 2406.09788, 2024.
2. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," arXiv 2312.00752, 2023.
3. Liu et al., "VMamba: Visual State Space Model," NeurIPS 2024.
4. Yang et al., "PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition," arXiv 2403.17695, 2024.
5. Li et al., "PoseMamba: Monocular 3D Human Pose Estimation with Bidirectional Global-Local SSM," AAAI 2025.
6. Xu et al., "ViTPose: Simple Vision Transformer Baselines for Human Pose Estimation," NeurIPS 2022.
7. Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation," CVPR 2019.
8. Falisse et al., "OpenCap: Human movement dynamics from smartphone videos," PLoS Comp Bio, 2023.
9. Bolya et al., "Token Merging: Your ViT But Faster," ICLR 2023.
10. Liang et al., "Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer," CVPR 2022 / EViT ICLR 2022.
11. Rao et al., "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification," NeurIPS 2021.
12. Heo et al., "SDPose: Tokenized Pose Estimation via Circulation-Guide Self-Distillation," arXiv 2024.
13. Jamil et al., "PoseSynViT: Lightweight and Scalable Vision Transformers for Human Pose Estimation," CVPR-W 2025.
