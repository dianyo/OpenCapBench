import os

# Configuration mode: "local" or "windows" or "sherlock"
# Change this to "local" for your setup
config_global = "local"  # Default changed to "local" for standard setup

config_base_local = {
    # MMPose Directory - If using system-wide install via pip/uv, leave as empty string
    # If using cloned fork, set to absolute path: "/path/to/mmpose"
    "mmposeDirectory": "",  # Leave empty for system-wide mmpose install
    
    "OutputBoxDirectory": "OutputBox",
    
    # Person Detector Configuration (auto-downloads from URL)
    "model_config_person": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    
    # Alternative person detector (ConvNeXt - better performance):
    # "model_config_person": "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
    # "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
    
    # Pose Model Configuration
    # TODO: Update these paths based on your model choice:
    # Option 1: SynthPose VitPose (recommended for paper results)
    # Download from: https://huggingface.co/yonigozlan/synthpose-vitpose-base-hf
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py",
    "model_ckpt_pose": "/home/ubuntu/joe/OpenCapBench/models/synthpose-vitpose-base/pytorch_model.pth",
    
    # Option 2: Standard COCO-trained models (for comparison):
    # "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py",
    # "model_ckpt_pose": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-e881a4b6_20210203.pth",
    
    # Data Directory - Where you downloaded the OpenCap dataset
    "dataDir": "/home/ubuntu/joe/OpenCapBench/data",
    
    # Batch sizes (adjust based on your GPU memory)
    "batch_size_det": 4,
    "batch_size_pose": 32,
    
    # Scaling configuration
    "useGTscaling": True,  # Use ground truth scaling from OpenCap data
    
    # Marker set: "Anatomical" for SynthPose models, "Coco" for standard COCO models
    "marker_set": "Anatomical",
    
    # Alternative models (advanced usage)
    "alt_model": None,  # None or "VirtualMarker" or "CLIFF"
}

# Set absolute path for pose model checkpoint
if config_base_local["mmposeDirectory"]:
    # If using cloned mmpose repo
    config_base_local["model_ckpt_pose_absolute"] = os.path.join(
        config_base_local["mmposeDirectory"], config_base_local["model_ckpt_pose"]
    )
else:
    # If using system-wide install or absolute path already provided
    if config_base_local["model_ckpt_pose"].startswith("http"):
        # URL - will be auto-downloaded
        config_base_local["model_ckpt_pose_absolute"] = config_base_local["model_ckpt_pose"]
    else:
        # Assume it's already an absolute path
        config_base_local["model_ckpt_pose_absolute"] = os.path.abspath(
            config_base_local["model_ckpt_pose"]
        )

config_base_windows = {
    "mmposeDirectory": "C:/Data/OpenCap/mmpose",
    "OutputBoxDirectory": "OutputBox",
    "model_config_person": "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    # "model_config_person" : "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
    # "model_ckpt_person" :"https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
    # "model_config_pose" : "configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py",
    # "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288.py",
    # "model_ckpt_pose" : "pretrain/coco/td-hm_hrnet-w48_dark-8xb32-210e_coco-384x288-39c3c381_20220916.pth",
    # "model_ckpt_pose" : "pretrain/coco/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth",
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/hrnet48/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_3DPW_eval_rich-384x288_pretrained.py",
    "model_ckpt_pose": "C:/Data/OpenCapBench/data/models/epoch_30.pth",
    "dataDir": "C:/Data/OpenCapBench/data",
    "batch_size_det": 4,
    "batch_size_pose": 32,
    "useGTscaling": True,
    "marker_set": "Anatomical",  # "Coco" or "Anatomical"
    "alt_model": None,  # None or "VirtualMarker" or "CLIFF"
}



config_base_sherlock = {
    "mmposeDirectory": "/home/users/yonigoz/RA/mmpose",
    "OutputBoxDirectory": "OutputBox",
    "model_config_person": "demo/mmdetection_cfg/configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py",
    "model_ckpt_person": "https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth",
    # "model_config_person" : "demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py",
    # "model_ckpt_person" : "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    "model_config_pose": "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_ViTPose-huge_8xb64-210e_merge_bedlam_infinity_eval_bedlam-256x192.py",
    "model_ckpt_pose": "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_eval_bedlam/ViT/huge/best_infinity_AP_epoch_10.pth",
    # "model_config_pose" : "configs/body_2d_keypoint/topdown_heatmap/infinity/td-hm_hrnet-w48_dark-8xb32-210e_merge_bedlam_infinity_coco_eval_bedlam-384x288_pretrained.py",
    # "model_ckpt_pose" : "/scratch/users/yonigoz/mmpose_data/work_dirs/merge_bedlam_infinity_coco_eval_bedlam/HRNet/w48_dark_pretrained/best_infinity_AP_epoch_18.pth",
    "dataDir": "/scratch/users/yonigoz/OpenCap_data",
    "batch_size_det": 16,
    "batch_size_pose": 2,
    "useGTscaling": True,
    "marker_set": "Anatomical",
    "alt_model": None,
}
config_base_sherlock["model_ckpt_pose_absolute"] = config_base_sherlock[
    "model_ckpt_pose"
]

config = {}
if config_global == "local":
    config = config_base_local
if config_global == "sherlock":
    config = config_base_sherlock
if config_global == "windows":
    config = config_base_windows


def getMMposeAnatomicalCocoMarkerNames():
    marker_names = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "sternum",
        "rshoulder",
        "lshoulder",
        "r_lelbow",
        "l_lelbow",
        "r_melbow",
        "l_melbow",
        "r_lwrist",
        "l_lwrist",
        "r_mwrist",
        "l_mwrist",
        "r_ASIS",
        "l_ASIS",
        "r_PSIS",
        "l_PSIS",
        "r_knee",
        "l_knee",
        "r_mknee",
        "l_mknee",
        "r_ankle",
        "l_ankle",
        "r_mankle",
        "l_mankle",
        "r_5meta",
        "l_5meta",
        "r_toe",
        "l_toe",
        "r_big_toe",
        "l_big_toe",
        "l_calc",
        "r_calc",
        "C7",
        "L2",
        "T11",
        "T6",
    ]

    return marker_names


def getMMposeMarkerNames():
    markerNames = [
        "Nose",
        "LEye",
        "REye",
        "LEar",
        "REar",
        "LShoulder",
        "RShoulder",
        "LElbow",
        "RElbow",
        "LWrist",
        "RWrist",
        "LHip",
        "RHip",
        "LKnee",
        "RKnee",
        "LAnkle",
        "RAnkle",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]

    return markerNames


def getOpenPoseMarkerNames():
    markerNames = [
        "Nose",
        "Neck",
        "RShoulder",
        "RElbow",
        "RWrist",
        "LShoulder",
        "LElbow",
        "LWrist",
        "midHip",
        "RHip",
        "RKnee",
        "RAnkle",
        "LHip",
        "LKnee",
        "LAnkle",
        "REye",
        "LEye",
        "REar",
        "LEar",
        "LBigToe",
        "LSmallToe",
        "LHeel",
        "RBigToe",
        "RSmallToe",
        "RHeel",
    ]

    return markerNames


def getMMposeAnatomicalCocoMarkerPairs():
    markerNames = {
        "left_eye": "right_eye",
        "left_ear": "right_ear",
        "left_shoulder": "right_shoulder",
        "left_elbow": "right_elbow",
        "left_wrist": "right_wrist",
        "left_hip": "right_hip",
        "left_knee": "right_knee",
        "left_ankle": "right_ankle",
        "rshoulder": "lshoulder",
        "lshoulder": "rshoulder",
        "r_lelbow": "l_lelbow",
        "l_lelbow": "r_lelbow",
        "r_melbow": "l_melbow",
        "l_melbow": "r_melbow",
        "r_lwrist": "l_lwrist",
        "l_lwrist": "r_lwrist",
        "r_mwrist": "l_mwrist",
        "l_mwrist": "r_mwrist",
        "r_ASIS": "l_ASIS",
        "l_ASIS": "r_ASIS",
        "r_PSIS": "l_PSIS",
        "l_PSIS": "r_PSIS",
        "r_knee": "l_knee",
        "l_knee": "r_knee",
        "r_mknee": "l_mknee",
        "l_mknee": "r_mknee",
        "r_ankle": "l_ankle",
        "l_ankle": "r_ankle",
        "r_mankle": "l_mankle",
        "l_mankle": "r_mankle",
        "r_5meta": "l_5meta",
        "l_5meta": "r_5meta",
        "r_toe": "l_toe",
        "l_toe": "r_toe",
        "r_big_toe": "l_big_toe",
        "l_big_toe": "r_big_toe",
        "l_calc": "r_calc",
        "r_calc": "l_calc",
    }

    return markerNames
