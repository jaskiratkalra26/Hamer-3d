# 3D Hand Tracking Integration Package

A modular, high-speed 3D Hand Mesh Reconstruction module built on **HaMeR**, **ViTPose**, and **RegNetY/ViTDet**. Designed to be easily integrated into any third-party Python application, ROS node, or video processing pipeline.

---

## 🚀 Quick Setup Instructions for Any Project / New VM

### 1. Run 1-Click Installer
```bash
# Run the automated dependency & model weights installer
bash hand_tracker/install.sh
```

### 2. Activate Environment
```bash
source .venv_hand_tracker/bin/activate
```

### 3. Run Example Demo
```bash
python hand_tracker/example_usage.py
```

---

## 💻 Python Integration Guide

Import `HandTracker3D` directly into your external Python code:

```python
import cv2
from hand_tracker import HandTracker3D

# 1. Initialize Tracker once (runs on GPU with FP16 + fast RegNetY detector + resolution scaling)
tracker = HandTracker3D(
    body_detector='regnety',   # 'regnety' (fast) or 'vitdet' (accurate)
    scale_inference=0.5,       # 540p downscaled detection scanning
    det_stride=5               # Re-use bounding boxes across 5 frames
)

# 2. Process any OpenCV BGR image frame in memory
results = tracker.process_frame(frame_bgr)

# 3. Access 3D Hand Mesh Data & Camera Translation
for hand in results['hands']:
    is_right = hand['is_right']      # 1 (Right hand) or 0 (Left hand)
    verts_3d = hand['verts_3d']      # (778, 3) 3D mesh vertices in camera space
    cam_t    = hand['cam_t']         # [tx, ty, tz] 3D camera translation vector

# 4. Optional: Render 3D Wireframe Overlay
rendered_frame = tracker.render_overlay(frame_bgr, results)
```

---

## 🛠 Features & Performance Optimizations Included
- **High-Speed GPU Acceleration**: Integrated FP16 `torch.amp.autocast` + SIMD vectorized wireframe rendering.
- **Detector & Pose Striding**: Re-uses body bounding boxes across N frames while computing 3D hand mesh on 100% of frames.
- **Fast SIMD Preprocessing**: Replaced slow CPU convolutions with `cv2.GaussianBlur` for zero CPU bottlenecking.
- **Headless GPU Rendering**: Configured with automated OSMesa/EGL fallbacks for Linux server/cloud GPU VMs.
