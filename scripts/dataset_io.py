import os
import sys
import time
import json
import math
import shutil
import subprocess
import numpy as np
import cv2
import open3d as o3d


def _resolve_compute_device():
    try:
        if o3d.core.cuda.device_count() > 0:
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return o3d.core.Device("CPU:0")

O3D_DEVICE_GPU = _resolve_compute_device()
O3D_DEVICE_CPU = o3d.core.Device("CPU:0")

def load_intrinsics(dataset_dir):
    path = os.path.join(dataset_dir, "intrinsics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
        w  = int(c.get("width",  640)); h  = int(c.get("height", 480))
        fx = float(c.get("fx", 615.0)); fy = float(c.get("fy", 615.0))
        cx = float(c.get("cx", 320.0)); cy = float(c.get("cy", 240.0))
        print(f"   Intrinsics: {w}×{h} fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    else:
        w, h, fx, fy, cx, cy = 640, 480, 615.0, 615.0, 320.0, 240.0
        print("   [WARN] Dùng intrinsics mặc định.")
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    intrinsic_tensor = o3d.core.Tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
    return intrinsic, intrinsic_tensor

def _resolve_cam_key(fn, poses_data):
    stem = os.path.splitext(fn)[0]
    try:
        idx = int(stem)
        candidates = [f"cam_{idx:03d}", f"cam_{idx}", f"cam_{stem[-3:]}"]
    except ValueError:
        candidates = [f"cam_{stem}", f"cam_{stem[-3:]}"]
    for key in candidates:
        if key in poses_data:
            return key
    return None

def load_dataset(dataset_dir, start_frame=0, max_frames=1000, end_frame=0):
    color_dir     = os.path.join(dataset_dir, "rgb")
    depth_raw_dir = os.path.join(dataset_dir, "depth_raw")
    with open(os.path.join(dataset_dir, "poses.json"), encoding="utf-8") as f:
        poses_data = json.load(f)

    color_files = sorted(f for f in os.listdir(color_dir) if f.endswith(".png"))

    # Apply end offset (cut off corrupted/drifted tail frames)
    if end_frame > 0:
        color_files = color_files[:end_frame]
    
    # Apply start offset
    if start_frame > 0:
        color_files = color_files[start_frame:]
    
    # Lấy mẫu đều trên toàn bộ quỹ đạo 360° thay vì cắt cụt phần đuôi
    if max_frames > 0 and len(color_files) > max_frames:
        sample_idx = np.linspace(0, len(color_files) - 1, max_frames).round().astype(int)
        color_files = [color_files[i] for i in np.unique(sample_idx)]

    rgbd_images, color_np, depth_np, poses = [], [], [], []
    loaded_filenames = []

    for fn in color_files:
        cam_key = _resolve_cam_key(fn, poses_data)
        if cam_key is None:
            continue
        cf = os.path.join(color_dir, fn)
        df = os.path.join(depth_raw_dir, fn)
        if not os.path.exists(df):
            continue

        ci   = o3d.io.read_image(cf)
        di   = o3d.io.read_image(df)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            ci, di, depth_scale=1000.0, depth_trunc=2.5, convert_rgb_to_intensity=False)

        rgbd_images.append(rgbd)
        poses.append(np.array(poses_data[cam_key], dtype=np.float64).reshape(4, 4))
        color_np.append(cv2.imread(cf))
        depth_np.append(cv2.imread(df, cv2.IMREAD_UNCHANGED))
        loaded_filenames.append(fn)

    return rgbd_images, color_np, depth_np, poses, loaded_filenames