# -*- coding: utf-8 -*-
"""
Bước 5.5 — GPU Pure Silhouette Carving (SAM2 VisualHull) cho chế độ bàn xoay (--manual-turntable).
Bỏ hoàn toàn phạt màu LAB và Voxel Filling để không phá bề mặt đa màu (như Box).
Chỉ cắt bỏ các Active Voxels chiếu rơi hẳn vào vùng nền (>= bg_margin_px ngoài mask SAM2)
ở ít nhất `min_out_views` góc nhìn trong K keyframe phủ đều 360°.
"""

import os as _os
_orig_affinity = _os.sched_getaffinity(0) if hasattr(_os, "sched_getaffinity") else None
if _orig_affinity:
    try:
        _os.sched_setaffinity(0, {min(_orig_affinity)})
    except Exception:
        pass
import torch
if _orig_affinity:
    try:
        _os.sched_setaffinity(0, _orig_affinity)
    except Exception:
        pass

import gc
import time
import cv2
import numpy as np
import open3d as o3d


def select_sparse_keyframes(fuse_indices, max_keyframes: int = 24):
    """Chọn đều K keyframe phủ trọn 360°."""
    indices = list(fuse_indices)
    if len(indices) <= max_keyframes:
        return indices
    pos = np.linspace(0, len(indices) - 1, max_keyframes).round().astype(int)
    return [indices[p] for p in np.unique(pos)]


def _prepare_silhouette_tensors(
    kf_indices,
    masks,
    optimized_poses,
    device: torch.device,
    bg_margin_px: int = 3,
):
    """Tạo tensor vùng nền chắc chắn (cách biên mask SAM2 >= bg_margin_px) và T_w2c trên GPU."""
    bg_sure_list = []
    w2c_list = []

    margin = max(0, int(bg_margin_px))
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        if margin > 0
        else None
    )

    for idx in kf_indices:
        m_raw = masks[idx]
        if m_raw is None:
            continue

        mask_u8 = (m_raw > 0).astype(np.uint8)
        if kernel is not None:
            mask_dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        else:
            mask_dilated = mask_u8

        # True tại các pixel nằm hẳn ngoài nền (cách mép SAM2 >= bg_margin_px)
        bg_sure = (mask_dilated == 0)
        T_w2c = np.linalg.inv(np.asarray(optimized_poses[idx], dtype=np.float64)).astype(np.float32)

        bg_sure_list.append(bg_sure)
        w2c_list.append(T_w2c)

    if not bg_sure_list:
        return None

    return {
        "bg_sure": torch.from_numpy(np.stack(bg_sure_list, axis=0)).to(device),  # (K, H, W) bool
        "T_w2c": torch.from_numpy(np.stack(w2c_list, axis=0)).to(device),        # (K, 4, 4) float32
    }


@torch.inference_mode()
def _count_background_violations_gpu(
    points_xyz: np.ndarray,
    kf_data: dict,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    chunk_size: int = 150000,
):
    """Đếm số góc nhìn mà từng voxel chiếu rơi hẳn vào vùng nền `bg_sure`."""
    device = kf_data["T_w2c"].device
    bg_sure = kf_data["bg_sure"]   # (K, H, W) bool
    T_w2c = kf_data["T_w2c"]       # (K, 4, 4)
    K, H, W = bg_sure.shape

    R_all = T_w2c[:, :3, :3]
    t_all = T_w2c[:, :3, 3]
    kf_ids = torch.arange(K, device=device).view(K, 1)

    N = len(points_xyz)
    out_views = np.zeros(N, dtype=np.int32)
    valid_views = np.zeros(N, dtype=np.int32)

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        pts = torch.from_numpy(points_xyz[start:end].astype(np.float32)).to(device)

        pts_cam = torch.einsum("kij,nj->kni", R_all, pts) + t_all[:, None, :]
        z = pts_cam[:, :, 2]
        z_safe = torch.clamp(z, min=1e-4)
        u = torch.round(pts_cam[:, :, 0] * fx / z_safe + cx).long()
        v = torch.round(pts_cam[:, :, 1] * fy / z_safe + cy).long()

        in_bounds = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u_c = torch.clamp(u, 0, W - 1)
        v_c = torch.clamp(v, 0, H - 1)

        is_bg = bg_sure[kf_ids, v_c, u_c] & in_bounds

        out_views[start:end] = is_bg.sum(dim=0).cpu().numpy().astype(np.int32)
        valid_views[start:end] = in_bounds.sum(dim=0).cpu().numpy().astype(np.int32)

    return out_views, valid_views


def carve_tsdf_by_silhouette(
    vbg,
    fuse_indices,
    masks,
    optimized_poses,
    intrinsic,
    bg_margin_px: int = 3,
    min_out_views: int = 3,
    max_keyframes: int = 24,
):
    """Gọt sạch các voxel thừa nằm ngoài hình bóng SAM2 (>= bg_margin_px vào nền)
    ở ít nhất `min_out_views` góc nhìn, giữ nguyên toàn bộ lõi bề mặt thật.
    """
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf_indices = select_sparse_keyframes(fuse_indices, max_keyframes=max_keyframes)
    kf_data = _prepare_silhouette_tensors(
        kf_indices, masks, optimized_poses,
        device=device, bg_margin_px=bg_margin_px,
    )
    if kf_data is None:
        return {"skipped": True, "reason": "no_valid_keyframes"}

    K_mat = np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64)
    fx, fy, cx, cy = float(K_mat[0, 0]), float(K_mat[1, 1]), float(K_mat[0, 2]), float(K_mat[1, 2])

    buf_indices_i32 = vbg.hashmap().active_buf_indices().to(o3d.core.Dtype.Int32)
    n_blocks = int(buf_indices_i32.shape[0])
    if n_blocks == 0:
        return {"skipped": True, "reason": "empty_voxel_grid"}

    buf_indices_i64 = buf_indices_i32.to(o3d.core.Dtype.Int64)
    tsdf_attr = vbg.attribute("tsdf")
    weight_attr = vbg.attribute("weight")
    tsdf_np = tsdf_attr[buf_indices_i64].cpu().numpy()      # (M, R, R, R, 1)
    weight_np = weight_attr[buf_indices_i64].cpu().numpy()  # (M, R, R, R, 1)

    R = int(tsdf_np.shape[1])
    coords_flat, _ = vbg.voxel_coordinates_and_flattened_indices(buf_indices_i32)
    coords_4d = coords_flat.cpu().numpy().reshape(n_blocks, R, R, R, 3)

    tsdf_4d = tsdf_np[..., 0]
    weight_4d = weight_np[..., 0]

    # Chỉ xét các voxel đang nằm BÊN TRONG hoặc SÁT BỀ MẶT (tsdf < 0.25)
    b_ids, i1, i2, i3 = np.nonzero((weight_4d > 0.0) & (tsdf_4d < 0.25))
    active_count = int(len(b_ids))
    clipped_count = 0

    if active_count > 0:
        pts_active = coords_4d[b_ids, i1, i2, i3]
        out_views, valid_views = _count_background_violations_gpu(
            pts_active, kf_data, fx, fy, cx, cy
        )
        drop = (valid_views >= int(min_out_views)) & (out_views >= int(min_out_views))
        clipped_count = int(np.count_nonzero(drop))
        if clipped_count > 0:
            db, d1, d2, d3 = b_ids[drop], i1[drop], i2[drop], i3[drop]
            # Đổi sang không gian trống (+1.0) nhưng GIỮ NGUYÊN weight > 0 để không thủng Marching Cubes
            tsdf_4d[db, d1, d2, d3] = 1.0
            weight_4d[db, d1, d2, d3] = np.maximum(weight_4d[db, d1, d2, d3], 1.0)

            vbg_dev = tsdf_attr.device
            tsdf_attr[buf_indices_i64] = o3d.core.Tensor(tsdf_np, dtype=tsdf_attr.dtype, device=vbg_dev)
            weight_attr[buf_indices_i64] = o3d.core.Tensor(weight_np, dtype=weight_attr.dtype, device=vbg_dev)

    del kf_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "mode": "pure_silhouette_carving",
        "device": str(device),
        "block_resolution": R,
        "keyframes": len(kf_indices),
        "bg_margin_px": int(bg_margin_px),
        "min_out_views": int(min_out_views),
        "active_voxels": active_count,
        "clipped_voxels": clipped_count,
        "clipped_pct": round(100.0 * clipped_count / max(active_count, 1), 2),
        "time_s": round(time.time() - t0, 2),
    }
