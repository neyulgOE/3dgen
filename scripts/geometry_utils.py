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
from scipy.spatial import cKDTree


def _resolve_compute_device():
    try:
        if o3d.core.cuda.device_count() > 0:
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return o3d.core.Device("CPU:0")


O3D_DEVICE_GPU = _resolve_compute_device()
O3D_DEVICE_CPU = o3d.core.Device("CPU:0")

def rotation_angle_deg(transform):
    """Return the angle of a rigid transform's rotation in degrees."""
    trace = np.trace(transform[:3, :3])
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))

def pose_is_plausible(transform, max_translation_m, max_rotation_deg):
    translation_m = float(np.linalg.norm(transform[:3, 3]))
    rotation_deg = rotation_angle_deg(transform)
    return translation_m <= max_translation_m and rotation_deg <= max_rotation_deg, translation_m, rotation_deg

def inverse_consistency(transform_forward, transform_backward):
    cycle = transform_forward @ transform_backward
    return float(np.linalg.norm(cycle[:3, 3])), rotation_angle_deg(cycle)

def frame_point_cloud(index, color_np, depth_np, masks, intrinsic_tensor, object_only=True):
    """Create either a target-only cloud, a background cloud, or a near-scene handheld cloud."""
    depth_m = depth_np[index].copy().astype(np.float32)
    mask = masks[index]
    
    if object_only is True:
        depth_m[mask == 0] = 0
        depth_max = 2.0
    elif object_only is False:
        if mask is not None:
            depth_m[mask > 0] = 0
        depth_max = 1.5
    else:
        # Handheld SLAM: cắt bỏ nền xa > 1.2m (nơi nhiễu stereo D435f lên tới 30-80mm
        # làm kéo lệch ICP và làm phình RMSE của View Graph).
        depth_max = 1.2
    
    color_t = o3d.core.Tensor(cv2.cvtColor(color_np[index], cv2.COLOR_BGR2RGB), o3d.core.Dtype.UInt8, O3D_DEVICE_GPU)
    depth_t = o3d.core.Tensor(depth_m.astype(np.uint16), o3d.core.Dtype.UInt16, O3D_DEVICE_GPU)
    t_color = o3d.t.geometry.Image(color_t)
    t_depth = o3d.t.geometry.Image(depth_t)
    rgbd = o3d.t.geometry.RGBDImage(t_color, t_depth)
    
    return o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic_tensor, depth_scale=1000.0, depth_max=depth_max)

def background_rgbd_image(index, color_np, depth_np, masks):
    depth = depth_np[index].copy()
    color = color_np[index].copy()
    mask = masks[index]
    if mask is not None:
        exclusion = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
        depth[exclusion > 0] = 0
        color[exclusion > 0] = 0
        
    color_t = o3d.core.Tensor(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), o3d.core.Dtype.UInt8, O3D_DEVICE_GPU)
    depth_t = o3d.core.Tensor(depth.astype(np.uint16), o3d.core.Dtype.UInt16, O3D_DEVICE_GPU)
    t_color = o3d.t.geometry.Image(color_t)
    t_depth = o3d.t.geometry.Image(depth_t)
    return o3d.t.geometry.RGBDImage(t_color, t_depth)

def full_rgbd_image(index, color_np, depth_np, masks):
    depth = depth_np[index].copy()
    color = color_np[index].copy()
    # Loại bỏ các điểm nền quá xa (> 1200mm) để tránh nhiễu nền làm trôi RGB-D odometry
    depth[depth > 1200] = 0
    
    color_t = o3d.core.Tensor(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), o3d.core.Dtype.UInt8, O3D_DEVICE_GPU)
    depth_t = o3d.core.Tensor(depth.astype(np.uint16), o3d.core.Dtype.UInt16, O3D_DEVICE_GPU)
    t_color = o3d.t.geometry.Image(color_t)
    t_depth = o3d.t.geometry.Image(depth_t)
    return o3d.t.geometry.RGBDImage(t_color, t_depth)

def estimate_turntable_camera_poses(indices, rgbd_images, color_np, depth_np, masks,
                                    intrinsic_tensor, init_poses):
    poses = [np.eye(4) for _ in range(len(rgbd_images))]
    if not indices:
        return poses, [], []
        
    background = {index: background_rgbd_image(index, color_np, depth_np, masks)
                  for index in indices}
    anchor = indices[0]
    accepted = [anchor]
    events = [{"frame": anchor, "accepted": True, "reason": "initial_background_anchor"}]
    
    for index in indices[1:]:
        initial = np.linalg.inv(init_poses[anchor]) @ init_poses[index]
        initial_tensor = o3d.core.Tensor(initial, o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
        try:
            forward = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                background[index], background[anchor],
                intrinsic_tensor, initial_tensor,
                depth_scale=1000.0, depth_max=1.5)
        except RuntimeError as e:
            if "Singular 6x6 linear system" in str(e):
                forward = type('obj', (object,), {'fitness': 0.0, 'transformation': initial_tensor})()
            else:
                raise e
            
        if forward.fitness == 0.0:
            events.append({"frame": index, "accepted": False, "reason": "background_odometry_failed"})
            continue
            
        forward_mat = forward.transformation.cpu().numpy()
        plausible, translation, rotation = pose_is_plausible(forward_mat, 0.10, 20.0)
        
        info_f_tensor = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
            background[index].depth, background[anchor].depth,
            intrinsic_tensor, forward.transformation,
            dist_threshold=0.05, depth_scale=1000.0, depth_max=1.5)
        info_f = info_f_tensor.cpu().numpy()
        
        information = float(np.trace(info_f))
        if not plausible or information < 2e4:
            events.append({"frame": index, "accepted": False, "reason": "background_quality_gate",
                           "translation_mm": round(translation * 1000.0, 2),
                           "rotation_deg": round(rotation, 2), "information": round(information, 1)})
            continue
        poses[index] = poses[anchor] @ forward_mat
        accepted.append(index)
        events.append({"frame": index, "accepted": True,
                       "translation_mm": round(translation * 1000.0, 2),
                       "rotation_deg": round(rotation, 2), "information": round(information, 1)})
        anchor = index
    return poses, accepted, events

def estimate_turntable_pivots(indices, depth_np, masks, intrinsic):
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    estimates = []
    per_frame = {}
    for index in indices:
        mask = masks[index]
        if mask is None:
            continue
        ys, xs = np.nonzero(mask > 0)
        if len(xs) < 500:
            continue
        depth = depth_np[index][ys, xs].astype(np.float32)
        good = depth > 50
        if np.count_nonzero(good) < 500:
            continue
        xs = xs[good].astype(np.float32)
        ys = ys[good].astype(np.float32)
        depth = depth[good]

        x_lo, x_hi = np.percentile(xs, [15, 85])
        y_lo, y_hi = np.percentile(ys, [25, 75])
        if x_hi <= x_lo or y_hi <= y_lo:
            continue

        front_m = float(np.percentile(depth, 8)) / 1000.0
        center_x_px = float((x_lo + x_hi) * 0.5)
        center_y_px = float((y_lo + y_hi) * 0.5)
        body_width_m = ((x_hi - x_lo) / 0.70) * front_m / fx
        radius_m = float(np.clip(body_width_m * 0.5, 0.015, 0.080))
        axis_depth_m = front_m + radius_m
        estimate = np.array([
            (center_x_px - cx) * axis_depth_m / fx,
            (center_y_px - cy) * axis_depth_m / fy,
            axis_depth_m,
            radius_m,
        ], dtype=np.float64)
        estimates.append(estimate)
        per_frame[index] = estimate[:3]

    if len(estimates) < 3:
        return None, per_frame, {"reason": "too_few_pivot_estimates", "count": len(estimates)}
    estimates = np.asarray(estimates, dtype=np.float64)
    pivot = np.median(estimates[:, :3], axis=0)
    radius = float(np.median(estimates[:, 3]))
    drift = np.linalg.norm(estimates[:, :3] - pivot, axis=1)
    return pivot, per_frame, {"reason": "mask_silhouette_depth", "count": int(len(estimates)),
                              "radius_m": radius,
                              "median_centre_drift_mm": round(float(np.median(drift)) * 1000.0, 2),
                              "p95_centre_drift_mm": round(float(np.percentile(drift, 95)) * 1000.0, 2)}

def estimate_turntable_pivot(indices, depth_np, masks, intrinsic):
    pivot, _, diagnostics = estimate_turntable_pivots(indices, depth_np, masks, intrinsic)
    return pivot, diagnostics

def turntable_pose(pivot, axis, angle_deg):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        raise ValueError("turntable axis must be non-zero")
    x, y, z = axis / norm
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),     y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation @ pivot
    return transform

def turntable_pose_between(source_pivot, target_pivot, axis, angle_deg):
    transform = turntable_pose(np.zeros(3), axis, angle_deg)
    transform[:3, 3] = (np.asarray(target_pivot, dtype=np.float64)
                        - transform[:3, :3] @ np.asarray(source_pivot, dtype=np.float64))
    return transform

def turntable_feature_cloud(cloud, pivot, axis):
    points = np.asarray(cloud.points)
    if len(points) < 100:
        return o3d.geometry.PointCloud()
    delta = points - pivot
    along_axis = np.outer(delta @ axis, axis)
    radial_distance = np.linalg.norm(delta - along_axis, axis=1)
    threshold = np.percentile(radial_distance, 82)
    selected = np.flatnonzero(radial_distance >= threshold)
    if len(selected) < 35:
        return o3d.geometry.PointCloud()
    featured = o3d.geometry.PointCloud()
    featured.points = o3d.utility.Vector3dVector(points[selected])
    if cloud.has_colors():
        featured.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[selected])
    if cloud.has_normals():
        featured.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[selected])
    return featured

def estimate_turntable_yaw(source, reference, feature_source, feature_reference,
                           pivot, axis, previous_yaw, args):
    if len(source.points) < 100 or len(reference.points) < 100:
        return None, {"reason": "too_few_object_points"}
    start = previous_yaw - args.turntable_max_step_deg
    stop = previous_yaw + args.turntable_max_step_deg
    angles = np.arange(start, stop + args.turntable_yaw_step_deg * 0.5,
                       args.turntable_yaw_step_deg)
    best = None
    for angle in angles:
        pose = turntable_pose(pivot, axis, float(angle))
        candidate = o3d.geometry.PointCloud(source)
        candidate.transform(pose)
        body_score = o3d.pipelines.registration.evaluate_registration(
            candidate, reference, args.turntable_max_correspondence)
        feature_score = None
        if len(feature_source.points) >= 35 and len(feature_reference.points) >= 35:
            feature_candidate = o3d.geometry.PointCloud(feature_source)
            feature_candidate.transform(pose)
            feature_score = o3d.pipelines.registration.evaluate_registration(
                feature_candidate, feature_reference, args.turntable_max_correspondence)
        feature_fitness = float(feature_score.fitness) if feature_score is not None else 0.0
        feature_rmse = float(feature_score.inlier_rmse) if feature_score is not None else np.inf
        key = (feature_fitness, float(body_score.fitness), -feature_rmse,
               -float(body_score.inlier_rmse))
        if best is None or key > best[0]:
            best = (key, float(angle), pose, body_score, feature_score)
    if best is None:
        return None, {"reason": "no_yaw_candidates"}
    _, angle, pose, body_score, feature_score = best
    diagnostics = {"yaw_deg": round(angle, 3),
                   "fitness": round(float(body_score.fitness), 4),
                   "rmse_mm": round(float(body_score.inlier_rmse) * 1000.0, 3),
                   "feature_fitness": round(float(feature_score.fitness), 4) if feature_score is not None else None,
                   "feature_rmse_mm": round(float(feature_score.inlier_rmse) * 1000.0, 3) if feature_score is not None else None}
    accepted = (body_score.fitness >= args.turntable_min_fitness and
                body_score.inlier_rmse <= args.turntable_max_rmse and
                (feature_score is None or feature_score.fitness >= args.turntable_min_feature_fitness))
    diagnostics["reason"] = "verified" if accepted else "quality_gate_failed"
    return (pose if accepted else None), diagnostics

def estimate_turntable_axis_from_table(indices, depth_np, masks, intrinsic, pivot=None):
    """Ước lượng vector trục quay vuông góc mặt bàn bằng RANSAC trên vùng nền ngoài mask.
    Trả về (axis_unit_vec, diagnostics). Fallback về (0, 1, 0) nếu không tìm được mặt phẳng bàn hợp lệ.
    """
    default_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if not indices:
        return default_axis, {"reason": "no_indices", "fallback": True}

    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()

    # Lấy tối đa 6 frame trải đều để gộp nền tĩnh, giảm nhiễu depth D435f
    sample_pos = np.linspace(0, len(indices) - 1, min(6, len(indices))).round().astype(int)
    sample_indices = [indices[p] for p in np.unique(sample_pos)]

    # Giới hạn dải độ sâu quanh vật thể để chỉ bắt mặt bàn đỡ, tránh tường/nền xa
    if pivot is not None:
        z_min_mm = max(120.0, (float(pivot[2]) - 0.25) * 1000.0)
        z_max_mm = min(1500.0, (float(pivot[2]) + 0.35) * 1000.0)
    else:
        z_min_mm, z_max_mm = 150.0, 1200.0

    dilate_kernel = np.ones((25, 25), np.uint8)
    bg_pts_list = []

    for idx in sample_indices:
        depth = depth_np[idx].astype(np.float32)
        H_img = depth.shape[0]
        valid = (depth >= z_min_mm) & (depth <= z_max_mm)
        valid &= (np.arange(H_img)[:, None] > int(H_img * 0.55))
        mask = masks[idx]
        if mask is not None:
            excl = cv2.dilate((mask > 0).astype(np.uint8), dilate_kernel, iterations=1)
            valid &= (excl == 0)

        ys, xs = np.nonzero(valid)
        if len(xs) < 200:
            continue
        # Lấy mẫu thưa để RANSAC chạy nhanh (< 50ms)
        if len(xs) > 4000:
            step = len(xs) // 4000
            xs, ys = xs[::step], ys[::step]

        zs = depth[ys, xs] / 1000.0
        pts = np.stack([(xs - cx) * zs / fx, (ys - cy) * zs / fy, zs], axis=1)
        bg_pts_list.append(pts)

    if not bg_pts_list:
        return default_axis, {"reason": "no_background_points", "fallback": True}

    all_bg = np.vstack(bg_pts_list)
    if len(all_bg) < 500:
        return default_axis, {"reason": "too_few_background_points", "count": len(all_bg), "fallback": True}

    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(all_bg)
    pcd_bg = pcd_bg.voxel_down_sample(0.005)
    if len(pcd_bg.points) < 300:
        return default_axis, {"reason": "too_few_downsampled_bg", "fallback": True}

    try:
        plane_model, inliers = pcd_bg.segment_plane(
            distance_threshold=0.005, ransac_n=3, num_iterations=1000
        )
    except Exception as e:
        return default_axis, {"reason": f"ransac_error: {e}", "fallback": True}

    inlier_ratio = float(len(inliers)) / max(1, len(pcd_bg.points))
    n = np.array(plane_model[:3], dtype=np.float64)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-8 or len(inliers) < 500 or inlier_ratio < 0.15:
        return default_axis, {
            "reason": "weak_table_plane",
            "inliers": len(inliers),
            "inlier_ratio": round(inlier_ratio, 3),
            "fallback": True,
        }
    n /= n_norm

    # Quy ước hệ tọa độ camera OpenCV: +Y hướng xuống (khớp mặc định (0, 1, 0))
    if n[1] < 0:
        n = -n

    # Chốt an toàn: chỉ nhận mặt phẳng nghiêng tối đa 45° so với [0, 1, 0] để tránh bắt nhầm tường
    if n[1] < 0.7071:
        return default_axis, {
            "reason": "plane_not_horizontal_table",
            "normal": [round(float(v), 4) for v in n],
            "fallback": True,
        }

    tilt_deg = float(np.degrees(np.arccos(np.clip(n[1], -1.0, 1.0))))
    return n, {
        "reason": "ransac_table_plane",
        "axis": [round(float(v), 4) for v in n],
        "camera_tilt_deg": round(tilt_deg, 2),
        "inliers": int(len(inliers)),
        "inlier_ratio": round(inlier_ratio, 3),
        "fallback": False,
    }


def _score_turntable_alignment(src_pts, src_cols, ref_tree, ref_cols, pose, max_dist=0.007):
    """Chấm điểm khớp giữa đám mây nguồn (sau biến đổi pose) và đám mây tham chiếu.
    Kết hợp cả Hình học (fitness, RMSE) và Màu sắc RGB để phá tính đối xứng trụ tròn.
    """
    R_mat = pose[:3, :3]
    t_vec = pose[:3, 3]
    cand_pts = (src_pts @ R_mat.T) + t_vec

    dists, nn_idx = ref_tree.query(cand_pts, k=1, distance_upper_bound=max_dist)
    valid = np.isfinite(dists)
    n_inliers = int(np.count_nonzero(valid))
    if n_inliers < 20:
        return 0.0, max_dist, 1.0

    fitness = float(n_inliers) / float(len(src_pts))
    rmse = float(np.sqrt(np.mean(dists[valid] ** 2)))

    if src_cols is not None and ref_cols is not None:
        col_diff = np.linalg.norm(src_cols[valid] - ref_cols[nn_idx[valid]], axis=1)
        color_err = float(np.mean(col_diff))
    else:
        color_err = 0.0

    return fitness, rmse, color_err


def detect_turntable_direction(valid_candidates, color_np, depth_np, masks,
                               intrinsic_tensor, pivot, axis, turn_start, timeline_span, args):
    """Tự động dò chiều quay bàn xoay (+1.0 hoặc -1.0) trên cung đầu tiên (~60-70°)."""
    n_probe = min(24, len(valid_candidates))
    if n_probe < 3:
        return -1.0, {"reason": "too_few_frames_default_neg"}

    anchor_cloud = frame_point_cloud(
        valid_candidates[0], color_np, depth_np, masks, intrinsic_tensor
    ).voxel_down_sample(args.turntable_voxel).cpu().to_legacy()
    ref_pts = np.asarray(anchor_cloud.points)
    if len(ref_pts) < 100:
        return -1.0, {"reason": "sparse_anchor_default_neg"}

    ref_cols = np.asarray(anchor_cloud.colors) if anchor_cloud.has_colors() else None
    ref_tree = cKDTree(ref_pts)
    max_dist = min(0.008, float(args.turntable_max_correspondence))

    probe_indices = valid_candidates[1:n_probe:2]
    scores = {}
    for sign in (-1.0, 1.0):
        total = 0.0
        for idx in probe_indices:
            angle = sign * 360.0 * (idx - turn_start) / max(1, timeline_span)
            pcd = frame_point_cloud(
                idx, color_np, depth_np, masks, intrinsic_tensor
            ).voxel_down_sample(args.turntable_voxel).cpu().to_legacy()
            pts = np.asarray(pcd.points)
            if len(pts) < 100:
                continue
            cols = np.asarray(pcd.colors) if pcd.has_colors() else None
            pose = turntable_pose(pivot, axis, angle)
            fit, rmse, col_err = _score_turntable_alignment(pts, cols, ref_tree, ref_cols, pose, max_dist)
            total += fit - 25.0 * rmse - 0.65 * col_err
        scores[sign] = round(total, 4)

    best_sign = 1.0 if scores[1.0] > scores[-1.0] else -1.0
    return best_sign, {"scores": {"+1": scores[1.0], "-1": scores[-1.0]}, "chosen_sign": best_sign}


def _mask_bbox_padded(mask, pad_ratio=0.12):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 100:
        return None
    h, w = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    px = int((x1 - x0) * pad_ratio) + 8
    py = int((y1 - y0) * pad_ratio) + 8
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def _extract_sift_in_mask(color_img, mask, bbox, sift):
    """Trích đặc trưng SIFT CHỈ nằm bên trong lõi mask (co 5px để loại 100% nền/màn hình sau)."""
    x0, y0, x1, y1 = bbox
    gray = cv2.cvtColor(color_img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    m_crop = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    # Co 5px vào trong thân vật thể để tránh bắt nhầm chữ/viền nền ở mép mask
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    m_inner = cv2.erode(m_crop, erode_k, iterations=1)
    if np.count_nonzero(m_inner) < 200:
        m_inner = m_crop
    kps, desc = sift.detectAndCompute(gray, m_inner)
    return kps, desc


def find_turntable_loop_frame(valid_candidates, color_np, masks, min_fraction=0.35):
    """Tìm chính xác frame quay hết 1 vòng 360° về trùng frame 0 bằng:
    1. Fixed-Box Mask IoU (không tự co giãn từng frame -> bắt chính xác vị trí & kích thước quai).
    2. SIFT feature matching trong lõi mask kèm kiểm tra nhất quán tọa độ (bắt họa tiết/logo, loại mặt lưng trơn).
    """
    n = len(valid_candidates)
    if n < 20:
        return None, {"reason": "too_few_frames"}

    ref_idx = valid_candidates[0]
    ref_mask = masks[ref_idx]
    bbox = _mask_bbox_padded(ref_mask)
    if bbox is None:
        return None, {"reason": "invalid_ref_mask"}
    x0, y0, x1, y1 = bbox
    box_diag = math.hypot(x1 - x0, y1 - y0)

    ref_m_crop = cv2.resize(
        (ref_mask[y0:y1, x0:x1] > 0).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST
    ) > 0

    try:
        sift = cv2.SIFT_create(nfeatures=400)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        use_sift = True
    except Exception:
        sift = cv2.ORB_create(nfeatures=400)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        use_sift = False

    ref_kps, ref_desc = _extract_sift_in_mask(color_np[ref_idx], ref_mask, bbox, sift)
    has_ref_tex = (ref_desc is not None and len(ref_kps) >= 8)

    # Ảnh cạnh (Sobel gradient) phòng trường hợp vật thể trơn hoàn toàn không có SIFT
    ref_gray = cv2.cvtColor(color_np[ref_idx][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref_grad = cv2.magnitude(cv2.Sobel(ref_gray, cv2.CV_32F, 1, 0, ksize=3),
                             cv2.Sobel(ref_gray, cv2.CV_32F, 0, 1, ksize=3))
    ref_grad = cv2.resize(ref_grad, (128, 128), interpolation=cv2.INTER_AREA)

    start_pos = max(15, int(n * min_fraction))
    scores = []
    match_counts = []
    iou_list = []

    for pos in range(start_pos, n):
        idx = valid_candidates[pos]
        m_full = masks[idx]
        if m_full is None:
            scores.append(-1.0)
            match_counts.append(0)
            iou_list.append(0.0)
            continue

        # 1. Fixed-Box Mask IoU (giữ nguyên tỷ lệ gốc, không phóng to khi mất quai)
        m_crop = cv2.resize(
            (m_full[y0:y1, x0:x1] > 0).astype(np.uint8), (128, 128), interpolation=cv2.INTER_NEAREST
        ) > 0
        inter = np.logical_and(ref_m_crop, m_crop)
        union = np.logical_or(ref_m_crop, m_crop)
        fixed_iou = float(np.count_nonzero(inter)) / max(1, int(np.count_nonzero(union)))

        # 2. Khớp đặc trưng SIFT trong lõi mask kèm lọc khoảng cách dịch chuyển
        spatial_matches = 0
        if has_ref_tex:
            kps, desc = _extract_sift_in_mask(color_np[idx], m_full, bbox, sift)
            if desc is not None and len(kps) >= 6:
                raw_matches = bf.knnMatch(ref_desc, desc, k=2)
                max_disp = 0.14 * box_diag  # Điểm trên họa tiết phải quay về đúng vị trí cũ
                for pair in raw_matches:
                    if len(pair) < 2:
                        continue
                    m1, m2 = pair
                    if m1.distance < (0.75 if use_sift else 0.80) * m2.distance:
                        p1 = ref_kps[m1.queryIdx].pt
                        p2 = kps[m1.trainIdx].pt
                        if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= max_disp:
                            spatial_matches += 1

        if has_ref_tex:
            feat_score = min(1.0, float(spatial_matches) / max(12.0, 0.25 * len(ref_kps)))
            score = 0.40 * fixed_iou + 0.60 * feat_score
        else:
            g = cv2.cvtColor(color_np[idx][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                                 cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
            grad = cv2.resize(grad, (128, 128), interpolation=cv2.INTER_AREA)
            if np.count_nonzero(inter) >= 50:
                v1 = ref_grad[inter] - ref_grad[inter].mean()
                v2 = grad[inter] - grad[inter].mean()
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                grad_ncc = max(0.0, float(np.dot(v1, v2) / denom)) if denom > 1e-6 else 0.0
            else:
                grad_ncc = 0.0
            score = 0.65 * fixed_iou + 0.35 * grad_ncc

        scores.append(score)
        match_counts.append(spatial_matches)
        iou_list.append(fixed_iou)

    if not scores:
        return None, {"reason": "empty_search"}

    scores_arr = np.asarray(scores, dtype=np.float64)
    if len(scores_arr) >= 5:
        smooth_scores = np.convolve(scores_arr, np.array([0.25, 0.5, 0.25]), mode="same")
    else:
        smooth_scores = scores_arr

    best_offset = int(np.argmax(smooth_scores))
    best_score = float(smooth_scores[best_offset])
    best_pos = start_pos + best_offset
    best_frame = valid_candidates[best_pos]

    if best_score < 0.62:
        return None, {
            "reason": "no_clear_360_match",
            "best_frame": int(best_frame),
            "best_score": round(best_score, 4),
            "best_matches": int(match_counts[best_offset]),
            "best_iou": round(iou_list[best_offset], 4),
        }

    return best_pos, {
        "reason": "matched_360_loop_sift_iou",
        "loop_pos": int(best_pos),
        "loop_frame": int(best_frame),
        "total_candidates": int(n),
        "score": round(best_score, 4),
        "sift_matches": int(match_counts[best_offset]),
        "fixed_iou": round(iou_list[best_offset], 4),
    }


def refine_timed_turntable_yaw(source, reference, feature_source, feature_reference,
                               pivot, axis, prev_yaw, expected_step, rot_sign, args):
    """Ước lượng bước xoay 1-DoF cho bàn xoay tay: tìm bước tiến đơn điệu quanh prev_yaw
    (cho phép dừng 0° hoặc quay nhanh tới ~14°/step, chặn ghì lùi trên thân trụ tròn).
    """
    src_pts = np.asarray(source.points)
    ref_pts = np.asarray(reference.points)
    expected_yaw = prev_yaw + expected_step
    if len(src_pts) < 100 or len(ref_pts) < 100:
        return turntable_pose(pivot, axis, float(expected_yaw)), {
            "yaw_deg": round(float(expected_yaw), 3),
            "step_deg": round(float(expected_step), 3),
            "reason": "time_prior_too_few_points",
            "fitness": None,
            "rmse_mm": None,
        }

    src_cols = np.asarray(source.colors) if source.has_colors() else None
    ref_cols = np.asarray(reference.colors) if reference.has_colors() else None
    ref_tree = cKDTree(ref_pts)

    use_feat = (len(feature_source.points) >= 35 and len(feature_reference.points) >= 35)
    if use_feat:
        f_src_pts = np.asarray(feature_source.points)
        f_src_cols = np.asarray(feature_source.colors) if feature_source.has_colors() else None
        f_ref_pts = np.asarray(feature_reference.points)
        f_ref_cols = np.asarray(feature_reference.colors) if feature_reference.has_colors() else None
        f_ref_tree = cKDTree(f_ref_pts)
    else:
        f_src_pts = f_src_cols = f_ref_cols = f_ref_tree = None

    max_dist = min(0.008, float(args.turntable_max_correspondence))

    def _eval_step(step_mag):
        angle_deg = prev_yaw + rot_sign * step_mag
        pose = turntable_pose(pivot, axis, float(angle_deg))
        b_fit, b_rmse, b_col = _score_turntable_alignment(
            src_pts, src_cols, ref_tree, ref_cols, pose, max_dist
        )
        if use_feat:
            f_fit, _, _ = _score_turntable_alignment(
                f_src_pts, f_src_cols, f_ref_tree, f_ref_cols, pose, max_dist * 1.25
            )
        else:
            f_fit = 0.0

        # Phạt nhẹ nếu lệch quá xa tốc độ quay trung bình, ưu tiên RMSE thấp + màu RGB + đặc trưng rìa
        exp_mag = abs(expected_step)
        prior_penalty = 0.008 * abs(step_mag - exp_mag)
        score = (
            0.80 * b_fit
            + 0.55 * f_fit
            - 45.0 * b_rmse
            - 0.85 * b_col
            - prior_penalty
        )
        return score, float(angle_deg), float(rot_sign * step_mag), pose, b_fit, b_rmse, b_col, f_fit

    # Chỉ tìm bước tiến quanh tốc độ quay kỳ vọng [0.45*exp, 2.2*exp] để tránh kẹt ở 0° trên trụ tròn
    exp_mag = abs(expected_step)
    min_step_mag = max(0.0, 0.45 * exp_mag)
    max_step_mag = min(16.0, max(8.0, 2.2 * exp_mag))
    coarse_steps = np.arange(min_step_mag, max_step_mag + 0.5, 0.5)

    best = None
    for s in coarse_steps:
        res = _eval_step(float(s))
        if best is None or res[0] > best[0]:
            best = res

    # Quét mịn 0.25° quanh bước tốt nhất
    best_step_mag = abs(best[2])
    fine_lo = max(min_step_mag, best_step_mag - 0.5)
    fine_hi = min(max_step_mag, best_step_mag + 0.5)
    for s in np.arange(fine_lo, fine_hi + 0.125, 0.25):
        res = _eval_step(float(s))
        if res[0] > best[0]:
            best = res

    _, angle, signed_step, pose, b_fit, b_rmse, b_col, f_fit = best
    return pose, {
        "yaw_deg": round(angle, 3),
        "step_deg": round(signed_step, 3),
        "expected_yaw_deg": round(float(expected_yaw), 3),
        "fitness": round(b_fit, 4),
        "rmse_mm": round(b_rmse * 1000.0, 3),
        "color_err": round(b_col, 4),
        "feature_fitness": round(f_fit, 4) if use_feat else None,
        "reason": "monotone_hand_turntable_refinement",
    }


def _project_delta_to_planar_3dof(T_delta, pivot, axis, max_dyaw_deg=12.0, max_shift_m=0.012):
    """Chiếu ma trận chỉnh nhỏ T_delta về đúng 3-DoF phẳng trên mặt bàn:
    - Xoay dyaw quanh (pivot, axis) giới hạn ±max_dyaw_deg
    - Tịnh tiến ngang (dx, dz) vuông góc axis giới hạn <= max_shift_m
    - Khóa cứng chiều cao dọc trục axis và góc nghiêng roll/pitch = 0.
    """
    R_d = T_delta[:3, :3]
    t_d = T_delta[:3, 3]
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    pivot = np.asarray(pivot, dtype=np.float64)

    # Tách góc xoay quanh trục axis bằng vector trực giao u
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = R_d @ u
    v_proj = v - np.dot(v, axis) * axis
    v_norm = float(np.linalg.norm(v_proj))
    if v_norm > 1e-6:
        v_proj /= v_norm
        sin_a = float(np.dot(np.cross(u, v_proj), axis))
        cos_a = float(np.clip(np.dot(u, v_proj), -1.0, 1.0))
        dyaw_deg = float(np.degrees(np.arctan2(sin_a, cos_a)))
    else:
        dyaw_deg = 0.0
    dyaw_deg = float(np.clip(dyaw_deg, -max_dyaw_deg, max_dyaw_deg))

    # Tách độ dịch tâm xoay và chỉ giữ thành phần ngang vuông góc axis
    t_pivot = (R_d @ pivot + t_d) - pivot
    t_horiz = t_pivot - np.dot(t_pivot, axis) * axis
    shift_norm = float(np.linalg.norm(t_horiz))
    if shift_norm > max_shift_m:
        t_horiz = t_horiz * (max_shift_m / shift_norm)
        shift_norm = max_shift_m

    T_3dof = turntable_pose(pivot, axis, dyaw_deg)
    T_3dof[:3, 3] += t_horiz
    return T_3dof, dyaw_deg, shift_norm


def smooth_and_enforce_monotonic_poses(active_indices, poses_map, pivot, axis, rot_sign):
    """Làm mượt nhẹ và ép đơn điệu chuỗi góc yaw giữa 2 lượt Pass 2 để chặn trượt ngược ở mặt lưng trụ."""
    if len(active_indices) < 3:
        return poses_map

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    pivot = np.asarray(pivot, dtype=np.float64)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, axis)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)

    raw_rads = []
    horiz_shifts = []
    for f_idx in active_indices:
        T = poses_map[f_idx]
        R_m, t_v = T[:3, :3], T[:3, 3]
        v = R_m @ u
        v_proj = v - np.dot(v, axis) * axis
        v_norm = float(np.linalg.norm(v_proj))
        if v_norm > 1e-6:
            v_proj /= v_norm
            sin_a = float(np.dot(np.cross(u, v_proj), axis))
            cos_a = float(np.clip(np.dot(u, v_proj), -1.0, 1.0))
            rad = float(np.arctan2(sin_a, cos_a))
        else:
            rad = 0.0
        raw_rads.append(rad)
        t_piv = (R_m @ pivot + t_v) - pivot
        t_h = t_piv - np.dot(t_piv, axis) * axis
        horiz_shifts.append(t_h)

    yaws = np.degrees(np.unwrap(np.asarray(raw_rads, dtype=np.float64)))
    # Làm mượt nhẹ [0.25, 0.5, 0.25] giữ nguyên 2 đầu
    yaws_smooth = yaws.copy()
    if len(yaws) >= 5:
        yaws_smooth[1:-1] = 0.25 * yaws[:-2] + 0.5 * yaws[1:-1] + 0.25 * yaws[2:]

    # Ép đơn điệu theo đúng chiều quay rot_sign
    for k in range(1, len(yaws_smooth)):
        if rot_sign * yaws_smooth[k] < rot_sign * yaws_smooth[k - 1]:
            yaws_smooth[k] = yaws_smooth[k - 1]

    out = {}
    for f_idx, y_deg, t_h in zip(active_indices, yaws_smooth, horiz_shifts):
        T_new = turntable_pose(pivot, axis, float(y_deg))
        T_new[:3, 3] += t_h
        out[f_idx] = T_new
    return out


def refine_turntable_global_pass(active_indices, init_yaws, frame_clouds, frame_features,
                                 pivot, axis, args,
                                 max_dyaw_deg=12.0, max_shift_m=0.012,
                                 exclude_radius=2, n_iters=2,
                                 init_poses_override=None):
    """Pass 2 (Leave-One-Out 3-DoF ICP):
    Căn từng frame vào mô hình 360° ĐÃ LOẠI CHÍNH NÓ (và ±exclude_radius frame kề)
    để bỏ tự thiên vị ở yaw0, đồng thời bù độ rơ xoay tay (dx, dz, dyaw).
    Trả về (refined_poses_dict, diagnostics).
    """
    n_active = len(active_indices)
    if init_poses_override is not None:
        init_poses = {f_idx: init_poses_override[f_idx].copy() for f_idx in active_indices}
    else:
        init_poses = {
            f_idx: turntable_pose(pivot, axis, float(yaw))
            for f_idx, yaw in zip(active_indices, init_yaws)
        }
    if n_active < 8:
        return init_poses, {"refined": False, "reason": "too_few_frames"}

    voxel = float(args.turntable_voxel)
    icp_dist = max(0.008, voxel * 3.5)

    # Chuẩn bị sẵn point + normal cục bộ cho từng frame (chỉ tính 1 lần)
    local_data = {}
    for f_idx in active_indices:
        pcd = frame_clouds.get(f_idx)
        if pcd is None or len(pcd.points) < 80:
            continue
        c = o3d.geometry.PointCloud(pcd)
        if not c.has_normals():
            c.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3.5, max_nn=25)
            )
            c.orient_normals_towards_camera_location(np.zeros(3))
        local_data[f_idx] = (
            np.asarray(c.points, dtype=np.float64),
            np.asarray(c.normals, dtype=np.float64),
        )

    current_poses = {k: v.copy() for k, v in init_poses.items()}
    total_yaw_shifts = []
    total_xy_shifts_mm = []
    accepted_count = 0

    for _it in range(n_iters):
        # Biến đổi tất cả các frame sang không gian world 1 lần cho vòng lặp này
        world_pts_list = []
        world_nrm_list = []
        world_pos_ids = []
        for pos, f_idx in enumerate(active_indices):
            if f_idx not in local_data:
                continue
            pts_l, nrm_l = local_data[f_idx]
            T = current_poses[f_idx]
            R_m, t_v = T[:3, :3], T[:3, 3]
            world_pts_list.append((pts_l @ R_m.T) + t_v)
            world_nrm_list.append(nrm_l @ R_m.T)
            world_pos_ids.append(np.full(len(pts_l), pos, dtype=np.int32))

        if not world_pts_list:
            break

        all_pts = np.vstack(world_pts_list)
        all_nrm = np.vstack(world_nrm_list)
        all_ids = np.concatenate(world_pos_ids)

        new_poses = {active_indices[0]: current_poses[active_indices[0]].copy()}
        iter_yaw_shifts = []
        iter_xy_shifts = []
        iter_accepted = 0

        for pos in range(1, n_active):
            f_idx = active_indices[pos]
            if f_idx not in local_data:
                new_poses[f_idx] = current_poses[f_idx].copy()
                continue

            # Loại bỏ chính frame đang xét và ±exclude_radius frame lân cận (tính vòng tròn 360°)
            dist_circ = np.minimum(np.abs(all_ids - pos), n_active - np.abs(all_ids - pos))
            keep = dist_circ > exclude_radius
            if int(np.count_nonzero(keep)) < 300:
                new_poses[f_idx] = current_poses[f_idx].copy()
                continue

            ref_pcd = o3d.geometry.PointCloud()
            ref_pcd.points = o3d.utility.Vector3dVector(all_pts[keep])
            ref_pcd.normals = o3d.utility.Vector3dVector(all_nrm[keep])

            pts_l, nrm_l = local_data[f_idx]
            T_cur = current_poses[f_idx]
            src_pcd = o3d.geometry.PointCloud()
            src_pcd.points = o3d.utility.Vector3dVector((pts_l @ T_cur[:3, :3].T) + T_cur[:3, 3])
            src_pcd.normals = o3d.utility.Vector3dVector(nrm_l @ T_cur[:3, :3].T)

            reg = o3d.pipelines.registration.registration_icp(
                src_pcd,
                ref_pcd,
                max_correspondence_distance=icp_dist,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15),
            )

            if float(reg.fitness) >= 0.45 and 1e-9 < float(reg.inlier_rmse) <= 0.009:
                # Giới hạn tổng độ lệch so với pose khởi tạo Pass 1 để không bị trượt tích lũy
                T_cand_world = reg.transformation @ T_cur
                T_total_delta = T_cand_world @ np.linalg.inv(init_poses[f_idx])
                T_3dof, dyaw, shift_m = _project_delta_to_planar_3dof(
                    T_total_delta, pivot, axis,
                    max_dyaw_deg=max_dyaw_deg, max_shift_m=max_shift_m
                )
                new_poses[f_idx] = T_3dof @ init_poses[f_idx]
                iter_yaw_shifts.append(abs(dyaw))
                iter_xy_shifts.append(shift_m * 1000.0)
                iter_accepted += 1
            else:
                new_poses[f_idx] = T_cur.copy()

        current_poses = new_poses
        total_yaw_shifts = iter_yaw_shifts
        total_xy_shifts_mm = iter_xy_shifts
        accepted_count = iter_accepted

    return current_poses, {
        "refined": True,
        "accepted_frames": f"{accepted_count}/{n_active - 1}",
        "mean_dyaw_deg": round(float(np.mean(total_yaw_shifts)), 2) if total_yaw_shifts else 0.0,
        "max_dyaw_deg": round(float(np.max(total_yaw_shifts)), 2) if total_yaw_shifts else 0.0,
        "mean_shift_mm": round(float(np.mean(total_xy_shifts_mm)), 2) if total_xy_shifts_mm else 0.0,
        "max_shift_mm": round(float(np.max(total_xy_shifts_mm)), 2) if total_xy_shifts_mm else 0.0,
    }

def submap_cloud(indices, poses, color_np, depth_np, masks, intrinsic_tensor, voxel_size,
                 object_only=None):
    cloud = None
    stride = max(1, len(indices) // 40)
    for index in indices[::stride]:
        frame_cloud = frame_point_cloud(index, color_np, depth_np, masks, intrinsic_tensor, object_only)
        pose_t = o3d.core.Tensor(poses[index], o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
        frame_cloud.transform(pose_t)
        if cloud is None:
            cloud = frame_cloud.clone()
        else:
            cloud = cloud.append(frame_cloud)
            
    if cloud is None:
        return o3d.t.geometry.PointCloud(O3D_DEVICE_GPU)
        
    cloud = cloud.voxel_down_sample(voxel_size)
    if len(cloud.point.positions) >= 30:
        cloud.estimate_normals(max_nn=40, radius=voxel_size * 3.0)
    return cloud

def filter_mesh_by_view_support(mesh, indices, poses, depth_np, masks, intrinsic,
                                min_support, depth_tolerance_m):
    if min_support <= 0 or len(mesh.vertices) == 0:
        return mesh, None
    vertices = np.asarray(mesh.vertices)
    support = np.zeros(len(vertices), dtype=np.uint16)
    fx, fy, cx, cy = intrinsic.get_focal_length()[0], intrinsic.get_focal_length()[1], *intrinsic.get_principal_point()
    for index in indices:
        mask = masks[index]
        if mask is None:
            continue
        world_to_camera = np.linalg.inv(poses[index])
        points = (world_to_camera[:3, :3] @ vertices.T).T + world_to_camera[:3, 3]
        valid_z = points[:, 2] > 0
        u = np.rint(fx * points[:, 0] / np.maximum(points[:, 2], 1e-8) + cx).astype(int)
        v = np.rint(fy * points[:, 1] / np.maximum(points[:, 2], 1e-8) + cy).astype(int)
        in_image = valid_z & (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        ids = np.where(in_image)[0]
        if len(ids) == 0:
            continue
        observed_depth = depth_np[index][v[ids], u[ids]].astype(np.float32) / 1000.0
        observed = ((mask[v[ids], u[ids]] > 0) & (observed_depth > 0) &
                    (np.abs(observed_depth - points[ids, 2]) <= depth_tolerance_m))
        support[ids[observed]] += 1
    triangles = np.asarray(mesh.triangles)
    keep = np.all(support[triangles] >= min_support, axis=1)
    mesh.remove_triangles_by_mask(~keep)
    mesh.remove_unreferenced_vertices()
    return mesh, support


def flatten_planar_regions(
    mesh,
    max_planes: int = 6,
    ransac_distance_m: float = 0.0045,
    normal_angle_thresh_deg: float = 18.0,
    min_cluster_ratio: float = 0.03,
    blend_alpha: float = 0.85,
):
    """
    Làm phẳng tự thích nghi các mặt phẳng thật (kể cả mặt bên hẹp của Box) và
    TỰ ĐỘNG BỎ QUA bề mặt cong/trụ (Cốc, chai lọ) theo 2 tiêu chuẩn bất biến tỷ lệ:
      1. Normal-Position Correlation Gate (|r| > 0.42)
      2. Scale-Invariant Radius of Curvature Test: R_est = span^2 / (8 * |sagitta|) < 0.22m
         kèm |sagitta| > 1.4mm (không bao giờ loại nhầm mặt hộp rộng bị gợn nhiễu D435f).
    """
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles)
    n_verts = len(vertices)
    if n_verts < 500 or len(triangles) < 500:
        return mesh, []

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals).copy()

    adj = [[] for _ in range(n_verts)]
    for a, b, c in triangles:
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].extend((bi, ci))
        adj[bi].extend((ai, ci))
        adj[ci].extend((ai, bi))

    remaining_mask = np.ones(n_verts, dtype=bool)
    min_verts = max(250, int(min_cluster_ratio * n_verts))
    cos_thresh = float(np.cos(np.radians(normal_angle_thresh_deg)))
    reports = []

    max_attempts = max_planes * 4
    for _attempt in range(max_attempts):
        if len(reports) >= max_planes:
            break

        rem_indices = np.where(remaining_mask)[0]
        if len(rem_indices) < min_verts:
            break

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices[rem_indices])
        try:
            plane_model, inliers_local = pcd.segment_plane(
                distance_threshold=ransac_distance_m,
                ransac_n=3,
                num_iterations=1000,
            )
        except Exception:
            break

        if len(inliers_local) < min_verts:
            break

        n_plane = np.array(plane_model[:3], dtype=np.float64)
        n_norm = float(np.linalg.norm(n_plane))
        if n_norm < 1e-8:
            break
        n_plane /= n_norm
        d_plane = float(plane_model[3]) / n_norm

        inlier_global = rem_indices[np.asarray(inliers_local, dtype=np.int64)]
        dots = np.abs(normals[inlier_global] @ n_plane)
        aligned_global = inlier_global[dots >= cos_thresh]

        if len(aligned_global) < min_verts:
            if len(aligned_global) > 0:
                remaining_mask[aligned_global] = False
            else:
                remaining_mask[inlier_global] = False
            continue

        # Tìm miền liên thông lớn nhất trên mesh trong tập aligned_global
        in_set = np.zeros(n_verts, dtype=bool)
        in_set[aligned_global] = True
        visited = np.zeros(n_verts, dtype=bool)
        best_cluster = []

        for seed in aligned_global:
            s = int(seed)
            if visited[s]:
                continue
            comp = []
            queue = [s]
            visited[s] = True
            while queue:
                curr = queue.pop()
                comp.append(curr)
                for nb in adj[curr]:
                    if in_set[nb] and not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            if len(comp) > len(best_cluster):
                best_cluster = comp

        if len(best_cluster) < min_verts:
            remaining_mask[aligned_global] = False
            continue

        cluster_idx = np.array(best_cluster, dtype=np.int64)
        # Chỉ loại cụm đã xét, bảo toàn 100% mép tường vuông góc 90° của Box
        remaining_mask[cluster_idx] = False

        pts = vertices[cluster_idx]
        nms = normals[cluster_idx]

        if float(np.mean(nms @ n_plane)) < 0.0:
            n_plane = -n_plane
            d_plane = -d_plane

        # ── KIỂM TRA HÌNH HỌC BẤT BIẾN TỶ LỆ: MẶT PHẲNG vs MẶT CONG/TRỤ ────
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        pts_2d = centered - np.outer(centered @ n_plane, n_plane)
        cov = (pts_2d.T @ pts_2d) / max(1, len(pts_2d))
        _, eigvecs = np.linalg.eigh(cov)
        u_major = eigvecs[:, 2]
        u_minor = eigvecs[:, 1]

        # 1. Normal-Position Correlation Gate
        is_curved_surface = False
        for axis_vec in (u_minor, u_major):
            coord = centered @ axis_vec
            norm_proj = nms @ axis_vec
            c_std = float(np.std(coord))
            n_std = float(np.std(norm_proj))
            if c_std > 0.005 and n_std > 0.04:
                corr = float(np.corrcoef(coord, norm_proj)[0, 1])
                if np.isfinite(corr) and abs(corr) > 0.42:
                    is_curved_surface = True
                    break

        # 2. Scale-Invariant Radius of Curvature Test (R_est = span^2 / (8 * |sagitta|))
        if not is_curved_surface:
            coord_minor = centered @ u_minor
            span_minor = float(np.percentile(coord_minor, 95) - np.percentile(coord_minor, 5))
            if span_minor > 0.015:
                signed_dist = (pts @ n_plane) + d_plane
                q_abs = np.abs(coord_minor)
                inner_mask = q_abs <= np.percentile(q_abs, 35)
                outer_mask = q_abs >= np.percentile(q_abs, 70)
                if np.any(inner_mask) and np.any(outer_mask):
                    arch_sagitta = abs(
                        float(np.median(signed_dist[inner_mask]) - np.median(signed_dist[outer_mask]))
                    )
                    if arch_sagitta > 0.0014:
                        r_est_m = (span_minor * span_minor) / (8.0 * arch_sagitta)
                        if r_est_m < 0.22:
                            is_curved_surface = True

        if is_curved_surface:
            continue

        # ── LÀM PHẲNG MẶT PHẲNG THẬT KÈM FEATHERING BIÊN ───────────────────
        cluster_set = set(best_cluster)
        is_boundary = np.array(
            [any(nb not in cluster_set for nb in adj[int(v)]) for v in cluster_idx],
            dtype=bool,
        )
        weights = np.where(is_boundary, 0.35 * blend_alpha, blend_alpha)

        signed_d = (pts @ n_plane) + d_plane
        signed_d_clamped = np.clip(signed_d, -0.0040, 0.0040)
        correction = np.outer(weights * signed_d_clamped, n_plane)
        vertices[cluster_idx] = pts - correction

        corr_mm = np.abs(weights * signed_d_clamped) * 1000.0
        reports.append(
            {
                "plane": len(reports),
                "vertices": int(len(cluster_idx)),
                "rms_correction_mm": round(float(np.sqrt(np.mean(corr_mm ** 2))), 3),
                "max_correction_mm": round(float(np.max(corr_mm)), 3),
            }
        )

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh, reports


def smooth_open_boundary_loops(mesh, iterations=12, alpha=0.5):
    """Làm mịn các vòng biên hở (open boundary loops) trên mesh (như miệng cốc, cổ chai, mép vỏ).
    Thuật toán Laplacian 1D chạy dọc chu vi đường biên để triệt tiêu răng cưa / gai nhiễu depth sensor."""
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices).copy()
    if len(triangles) == 0:
        return mesh, {"loops": 0}

    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) < 3:
        return mesh, {"loops": 0}

    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))

    seen, loops = set(), []
    for start in adjacency:
        if start in seen:
            continue
        curr = start
        loop = [curr]
        seen.add(curr)
        while True:
            nbrs = [n for n in adjacency[curr] if n not in seen]
            if not nbrs:
                break
            curr = nbrs[0]
            loop.append(curr)
            seen.add(curr)
        if len(loop) >= 6:
            loops.append(loop)

    if not loops:
        return mesh, {"loops": 0}

    for loop in loops:
        loop_arr = np.array(loop)
        for _ in range(iterations):
            prev_v = vertices[np.roll(loop_arr, 1)]
            next_v = vertices[np.roll(loop_arr, -1)]
            curr_v = vertices[loop_arr]
            vertices[loop_arr] = (1.0 - alpha) * curr_v + (alpha / 2.0) * (prev_v + next_v)

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh, {"loops": len(loops), "iterations": iterations}


def trim_and_cap_support_plane(mesh, cut_margin_mm=4.0, ransac_dist_mm=4.0):
    """Tự động phát hiện mặt phẳng đỡ (support/table plane) dưới chân vật thể bằng RANSAC.
    Cắt bỏ phần lem nhem của mặt bàn/sàn và bịt kín đáy phẳng (Planar Disk Cap)."""
    v = np.asarray(mesh.vertices).copy()
    c = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
    triangles = np.asarray(mesh.triangles).copy()

    if len(v) < 100:
        return mesh, {"trimmed": False, "reason": "too_few_vertices"}

    # Trong hệ tọa độ camera chuẩn (OpenCV/RealSense), trục Y hướng xuống, đáy vật thể ở Y max.
    y_min, y_max = v[:, 1].min(), v[:, 1].max()
    bottom_thresh = y_min + 0.80 * (y_max - y_min)
    bottom_mask = v[:, 1] > bottom_thresh
    bottom_pts = v[bottom_mask]

    if len(bottom_pts) < 30:
        return mesh, {"trimmed": False, "reason": "too_few_bottom_points"}

    pcd_bottom = o3d.geometry.PointCloud()
    pcd_bottom.points = o3d.utility.Vector3dVector(bottom_pts)
    try:
        plane_model, inliers = pcd_bottom.segment_plane(
            distance_threshold=ransac_dist_mm / 1000.0, ransac_n=3, num_iterations=1000)
    except Exception as e:
        return mesh, {"trimmed": False, "reason": f"ransac_failed: {e}"}

    a, b, c_pl, d = plane_model
    normal = np.array([a, b, c_pl])
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-8:
        return mesh, {"trimmed": False, "reason": "degenerate_plane_normal"}
    normal = normal / normal_len
    # Đảm bảo vector pháp tuyến hướng lên trên (vào lòng vật thể, tức chiều -Y)
    if normal[1] > 0:
        normal = -normal

    y_cut = y_max - (cut_margin_mm / 1000.0)
    centroid = bottom_pts.mean(axis=0)
    origin = np.array([centroid[0], y_cut, centroid[2]])

    signed_dists = (v - origin) @ normal
    valid_v_mask = signed_dists >= 0

    tri_valid = valid_v_mask[triangles[:, 0]] & valid_v_mask[triangles[:, 1]] & valid_v_mask[triangles[:, 2]]
    kept_triangles = triangles[tri_valid]

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(v)
    if c is not None and len(c) == len(v):
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(c)
    new_mesh.triangles = o3d.utility.Vector3iVector(kept_triangles)
    new_mesh.remove_unreferenced_vertices()

    v_new = np.asarray(new_mesh.vertices)
    c_new = np.asarray(new_mesh.vertex_colors) if new_mesh.has_vertex_colors() else None
    tri_new = np.asarray(new_mesh.triangles)

    edges = np.sort(np.vstack((tri_new[:, [0, 1]], tri_new[:, [1, 2]], tri_new[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]

    boundary_v_idx = np.unique(boundary_edges)
    dists_to_cut = np.abs((v_new[boundary_v_idx] - origin) @ normal)
    base_b_idx = boundary_v_idx[dists_to_cut < 0.006]

    capped = False
    if len(base_b_idx) >= 6:
        # Chiếu các đỉnh mép đáy phẳng tuyệt đối lên mặt phẳng đỡ
        for b_i in base_b_idx:
            pt = v_new[b_i]
            dist_p = (pt - origin) @ normal
            v_new[b_i] = pt - dist_p * normal

        base_center = v_new[base_b_idx].mean(axis=0)
        center_idx = len(v_new)
        v_new = np.vstack([v_new, base_center[None, :]])
        if c_new is not None:
            base_color = np.median(c_new[base_b_idx], axis=0)
            c_new = np.vstack([c_new, base_color[None, :]])

        ref_vec = v_new[base_b_idx[0]] - base_center
        ref_norm = np.linalg.norm(ref_vec)
        if ref_norm > 1e-8:
            ref_vec = ref_vec / ref_norm
            cross_vec = np.cross(normal, ref_vec)
            cross_norm = np.linalg.norm(cross_vec)
            if cross_norm > 1e-8:
                cross_vec = cross_vec / cross_norm
                pts_rel = v_new[base_b_idx] - base_center
                angles = np.arctan2(pts_rel @ cross_vec, pts_rel @ ref_vec)
                sorted_order = np.argsort(angles)
                sorted_base = base_b_idx[sorted_order]

                new_tris = []
                for i in range(len(sorted_base)):
                    i_next = (i + 1) % len(sorted_base)
                    new_tris.append([center_idx, sorted_base[i], sorted_base[i_next]])

                tri_new = np.vstack([tri_new, new_tris])
                capped = True

    capped_mesh = o3d.geometry.TriangleMesh()
    capped_mesh.vertices = o3d.utility.Vector3dVector(v_new)
    if c_new is not None:
        capped_mesh.vertex_colors = o3d.utility.Vector3dVector(c_new)
    capped_mesh.triangles = o3d.utility.Vector3iVector(tri_new)
    capped_mesh.compute_vertex_normals()

    return capped_mesh, {
        "trimmed": True,
        "capped": capped,
        "trimmed_vertices": len(v) - len(v_new),
        "support_normal": normal.tolist()
    }
