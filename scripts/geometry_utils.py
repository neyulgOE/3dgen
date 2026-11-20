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

def refine_timed_turntable_yaw(source, reference, feature_source, feature_reference,
                                source_pivot, target_pivot, axis, expected_yaw, args):
    if len(source.points) < 100 or len(reference.points) < 100:
        return turntable_pose_between(source_pivot, target_pivot, axis, -expected_yaw), {
            "yaw_deg": round(float(expected_yaw), 3), "reason": "time_prior_too_few_points",
            "feature_fitness": None, "rmse_mm": None}
    correction_limit = min(18.0, max(6.0, args.turntable_max_step_deg * 0.75))
    step = max(1.0, args.turntable_yaw_step_deg)
    angles = np.arange(expected_yaw - correction_limit,
                       expected_yaw + correction_limit + step * 0.5, step)
    best = None
    for angle in angles:
        pose = turntable_pose_between(source_pivot, target_pivot, axis, -float(angle))
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
        key = (feature_fitness, float(body_score.fitness),
               -float(body_score.inlier_rmse), -0.01 * abs(float(angle) - expected_yaw))
        if best is None or key > best[0]:
            best = (key, float(angle), pose, body_score, feature_score)
    _, angle, pose, body_score, feature_score = best
    return pose, {
        "yaw_deg": round(angle, 3), "expected_yaw_deg": round(float(expected_yaw), 3),
        "yaw_correction_deg": round(angle - float(expected_yaw), 3),
        "fitness": round(float(body_score.fitness), 4),
        "rmse_mm": round(float(body_score.inlier_rmse) * 1000.0, 3),
        "feature_fitness": round(float(feature_score.fitness), 4) if feature_score is not None else None,
        "reason": "timed_handle_refinement",
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
    ransac_distance_m: float = 0.0035,
    normal_angle_thresh_deg: float = 15.0,
    min_cluster_ratio: float = 0.05,
    blend_alpha: float = 0.85,
):
    """
    Làm phẳng tự thích nghi các bề mặt phẳng lớn (vách hộp) và TỰ ĐỘNG BỎ QUA
    các bề mặt cong/trụ (thân cốc, chai lọ) nhờ:
      1. Normal-Position Correlation Gate (|r| > 0.42)
      2. Parabolic Arch / Sagitta Sign Test (> 1.0mm)
    Chỉ đánh dấu loại trừ cụm đồng hướng pháp tuyến để bảo toàn cạnh 90° của hộp.
    """
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles)
    n_verts = len(vertices)
    if n_verts < 500 or len(triangles) < 500:
        return mesh, []

    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals).copy()

    # Dựng danh sách kề đỉnh từ lưới tam giác để tìm miền liên thông thực sự
    adj = [[] for _ in range(n_verts)]
    for a, b, c in triangles:
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].extend((bi, ci))
        adj[bi].extend((ai, ci))
        adj[ci].extend((ai, bi))

    remaining_mask = np.ones(n_verts, dtype=bool)
    min_verts = max(350, int(min_cluster_ratio * n_verts))
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

        # Tìm thành phần liên thông lớn nhất trên mesh trong tập aligned_global
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
        remaining_mask[cluster_idx] = False

        pts = vertices[cluster_idx]
        nms = normals[cluster_idx]

        if float(np.mean(nms @ n_plane)) < 0.0:
            n_plane = -n_plane
            d_plane = -d_plane

        # ── KIỂM TRA HÌNH HỌC: MẶT PHẲNG THẬT (BOX) vs MẶT CONG/TRỤ (CỐC) ──
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

        # 2. Parabolic Arch / Sagitta Sign Test dọc theo trục ngang u_minor
        if not is_curved_surface:
            coord_minor = centered @ u_minor
            span_minor = float(np.percentile(coord_minor, 95) - np.percentile(coord_minor, 5))
            if span_minor > 0.012:
                signed_dist = (pts @ n_plane) + d_plane
                q_abs = np.abs(coord_minor)
                inner_mask = q_abs <= np.percentile(q_abs, 35)
                outer_mask = q_abs >= np.percentile(q_abs, 70)
                if np.any(inner_mask) and np.any(outer_mask):
                    arch_sagitta = float(
                        np.median(signed_dist[inner_mask]) - np.median(signed_dist[outer_mask])
                    )
                    if abs(arch_sagitta) > 0.0010:
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
        signed_d_clamped = np.clip(signed_d, -0.0035, 0.0035)
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
