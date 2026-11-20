# -*- coding: utf-8 -*-
"""
Submap-based Pose Graph Optimizer (Geometry-Adaptive Full-Scene v2).
- Dựng Submap sạch bằng cách căn Fine ICP từng frame lấy mẫu vào anchor_frame trước khi gộp.
- Tự động ước lượng depth_trunc cận cảnh (0.65m - 0.95m) giữ đủ mặt bàn neo 6-DOF.
- Chặn false loop trên Box/Cốc: khoảng cách tối thiểu max(8, M//5), lọc chặt LightGlue + ICP,
  và chỉ giữ tối đa Top-3 cạnh loop chất lượng nhất cho mỗi Submap.
- Tinh chỉnh Fine ICP per-frame vào chính Submap sạch đã tối ưu trước khi tích hợp TSDF.
"""
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation, Slerp

from feature_tracker import LightGlueFeatureTracker


def rotation_angle_deg(T: np.ndarray) -> float:
    tr = float(np.trace(T[:3, :3]))
    cos_val = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _interpolate_se3(T_start: np.ndarray, T_end: np.ndarray, alpha: float) -> np.ndarray:
    """Nội suy mượt SE(3) (SLERP cho góc xoay + tuyến tính cho tịnh tiến)."""
    if alpha <= 0.0:
        return T_start.copy()
    if alpha >= 1.0:
        return T_end.copy()
    rots = Rotation.from_matrix([T_start[:3, :3], T_end[:3, :3]])
    slerp = Slerp([0.0, 1.0], rots)
    R_interp = slerp([alpha])[0].as_matrix()
    t_interp = (1.0 - alpha) * T_start[:3, 3] + alpha * T_end[:3, 3]
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = R_interp
    T_out[:3, 3] = t_interp
    return T_out


def _estimate_adaptive_depth_trunc(
    valid_frames: List[int],
    depth_list: List[np.ndarray],
    masks: Optional[List[Optional[np.ndarray]]] = None,
    depth_scale: float = 1000.0,
    margin_m: float = 0.45,
    min_trunc_m: float = 0.65,
    max_trunc_m: float = 0.95,
) -> float:
    """
    Tính tự động depth_trunc cho Full-Scene ICP:
    Đủ rộng để giữ mặt bàn neo 6-DOF quanh cả Box lẫn Cốc (0.65m - 0.95m),
    nhưng vẫn cắt sạch hậu cảnh xa > 1.0m gây nhiễu trên D435f.
    """
    samples = []
    step = max(1, len(valid_frames) // 24)
    for f_idx in valid_frames[::step]:
        d = depth_list[f_idx]
        if d is None:
            continue
        if masks is not None and f_idx < len(masks) and masks[f_idx] is not None:
            vals = d[(masks[f_idx] > 0) & (d > 0)]
        else:
            H, W = d.shape[:2]
            roi = d[H // 4 : (3 * H) // 4, W // 4 : (3 * W) // 4]
            vals = roi[roi > 0]
        if vals.size >= 50:
            samples.append(float(np.percentile(vals, 90)) / depth_scale)

    if not samples:
        return 0.80

    obj_far_m = float(np.median(samples))
    return float(np.clip(obj_far_m + margin_m, min_trunc_m, max_trunc_m))


def _rgbd_to_full_scene_pcd(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0,
    depth_trunc: float = 0.80,
    voxel_size: float = 0.007,
) -> o3d.geometry.PointCloud:
    """Tạo PointCloud toàn cảnh (KHÔNG cắt SAM2 mask) trong vùng cận cảnh sạch nhiễu."""
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth_raw))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
    if len(pcd.points) == 0:
        return pcd
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) >= 30:
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3.0, max_nn=25)
        )
    return pcd


@dataclass
class SubmapNode:
    submap_id: int
    anchor_frame: int
    frame_indices: List[int]
    init_world_pose: np.ndarray
    pcd_local: o3d.geometry.PointCloud
    local_frame_poses: Dict[int, np.ndarray] = field(default_factory=dict)


def _build_full_scene_submaps(
    valid_frames: List[int],
    rgb_list: List[np.ndarray],
    depth_list: List[np.ndarray],
    init_poses: List[np.ndarray],
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    submap_size: int = 6,
    depth_scale: float = 1000.0,
    depth_trunc: float = 0.80,
    voxel_size: float = 0.007,
) -> List[SubmapNode]:
    """
    Gom mỗi 6 frame liên tiếp thành 1 Full-Scene Submap SẠCH:
    Căn Fine Point-to-Plane ICP từng frame lấy mẫu trực tiếp vào anchor_frame
    trước khi gộp vào pcd_local để loại bỏ nhòe do odometry cũ bị trượt.
    """
    submaps: List[SubmapNode] = []
    n = len(valid_frames)
    for sid, start in enumerate(range(0, n, submap_size)):
        chunk = valid_frames[start : min(start + submap_size, n)]
        if not chunk:
            continue
        anchor = chunk[0]
        T_anchor_w = init_poses[anchor].copy()
        T_anchor_inv = np.linalg.inv(T_anchor_w)

        pcd_anchor = _rgbd_to_full_scene_pcd(
            rgb_list[anchor],
            depth_list[anchor],
            intrinsic_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            voxel_size=voxel_size,
        )
        merged = o3d.geometry.PointCloud(pcd_anchor)
        local_poses = {anchor: np.eye(4, dtype=np.float64)}

        sample_step = max(1, len(chunk) // 3)
        for k, f_idx in enumerate(chunk[1:], start=1):
            T_local_init = T_anchor_inv @ init_poses[f_idx]
            T_local_final = T_local_init

            if k % sample_step == 0 or k == len(chunk) - 1:
                pcd_f = _rgbd_to_full_scene_pcd(
                    rgb_list[f_idx],
                    depth_list[f_idx],
                    intrinsic_o3d,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                    voxel_size=voxel_size,
                )
                if len(pcd_f.points) >= 80 and len(pcd_anchor.points) >= 80:
                    reg = o3d.pipelines.registration.registration_icp(
                        pcd_f,
                        pcd_anchor,
                        max_correspondence_distance=voxel_size * 3.0,
                        init=T_local_init,
                        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20),
                    )
                    T_cand = reg.transformation
                    delta = np.linalg.inv(T_local_init) @ T_cand
                    if (
                        float(reg.fitness) >= 0.55
                        and 1e-9 < float(reg.inlier_rmse) <= 0.010
                        and float(np.linalg.norm(delta[:3, 3])) <= 0.030
                        and rotation_angle_deg(delta) <= 8.0
                    ):
                        T_local_final = T_cand
                        pcd_f.transform(T_local_final)
                        merged += pcd_f
            local_poses[f_idx] = T_local_final

        if len(merged.points) > 0:
            merged = merged.voxel_down_sample(voxel_size)
            if len(merged.points) >= 30:
                merged.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 3.0, max_nn=30)
                )

        submaps.append(
            SubmapNode(
                submap_id=sid,
                anchor_frame=anchor,
                frame_indices=chunk,
                init_world_pose=T_anchor_w,
                pcd_local=merged,
                local_frame_poses=local_poses,
            )
        )
    return submaps


def _verify_submap_pair_icp(
    pcd_src: o3d.geometry.PointCloud,
    pcd_tgt: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    voxel_size: float = 0.007,
    min_fitness: float = 0.45,
    max_rmse: float = 0.010,
    max_cycle_t: float = 0.012,
    max_cycle_r: float = 2.0,
):
    """
    Tinh chỉnh Coarse-to-Fine Point-to-Plane ICP và kiểm định thuận-nghịch.
    Chiều nghịch LUÔN khởi tạo từ inv(T_fwd) để triệt tiêu lỗi trượt góc trụ đối xứng.
    """
    if len(pcd_src.points) < 60 or len(pcd_tgt.points) < 60:
        return False, np.eye(4), np.eye(6), {"reason": "sparse_cloud"}

    T_cur = T_init.copy()
    reg_fwd = None
    for dist_mult, max_iter in [(4.0, 25), (2.2, 30)]:
        reg_fwd = o3d.pipelines.registration.registration_icp(
            pcd_src,
            pcd_tgt,
            max_correspondence_distance=voxel_size * dist_mult,
            init=T_cur,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter),
        )
        T_cur = reg_fwd.transformation

    T_fwd = T_cur
    fitness = float(reg_fwd.fitness)
    rmse = float(reg_fwd.inlier_rmse)

    if fitness < min_fitness or rmse > max_rmse or rmse <= 1e-9:
        return False, T_fwd, np.eye(6), {
            "reason": "icp_quality_failed",
            "fitness": fitness,
            "rmse": rmse,
        }

    T_bwd_init = np.linalg.inv(T_fwd)
    reg_bwd = o3d.pipelines.registration.registration_icp(
        pcd_tgt,
        pcd_src,
        max_correspondence_distance=voxel_size * 2.2,
        init=T_bwd_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25),
    )
    T_bwd = reg_bwd.transformation
    T_cycle = T_bwd @ T_fwd
    cycle_t = float(np.linalg.norm(T_cycle[:3, 3]))
    cycle_r = rotation_angle_deg(T_cycle)

    if cycle_t > max_cycle_t or cycle_r > max_cycle_r:
        return False, T_fwd, np.eye(6), {
            "reason": "cycle_consistency_failed",
            "fitness": fitness,
            "rmse": rmse,
            "cycle_t": cycle_t,
            "cycle_r": cycle_r,
        }

    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        pcd_src, pcd_tgt, voxel_size * 2.5, T_fwd
    )
    eigvals = np.linalg.eigvalsh(info)
    if float(np.min(eigvals)) < 1.0:
        return False, T_fwd, info, {"reason": "degenerate_info", "fitness": fitness, "rmse": rmse}

    return True, T_fwd, info, {
        "reason": "accepted",
        "fitness": fitness,
        "rmse": rmse,
        "cycle_t": cycle_t,
        "cycle_r": cycle_r,
    }


def _refine_frame_against_submap(
    rgb_frame: np.ndarray,
    depth_frame: np.ndarray,
    submap_pcd_local: o3d.geometry.PointCloud,
    T_sm_world: np.ndarray,
    T_frame_world_init: np.ndarray,
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0,
    depth_trunc: float = 0.80,
    fine_voxel: float = 0.005,
) -> Tuple[np.ndarray, bool]:
    """Căn tinh từng frame lẻ vào đám mây Submap sạch đã tối ưu bằng Fine Point-to-Plane ICP."""
    if len(submap_pcd_local.points) < 80:
        return T_frame_world_init, False

    pcd_f = _rgbd_to_full_scene_pcd(
        rgb_frame,
        depth_frame,
        intrinsic_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        voxel_size=fine_voxel,
    )
    if len(pcd_f.points) < 80:
        return T_frame_world_init, False

    T_local_init = np.linalg.inv(T_sm_world) @ T_frame_world_init
    reg = o3d.pipelines.registration.registration_icp(
        pcd_f,
        submap_pcd_local,
        max_correspondence_distance=fine_voxel * 2.5,
        init=T_local_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=15),
    )

    T_local_ref = reg.transformation
    delta = np.linalg.inv(T_local_init) @ T_local_ref
    delta_t = float(np.linalg.norm(delta[:3, 3]))
    delta_r = rotation_angle_deg(delta)

    if (
        float(reg.fitness) >= 0.55
        and 1e-9 < float(reg.inlier_rmse) <= 0.010
        and delta_t <= 0.025
        and delta_r <= 5.0
    ):
        return T_sm_world @ T_local_ref, True

    return T_frame_world_init, False


def build_and_optimize_pose_graph(
    valid_frames: List[int],
    rgb_list: List[np.ndarray],
    depth_list: List[np.ndarray],
    init_poses: List[np.ndarray],
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    masks: Optional[List[Optional[np.ndarray]]] = None,
    submap_size: int = 6,
    voxel_size: float = 0.007,
    depth_scale: float = 1000.0,
    depth_trunc: Optional[float] = None,
    loop_min_submap_gap: Optional[int] = None,
    max_loops_per_submap: int = 3,
    refine_per_frame: bool = True,
    log_dir: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """
    Pipeline Submap PGO tự thích nghi v2 (Chống False Loop & Sạch Submap):
      1. Tự ước lượng depth_trunc (0.65m - 0.95m).
      2. Gom mỗi 6 frames thành 1 Submap sạch (đã căn Fine ICP nội bộ vào anchor).
      3. Nối Odometry liên tiếp + tìm ứng viên Loop xa thực sự (gap >= max(8, M//5)),
         chỉ giữ Top-3 cạnh loop tốt nhất mỗi Submap để tránh ép nhầm mặt hộp.
      4. Xả sạch VRAM GPU, giải Global Optimization (Levenberg-Marquardt).
      5. Lan truyền pose về từng frame + chạy Fine ICP per-frame vào Submap.
    """
    t0 = time.time()

    def emit(msg: str):
        print(msg, flush=True)
        if log_callback is not None:
            log_callback(msg)

    K = np.asarray(intrinsic_o3d.intrinsic_matrix, dtype=np.float64)
    optimized_poses = [p.copy() for p in init_poses]

    if len(valid_frames) < 4:
        return optimized_poses, {"components": [len(valid_frames)], "node_degree": {}, "repaired_weak_nodes": []}

    # ── BƯỚC 1: Tự động ước lượng depth_trunc & dựng Full-Scene Submaps sạch ──
    if depth_trunc is None:
        depth_trunc = _estimate_adaptive_depth_trunc(
            valid_frames=valid_frames,
            depth_list=depth_list,
            masks=masks,
            depth_scale=depth_scale,
        )
    emit(
        f"[SubmapPGO] Dựng Full-Scene Submaps sạch (size={submap_size}, "
        f"auto_depth_trunc={depth_trunc:.2f}m) cho {len(valid_frames)} frames..."
    )

    submaps = _build_full_scene_submaps(
        valid_frames=valid_frames,
        rgb_list=rgb_list,
        depth_list=depth_list,
        init_poses=init_poses,
        intrinsic_o3d=intrinsic_o3d,
        submap_size=submap_size,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        voxel_size=voxel_size,
    )
    M = len(submaps)
    if loop_min_submap_gap is None:
        loop_min_submap_gap = max(8, M // 5)
    emit(f"[SubmapPGO] Đã gom thành {M} Submaps sạch (loop_gap>={loop_min_submap_gap}) trong {time.time() - t0:.2f}s.")

    # ── BƯỚC 2: Trích xuất đặc trưng SuperPoint 1 lần trên GPU ────────────────
    t_lg = time.time()
    tracker = LightGlueFeatureTracker(
        max_keypoints=1024,
        depth_scale=depth_scale,
        min_depth_m=0.08,
        max_depth_m=depth_trunc,
    )
    for sm in submaps:
        tracker.extract_and_cache(sm.anchor_frame, rgb_list[sm.anchor_frame])
    emit(f"[SubmapPGO] Trích xong SuperPoint cho {M} anchors trong {time.time() - t_lg:.2f}s.")

    # ── BƯỚC 3: Xây đồ thị PoseGraph ──────────────────────────────────────────
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for sm in submaps:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(sm.init_world_pose.copy()))

    edge_logs = []
    submap_degree = defaultdict(int)
    accepted_odom = 0
    accepted_loop = 0
    backbone_added = 0

    # 3A. Cạnh Odometry nối tiếp (k -> k-1)
    for k in range(1, M):
        sm_src = submaps[k]
        sm_tgt = submaps[k - 1]
        T_prior = np.linalg.inv(sm_tgt.init_world_pose) @ sm_src.init_world_pose

        lg_res = tracker.match_frames(
            src_id=sm_src.anchor_frame,
            tgt_id=sm_tgt.anchor_frame,
            rgb_src=rgb_list[sm_src.anchor_frame],
            depth_src=depth_list[sm_src.anchor_frame],
            rgb_tgt=rgb_list[sm_tgt.anchor_frame],
            depth_tgt=depth_list[sm_tgt.anchor_frame],
            K=K,
            ransac_thresh_m=0.018,
            min_inliers=14,
        )
        T_init = lg_res["T"] if lg_res["ok"] else T_prior

        ok, T_rel, info, stats = _verify_submap_pair_icp(
            sm_src.pcd_local,
            sm_tgt.pcd_local,
            T_init=T_init,
            voxel_size=voxel_size,
            min_fitness=0.38,
            max_rmse=0.012,
            max_cycle_t=0.015,
            max_cycle_r=2.5,
        )

        if ok:
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=k,
                    target_node_id=k - 1,
                    transformation=T_rel,
                    information=info,
                    uncertain=False,
                )
            )
            submap_degree[k] += 1
            submap_degree[k - 1] += 1
            accepted_odom += 1
            edge_logs.append({"src": k, "tgt": k - 1, "type": "odom", "status": "ACCEPT", **stats})
        else:
            T_bridge = lg_res["T"] if lg_res["ok"] else T_prior
            bridge_info = np.eye(6, dtype=np.float64) * 300.0
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=k,
                    target_node_id=k - 1,
                    transformation=T_bridge,
                    information=bridge_info,
                    uncertain=False,
                )
            )
            submap_degree[k] += 1
            submap_degree[k - 1] += 1
            backbone_added += 1
            edge_logs.append({"src": k, "tgt": k - 1, "type": "backbone", "status": "BRIDGE", **stats})

    # 3B. Thu thập ứng viên Loop Closure xa thực sự (lọc chặt chống false loop trên Box)
    loop_candidates = []
    for i in range(loop_min_submap_gap, M):
        sm_src = submaps[i]
        for j in range(0, i - loop_min_submap_gap + 1):
            sm_tgt = submaps[j]

            lg_res = tracker.match_frames(
                src_id=sm_src.anchor_frame,
                tgt_id=sm_tgt.anchor_frame,
                rgb_src=rgb_list[sm_src.anchor_frame],
                depth_src=depth_list[sm_src.anchor_frame],
                rgb_tgt=rgb_list[sm_tgt.anchor_frame],
                depth_tgt=depth_list[sm_tgt.anchor_frame],
                K=K,
                ransac_thresh_m=0.014,
                min_inliers=28,
            )
            if not lg_res["ok"] or lg_res["inlier_ratio"] < 0.35:
                continue

            if rotation_angle_deg(lg_res["T"]) > 65.0:
                continue

            ok, T_rel, info, stats = _verify_submap_pair_icp(
                sm_src.pcd_local,
                sm_tgt.pcd_local,
                T_init=lg_res["T"],
                voxel_size=voxel_size,
                min_fitness=0.52,
                max_rmse=0.0085,
                max_cycle_t=0.010,
                max_cycle_r=1.8,
            )
            if ok:
                score = float(lg_res["num_inliers"]) * stats["fitness"] / max(0.002, stats["rmse"])
                loop_candidates.append(
                    {
                        "score": score,
                        "src": i,
                        "tgt": j,
                        "T_rel": T_rel,
                        "info": info,
                        "lg_inliers": lg_res["num_inliers"],
                        "stats": stats,
                    }
                )

    # Giải phóng ngay VRAM GPU của SuperPoint + LightGlue
    tracker.release()

    # Chỉ giữ tối đa Top-3 cạnh loop chất lượng nhất cho mỗi Submap node
    loop_candidates.sort(key=lambda c: c["score"], reverse=True)
    loop_node_count = defaultdict(int)
    for cand in loop_candidates:
        i, j = cand["src"], cand["tgt"]
        if loop_node_count[i] >= max_loops_per_submap or loop_node_count[j] >= max_loops_per_submap:
            continue
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=i,
                target_node_id=j,
                transformation=cand["T_rel"],
                information=cand["info"],
                uncertain=True,
            )
        )
        loop_node_count[i] += 1
        loop_node_count[j] += 1
        submap_degree[i] += 1
        submap_degree[j] += 1
        accepted_loop += 1

        st = cand["stats"]
        edge_logs.append(
            {
                "src": i,
                "tgt": j,
                "src_frame": submaps[i].anchor_frame,
                "tgt_frame": submaps[j].anchor_frame,
                "type": "loop",
                "status": "ACCEPT",
                "lg_inliers": cand["lg_inliers"],
                **st,
            }
        )
        emit(
            f"   [SubmapPGO] ACCEPT loop SM {i}(f{submaps[i].anchor_frame})->SM {j}(f{submaps[j].anchor_frame}) "
            f"(LG={cand['lg_inliers']}, fit={st['fitness']:.3f}, rmse={st['rmse']*1000:.1f}mm, "
            f"cyc_t={st['cycle_t']*1000:.1f}mm, cyc_r={st['cycle_r']:.2f}°)"
        )

    emit(
        f"[SubmapPGO] Xây xong đồ thị ({M} nodes): odom={accepted_odom}, backbone={backbone_added}, "
        f"loop={accepted_loop}/{len(loop_candidates)} (top-{max_loops_per_submap}/node) trong {time.time() - t0:.2f}s."
    )

    # ── BƯỚC 4: Tối ưu toàn cục Levenberg-Marquardt trên M Submaps ────────────
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 2.5,
        edge_prune_threshold=0.25,
        preference_loop_closure=1.5,
        reference_node=0,
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option,
    )

    opt_submap_poses = [np.asarray(node.pose, dtype=np.float64) for node in pose_graph.nodes]
    max_shift_mm = max(
        float(np.linalg.norm(opt_submap_poses[k][:3, 3] - submaps[k].init_world_pose[:3, 3])) * 1000.0
        for k in range(M)
    )
    emit(f"[SubmapPGO] Global Optimization xong (max_shift={max_shift_mm:.2f}mm). Đang tinh chỉnh từng frame...")

    # ── BƯỚC 5: Lan truyền pose về từng frame + Fine ICP per-frame vào Submap ─
    t_ref = time.time()
    frame_degree: Dict[int, int] = {}
    refined_ok_count = 0

    for k, sm in enumerate(submaps):
        T_sm_opt = opt_submap_poses[k]
        L = len(sm.frame_indices)

        E_boundary = np.eye(4, dtype=np.float64)
        apply_smooth = False
        if k + 1 < M:
            next_anchor = submaps[k + 1].anchor_frame
            T_rel_next_init = np.linalg.inv(sm.init_world_pose) @ init_poses[next_anchor]
            T_pred_next = T_sm_opt @ T_rel_next_init
            E_cand = np.linalg.inv(T_pred_next) @ opt_submap_poses[k + 1]
            if np.linalg.norm(E_cand[:3, 3]) <= 0.06 and rotation_angle_deg(E_cand) <= 15.0:
                E_boundary = E_cand
                apply_smooth = True

        inherited_deg = max(4, submap_degree[k] * 2)
        for idx_in_sm, f_idx in enumerate(sm.frame_indices):
            T_raw = T_sm_opt @ sm.local_frame_poses[f_idx]
            if apply_smooth and L > 0:
                alpha = float(idx_in_sm) / float(L)
                E_alpha = _interpolate_se3(np.eye(4, dtype=np.float64), E_boundary, alpha)
                T_smooth = T_raw @ E_alpha
            else:
                T_smooth = T_raw

            if refine_per_frame and f_idx != sm.anchor_frame:
                T_final, ok_ref = _refine_frame_against_submap(
                    rgb_frame=rgb_list[f_idx],
                    depth_frame=depth_list[f_idx],
                    submap_pcd_local=sm.pcd_local,
                    T_sm_world=T_sm_opt,
                    T_frame_world_init=T_smooth,
                    intrinsic_o3d=intrinsic_o3d,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                    fine_voxel=0.005,
                )
                if ok_ref:
                    refined_ok_count += 1
            else:
                T_final = T_smooth

            optimized_poses[f_idx] = T_final
            frame_degree[f_idx] = inherited_deg

    emit(
        f"[SubmapPGO] Hoàn tất tinh chỉnh {refined_ok_count}/{len(valid_frames)} frames "
        f"trong {time.time() - t_ref:.2f}s. Tổng thời gian PGO: {time.time() - t0:.2f}s."
    )

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "view_graph_edges.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_submaps": M,
                    "num_frames": len(valid_frames),
                    "depth_trunc_m": round(depth_trunc, 3),
                    "submap_size": submap_size,
                    "loop_min_submap_gap": loop_min_submap_gap,
                    "max_loops_per_submap": max_loops_per_submap,
                    "accepted_odom": accepted_odom,
                    "backbone_added": backbone_added,
                    "accepted_loop": accepted_loop,
                    "total_loop_candidates": len(loop_candidates),
                    "per_frame_refined": refined_ok_count,
                    "max_shift_mm": round(max_shift_mm, 3),
                    "elapsed_sec": round(time.time() - t0, 2),
                    "edges": edge_logs,
                },
                f,
                indent=2,
            )

    diagnostics = {
        "components": [len(valid_frames)],
        "node_degree": frame_degree,
        "repaired_weak_nodes": [],
        "accepted_loop_edges": accepted_loop,
        "max_shift_mm": max_shift_mm,
        "depth_trunc_m": depth_trunc,
    }
    return optimized_poses, diagnostics
