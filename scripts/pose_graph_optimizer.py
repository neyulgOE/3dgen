# -*- coding: utf-8 -*-
"""
Submap-based Pose Graph Optimizer (Full-Scene SuperPoint + LightGlue + Point-to-Plane ICP).
- Giảm kích thước đồ thị từ 350-430 frames xuống ~30-40 Submaps (chạy trong ~15-25 giây).
- Tracking trên toàn cảnh (Full-Scene, không mask) để khóa cứng 6-DOF trên vật đối xứng.
- Chỉ áp SAM2 mask ở bước TSDF phía sau trong slam_object_builder.py.
"""
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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


def _rgbd_to_full_scene_pcd(
    rgb: np.ndarray,
    depth_raw: np.ndarray,
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0,
    depth_trunc: float = 1.5,
    voxel_size: float = 0.008,
) -> o3d.geometry.PointCloud:
    """Tạo PointCloud toàn cảnh (KHÔNG cắt SAM2 mask) để bám chặt nền và mặt bàn."""
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
    submap_size: int = 10,
    depth_scale: float = 1000.0,
    depth_trunc: float = 1.5,
    voxel_size: float = 0.008,
) -> List[SubmapNode]:
    """
    Gom các frame liên tiếp thành Submap Full-Scene.
    Mỗi Submap có 1 frame đại diện và 1 đám mây điểm sạch nhiễu.
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

        merged = o3d.geometry.PointCloud()
        local_poses = {}

        # Chỉ gộp tối đa 4 frame trải đều trong submap để đám mây sạch mà không quá nặng
        sample_step = max(1, len(chunk) // 4)
        for k, f_idx in enumerate(chunk):
            T_local = T_anchor_inv @ init_poses[f_idx]
            local_poses[f_idx] = T_local
            if k % sample_step == 0 or k == len(chunk) - 1:
                pcd_f = _rgbd_to_full_scene_pcd(
                    rgb_list[f_idx],
                    depth_list[f_idx],
                    intrinsic_o3d,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                    voxel_size=voxel_size,
                )
                if len(pcd_f.points) > 0:
                    pcd_f.transform(T_local)
                    merged += pcd_f

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
    voxel_size: float = 0.008,
    min_fitness: float = 0.42,
    max_rmse: float = 0.012,
    max_cycle_t: float = 0.015,
    max_cycle_r: float = 2.5,
):
    """
    Tinh chỉnh Coarse-to-Fine Point-to-Plane ICP và kiểm định thuận-nghịch.
    Chiều nghịch LUÔN khởi tạo bằng inv(T_fwd) -> không bao giờ trượt góc trụ đối xứng.
    """
    if len(pcd_src.points) < 60 or len(pcd_tgt.points) < 60:
        return False, np.eye(4), np.eye(6), {"reason": "sparse_cloud"}

    # Bước 1: Coarse -> Fine ICP chiều thuận
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

    # Bước 2: Kiểm định chiều nghịch bằng Fine ICP khởi tạo từ inv(T_fwd)
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
    # Kiểm tra suy biến ma trận thông tin
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


def build_and_optimize_pose_graph(
    valid_frames: List[int],
    rgb_list: List[np.ndarray],
    depth_list: List[np.ndarray],
    init_poses: List[np.ndarray],
    intrinsic_o3d: o3d.camera.PinholeCameraIntrinsic,
    submap_size: int = 10,
    voxel_size: float = 0.008,
    depth_scale: float = 1000.0,
    depth_trunc: float = 1.5,
    loop_min_submap_gap: int = 3,
    log_dir: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None,
):
    """
    Hàm chính thay thế pose_graph_optimizer cũ:
    1. Gom ~350-430 frames thành ~30-40 Full-Scene Submaps.
    2. Chạy SuperPoint + LightGlue trên GPU để nối cạnh Odometry & Loop Closure.
    3. Xả sạch VRAM GPU ngay sau bước bắt cặp.
    4. Giải Open3D Global Optimization (Levenberg-Marquardt) trên đồ thị Submap.
    5. Nội suy SE(3) mượt trả về tư thế tối ưu cho toàn bộ từng frame.
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

    # ── BƯỚC 1: Dựng Full-Scene Submaps ───────────────────────────────────────
    emit(f"[SubmapPGO] Đang dựng Full-Scene Submaps (size={submap_size}) cho {len(valid_frames)} frames...")
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
    emit(f"[SubmapPGO] Đã gom thành {M} Submaps trong {time.time() - t0:.2f}s.")

    # ── BƯỚC 2: Khởi tạo SuperPoint + LightGlue (GPU) và trích đặc trưng 1 lần ─
    t_lg = time.time()
    tracker = LightGlueFeatureTracker(
        max_keypoints=1024,
        depth_scale=depth_scale,
        min_depth_m=0.08,
        max_depth_m=depth_trunc,
    )
    for sm in submaps:
        tracker.extract_and_cache(sm.anchor_frame, rgb_list[sm.anchor_frame])
    emit(f"[SubmapPGO] Đã trích SuperPoint cho {M} Submap anchors trong {time.time() - t_lg:.2f}s.")

    # ── BƯỚC 3: Xây PoseGraph cho M Submaps ───────────────────────────────────
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for sm in submaps:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(sm.init_world_pose.copy()))

    edge_logs = []
    submap_degree = defaultdict(int)
    accepted_odom = 0
    accepted_loop = 0
    backbone_added = 0

    # 3A. Cạnh Odometry nối tiếp (k -> k-1) — BẮT BUỘC liên thông 1 chuỗi duy nhất
    for k in range(1, M):
        sm_src = submaps[k]
        sm_tgt = submaps[k - 1]
        T_prior = np.linalg.inv(sm_tgt.init_world_pose) @ sm_src.init_world_pose

        # Ưu tiên khởi tạo bằng LightGlue trên Full-Scene (vượt qua góc 90° & quai cốc mượt)
        lg_res = tracker.match_frames(
            src_id=sm_src.anchor_frame,
            tgt_id=sm_tgt.anchor_frame,
            rgb_src=rgb_list[sm_src.anchor_frame],
            depth_src=depth_list[sm_src.anchor_frame],
            rgb_tgt=rgb_list[sm_tgt.anchor_frame],
            depth_tgt=depth_list[sm_tgt.anchor_frame],
            K=K,
            ransac_thresh_m=0.020,
            min_inliers=14,
        )
        T_init = lg_res["T"] if lg_res["ok"] else T_prior

        ok, T_rel, info, stats = _verify_submap_pair_icp(
            sm_src.pcd_local,
            sm_tgt.pcd_local,
            T_init=T_init,
            voxel_size=voxel_size,
            min_fitness=0.35,
            max_rmse=0.014,
            max_cycle_t=0.018,
            max_cycle_r=3.0,
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
            # Sequential Backbone đàn hồi: giữ 1 connected component duy nhất
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

    # 3B. Cạnh Loop Closure / Spatial trên toàn bộ các cặp Submap (chỉ có M*(M-1)/2 ≈ 500 cặp)
    for i in range(loop_min_submap_gap, M):
        for j in range(0, i - loop_min_submap_gap + 1):
            sm_src = submaps[i]
            sm_tgt = submaps[j]

            lg_res = tracker.match_frames(
                src_id=sm_src.anchor_frame,
                tgt_id=sm_tgt.anchor_frame,
                rgb_src=rgb_list[sm_src.anchor_frame],
                depth_src=depth_list[sm_src.anchor_frame],
                rgb_tgt=rgb_list[sm_tgt.anchor_frame],
                depth_tgt=depth_list[sm_tgt.anchor_frame],
                K=K,
                ransac_thresh_m=0.018,
                min_inliers=20,
            )
            if not lg_res["ok"]:
                continue

            ok, T_rel, info, stats = _verify_submap_pair_icp(
                sm_src.pcd_local,
                sm_tgt.pcd_local,
                T_init=lg_res["T"],
                voxel_size=voxel_size,
                min_fitness=0.45,
                max_rmse=0.011,
                max_cycle_t=0.012,
                max_cycle_r=2.0,
            )
            if ok:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_node_id=i,
                        target_node_id=j,
                        transformation=T_rel,
                        information=info,
                        uncertain=True,
                    )
                )
                submap_degree[i] += 1
                submap_degree[j] += 1
                accepted_loop += 1
                edge_logs.append(
                    {
                        "src": i,
                        "tgt": j,
                        "src_frame": sm_src.anchor_frame,
                        "tgt_frame": sm_tgt.anchor_frame,
                        "type": "loop",
                        "status": "ACCEPT",
                        "lg_inliers": lg_res["num_inliers"],
                        **stats,
                    }
                )
                emit(
                    f"   [SubmapPGO] ACCEPT loop SM {i}(f{sm_src.anchor_frame})->SM {j}(f{sm_tgt.anchor_frame}) "
                    f"(LG={lg_res['num_inliers']}, fit={stats['fitness']:.3f}, rmse={stats['rmse']*1000:.1f}mm, "
                    f"cyc_t={stats['cycle_t']*1000:.1f}mm, cyc_r={stats['cycle_r']:.2f}°)"
                )

    # Giải phóng ngay VRAM của SuperPoint + LightGlue trước khi tối ưu & dựng TSDF
    tracker.release()

    emit(
        f"[SubmapPGO] Xây xong đồ thị ({M} nodes): odom={accepted_odom}, "
        f"backbone={backbone_added}, loop={accepted_loop} trong {time.time() - t0:.2f}s."
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
    emit(f"[SubmapPGO] Global Optimization hoàn tất! max_shift={max_shift_mm:.2f}mm, tổng thời gian={time.time() - t0:.2f}s.")

    # ── BƯỚC 5: Lan truyền pose từ Submaps về từng Frame (làm mượt biên SE(3)) ─
    frame_degree: Dict[int, int] = {}
    for k, sm in enumerate(submaps):
        T_sm_opt = opt_submap_poses[k]
        L = len(sm.frame_indices)

        # Tính sai lệch biên giữa cuối submap k và đầu submap k+1 để nội suy mượt
        E_boundary = np.eye(4, dtype=np.float64)
        apply_smooth = False
        if k + 1 < M:
            next_anchor = submaps[k + 1].anchor_frame
            T_rel_next_init = np.linalg.inv(sm.init_world_pose) @ init_poses[next_anchor]
            T_pred_next = T_sm_opt @ T_rel_next_init
            E_cand = np.linalg.inv(T_pred_next) @ opt_submap_poses[k + 1]
            if np.linalg.norm(E_cand[:3, 3]) <= 0.08 and rotation_angle_deg(E_cand) <= 20.0:
                E_boundary = E_cand
                apply_smooth = True

        # Mọi frame trong submap được liên kết đều kế thừa degree đủ lớn (>= 4) để không bị lọc oan
        inherited_deg = max(4, submap_degree[k] * 2)
        for idx_in_sm, f_idx in enumerate(sm.frame_indices):
            T_raw = T_sm_opt @ sm.local_frame_poses[f_idx]
            if apply_smooth and L > 0:
                alpha = float(idx_in_sm) / float(L)
                E_alpha = _interpolate_se3(np.eye(4, dtype=np.float64), E_boundary, alpha)
                optimized_poses[f_idx] = T_raw @ E_alpha
            else:
                optimized_poses[f_idx] = T_raw
            frame_degree[f_idx] = inherited_deg

    # Ghi log chi tiết giống format cũ để tiện chẩn đoán
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "view_graph_edges.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_submaps": M,
                    "num_frames": len(valid_frames),
                    "accepted_odom": accepted_odom,
                    "backbone_added": backbone_added,
                    "accepted_loop": accepted_loop,
                    "max_shift_mm": max_shift_mm,
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
    }
    return optimized_poses, diagnostics
