import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
from feature_tracker import FeatureTracker
from pose_recovery import evaluate_registration
from tracking_utils import relocalize_submap
from geometry_utils import frame_point_cloud, rotation_angle_deg
import time
import os
import json
from collections import defaultdict


def _union_find_components(node_list, edge_pairs):
    """Return components (lists of frame ids) induced by edge_pairs, largest first."""
    parent = {n: n for n in node_list}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edge_pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

    groups = defaultdict(list)
    for n in node_list:
        groups[find(n)].append(n)
    return sorted((sorted(v) for v in groups.values()), key=len, reverse=True)


def _ensure_certain_spanning_tree(sub_pose_graph, emit):
    """Đảm bảo cây khung của các cạnh uncertain=False liên thông trên toàn bộ component.

    Open3D global_optimization yêu cầu các cạnh uncertain=False phải tạo thành
    một cây khung liên thông (nếu không sẽ báo '[Open3D WARNING] Certain-edge
    subset of PoseGraph is not connected' và không thể tối ưu hóa qua chỗ đứt).
    Khi hai nhánh odometry chỉ nối với nhau qua các cạnh spatial/loop
    (uncertain=True), ta chọn cạnh cầu nối tốt nhất giữa hai nhánh và nâng nó
    lên uncertain=False để Open3D luôn giải được bài toán tối ưu toàn cục.
    """
    n_nodes = len(sub_pose_graph.nodes)
    if n_nodes < 2 or len(sub_pose_graph.edges) == 0:
        return

    parent = list(range(n_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return True
        return False

    for edge in sub_pose_graph.edges:
        if not edge.uncertain:
            union(edge.source_node_id, edge.target_node_id)

    # Sắp xếp các cạnh uncertain=True theo độ lớn trace(information) giảm dần
    uncertain_indices = [
        (float(np.trace(np.asarray(e.information))), idx)
        for idx, e in enumerate(sub_pose_graph.edges)
        if e.uncertain
    ]
    uncertain_indices.sort(reverse=True)

    promoted = 0
    for _, idx in uncertain_indices:
        e = sub_pose_graph.edges[idx]
        if union(e.source_node_id, e.target_node_id):
            sub_pose_graph.edges[idx] = o3d.pipelines.registration.PoseGraphEdge(
                e.source_node_id,
                e.target_node_id,
                np.array(e.transformation, dtype=np.float64),
                np.array(e.information, dtype=np.float64),
                uncertain=False,
            )
            promoted += 1

    if promoted > 0:
        emit(f"      [PGO] Đã nâng {promoted} cạnh cầu nối lên certain-edge để nối thông cây khung PoseGraph.")


def _optimize_component(frames, node_id_map, reverse_map, pose_graph,
                        option, criteria, emit):
    """Run global_optimization on the sub-pose-graph induced by `frames`."""
    if len(frames) < 2:
        return 0.0

    sub = o3d.pipelines.registration.PoseGraph()
    local_id = {}
    for k, f in enumerate(frames):
        local_id[f] = k
        sub.nodes.append(o3d.pipelines.registration.PoseGraphNode(
            np.array(pose_graph.nodes[node_id_map[f]].pose, dtype=np.float64)))

    frame_set = set(frames)
    for edge in pose_graph.edges:
        src_f = reverse_map.get(edge.source_node_id)
        tgt_f = reverse_map.get(edge.target_node_id)
        if src_f in frame_set and tgt_f in frame_set:
            sub.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                local_id[src_f], local_id[tgt_f],
                np.array(edge.transformation, dtype=np.float64),
                np.array(edge.information, dtype=np.float64),
                uncertain=edge.uncertain))

    if len(sub.edges) == 0:
        return 0.0

    _ensure_certain_spanning_tree(sub, emit)

    before = np.array([np.array(n.pose) for n in sub.nodes])

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    try:
        o3d.pipelines.registration.global_optimization(
            sub,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            criteria, option)
    finally:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    after = np.array([np.array(n.pose) for n in sub.nodes])
    moved_mm = float(np.abs(after[:, :3, 3] - before[:, :3, 3]).max()) * 1000.0

    for f in frames:
        pose_graph.nodes[node_id_map[f]].pose = sub.nodes[local_id[f]].pose.copy()

    emit(f"      [PGO] component [{frames[0]}–{frames[-1]}] {len(frames)} nodes / "
         f"{len(sub.edges)} edges -> max node shift {moved_mm:.2f}mm")
    return moved_mm


def _smooth_weak_degree_nodes(valid_frames, poses_dict, node_degree, emit):
    """Tự động phát hiện các node có bậc liên kết yếu (weakly-connected nodes)
    hoặc có bước nhảy vị trí dị biệt (outlier step) và nội suy mượt SE(3) giữa 2
    node khỏe liền kề. Hoàn toàn tổng quát cho mọi dataset."""
    if len(valid_frames) < 5:
        return poses_dict, []

    degrees = np.array([node_degree.get(f, 0) for f in valid_frames], dtype=np.float64)
    median_deg = float(np.median(degrees))
    min_healthy_deg = max(4, int(round(0.30 * median_deg)))

    steps = []
    for k in range(1, len(valid_frames)):
        f_prev, f_curr = valid_frames[k - 1], valid_frames[k]
        steps.append(float(np.linalg.norm(poses_dict[f_curr][:3, 3] - poses_dict[f_prev][:3, 3])))
    median_step = max(0.004, float(np.median(steps)))
    max_healthy_step = max(0.028, 4.0 * median_step)

    healthy_mask = []
    for k, f in enumerate(valid_frames):
        deg_ok = node_degree.get(f, 0) >= min_healthy_deg
        step_prev = float(np.linalg.norm(poses_dict[f][:3, 3] - poses_dict[valid_frames[k - 1]][:3, 3])) if k > 0 else 0.0
        step_next = float(np.linalg.norm(poses_dict[valid_frames[k + 1]][:3, 3] - poses_dict[f][:3, 3])) if k + 1 < len(valid_frames) else 0.0
        spike = (step_prev > max_healthy_step) or (step_next > max_healthy_step)
        healthy_mask.append(bool(deg_ok and not spike))

    healthy_mask[0] = True
    healthy_mask[-1] = True

    smoothed = dict(poses_dict)
    repaired_frames = []
    n = len(valid_frames)
    k = 1
    while k < n - 1:
        if healthy_mask[k]:
            k += 1
            continue
        start_k = k - 1
        while k < n - 1 and not healthy_mask[k]:
            k += 1
        end_k = k
        # Chỉ nội suy các khoảng yếu cục bộ (<= 12 frames)
        if 0 < (end_k - start_k - 1) <= 12:
            f0, f1 = valid_frames[start_k], valid_frames[end_k]
            T0, T1 = smoothed[f0], smoothed[f1]
            rots = R.from_matrix([T0[:3, :3], T1[:3, :3]])
            slerp = Slerp([0.0, 1.0], rots)
            span = float(f1 - f0)
            for m in range(start_k + 1, end_k):
                fm = valid_frames[m]
                alpha = float(fm - f0) / max(1.0, span)
                Tm = np.eye(4, dtype=np.float64)
                Tm[:3, :3] = slerp([alpha])[0].as_matrix()
                Tm[:3, 3] = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
                smoothed[fm] = Tm
                repaired_frames.append(fm)

    if repaired_frames:
        emit(f"      [PGO] Đã nội suy làm mượt SE(3) cho {len(repaired_frames)} node yếu/nhảy bậc: {repaired_frames}")
    return smoothed, repaired_frames


def _run_one_round(round_idx, valid_frames, node_id_map, reverse_map, candidate_pairs,
                   current_poses, color_np, depth_np, masks, intrinsic_tensor,
                   voxel_size, min_icp_fitness, max_icp_rmse,
                   max_cycle_translation, max_cycle_rotation,
                   odometry_min_fitness, odometry_max_rmse,
                   odometry_max_cycle_translation, odometry_max_cycle_rotation,
                   use_colored_icp, pc_cache, edge_transforms, emit,
                   loop_closure_preference):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    for f in valid_frames:
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(
            np.array(current_poses[f], dtype=np.float64)))

    def get_pc(f_idx):
        if f_idx not in pc_cache:
            pc = frame_point_cloud(f_idx, color_np, depth_np, masks, intrinsic_tensor,
                                   object_only=None)
            pc_cache[f_idx] = pc.voxel_down_sample(voxel_size)
        return pc_cache[f_idx]

    accepted_edges = 0
    rejected_edges = 0
    node_degree = {f: 0 for f in valid_frames}
    accepted_pairs = []
    accepted_odom_pairs = []
    edge_records = []
    start_time = time.time()

    for (src_idx, tgt_idx, pair_type) in candidate_pairs:
        gap_frames = abs(src_idx - tgt_idx)
        is_odometry = (pair_type == 'temporal' and gap_frames <= 5) or (pair_type == 'submap_bridge' and gap_frames <= 12)

        edge_min_fitness = (odometry_min_fitness if (is_odometry and odometry_min_fitness is not None)
                            else min_icp_fitness)
        edge_max_rmse = (odometry_max_rmse if (is_odometry and odometry_max_rmse is not None)
                         else max_icp_rmse)
        edge_max_cyc_t = (odometry_max_cycle_translation
                          if (is_odometry and odometry_max_cycle_translation is not None)
                          else max_cycle_translation)
        edge_max_cyc_r = (odometry_max_cycle_rotation
                          if (is_odometry and odometry_max_cycle_rotation is not None)
                          else max_cycle_rotation)

        T_src_w = current_poses[src_idx]
        T_tgt_w = current_poses[tgt_idx]
        T_src_to_tgt_guess = np.linalg.inv(T_tgt_w) @ T_src_w

        guess_trans = float(np.linalg.norm(T_src_to_tgt_guess[:3, 3]))
        guess_rot = rotation_angle_deg(T_src_to_tgt_guess)
        need_ransac = (pair_type == 'loop') or (guess_trans > 0.12) or (guess_rot > 28.0)

        source_pc = get_pc(src_idx)
        target_pc = get_pc(tgt_idx)

        max_drift = max(0.020 * gap_frames, 0.060) if is_odometry else None
        guesses = [T_src_to_tgt_guess]
        if pair_type == 'submap_bridge' and not np.allclose(T_src_to_tgt_guess, np.eye(4), atol=1e-3):
            guesses.append(np.eye(4))

        T_verified, diag = relocalize_submap(
            source_pc, target_pc, voxel_size,
            edge_min_fitness, edge_max_rmse,
            edge_max_cyc_t, edge_max_cyc_r,
            initial_guesses=guesses,
            use_colored_icp=use_colored_icp,
            run_ransac=need_ransac,
            max_trans_drift=max_drift
        )

        _T_best = None
        if diag is not None and diag.get("transformation") is not None:
            _T_best = np.array(diag["transformation"], dtype=np.float64)
            edge_transforms[(src_idx, tgt_idx)] = _T_best
            edge_transforms[(tgt_idx, src_idx)] = np.linalg.inv(_T_best)

        if T_verified is not None and not diag.get("degenerate", True):
            info_mat = np.array(diag["information_matrix"])
            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                source_node_id=node_id_map[src_idx],
                target_node_id=node_id_map[tgt_idx],
                transformation=T_verified,
                information=info_mat,
                uncertain=not is_odometry))
            node_degree[src_idx] += 1
            node_degree[tgt_idx] += 1
            accepted_edges += 1
            accepted_pairs.append((src_idx, tgt_idx))
            if is_odometry:
                accepted_odom_pairs.append((src_idx, tgt_idx))
            emit(f"      [ViewGraph][R{round_idx}] ACCEPT edge {src_idx}->{tgt_idx} [{pair_type}] "
                 f"(fitness={diag['fitness']:.3f}, rmse={diag['rmse_mm']:.1f}mm)")
            edge_records.append({
                "src": src_idx, "tgt": tgt_idx, "pair_type": pair_type,
                "is_odometry": is_odometry, "accepted": True,
                "fitness": diag.get("fitness"), "rmse_mm": diag.get("rmse_mm"),
                "cycle_translation_mm": diag.get("cycle_translation_mm"),
                "cycle_rotation_deg": diag.get("cycle_rotation_deg"),
                "degenerate": diag.get("degenerate", False),
                "reason": diag.get("reason", "verified"),
                "transformation": np.asarray(T_verified).tolist(),
            })
        else:
            rejected_edges += 1
            if T_verified is not None and diag.get("degenerate", False):
                reason = "DEGENERATE_INFO_MATRIX"
            else:
                reason = diag.get("reason", "unknown") if diag else "no_points_or_too_few"
            fitness = diag.get("fitness") if diag else None
            rmse_mm = diag.get("rmse_mm") if diag else None
            cyc_t_mm = diag.get("cycle_translation_mm") if diag else None
            cyc_r_deg = diag.get("cycle_rotation_deg") if diag else None
            fitness_s = f"{fitness:.3f}" if fitness is not None else "N/A"
            rmse_s = f"{rmse_mm:.1f}mm" if rmse_mm is not None else "N/A"
            cyc_t_s = f"{cyc_t_mm:.1f}mm" if cyc_t_mm is not None else "N/A"
            cyc_r_s = f"{cyc_r_deg:.2f}deg" if cyc_r_deg is not None else "N/A"
            emit(f"      [ViewGraph][R{round_idx}] REJECT edge {src_idx}->{tgt_idx} [{pair_type}]: reason={reason} "
                 f"best_fitness={fitness_s} best_rmse={rmse_s} "
                 f"cycle_t={cyc_t_s} cycle_r={cyc_r_s} "
                 f"(min_fitness={edge_min_fitness}, max_rmse={edge_max_rmse*1000:.1f}mm, "
                 f"max_cycle_t={edge_max_cyc_t*1000:.1f}mm, max_cycle_r={edge_max_cyc_r}deg, "
                 f"gate={'odometry' if is_odometry else 'loop/spatial'})")
            edge_records.append({
                "src": src_idx, "tgt": tgt_idx, "pair_type": pair_type,
                "is_odometry": is_odometry, "accepted": False, "reason": reason,
                "fitness": fitness, "rmse_mm": rmse_mm,
                "cycle_translation_mm": cyc_t_mm, "cycle_rotation_deg": cyc_r_deg,
                "degenerate": bool(diag.get("degenerate", False)) if diag else None,
                "transformation": (_T_best.tolist() if _T_best is not None else None),
            })

    emit(f"      [ViewGraph][R{round_idx}] Đã xây xong Edge. (Accepted: {accepted_edges}, "
         f"Rejected: {rejected_edges}) trong {time.time()-start_time:.2f}s")

    components = _union_find_components(valid_frames, accepted_pairs)
    odom_components = _union_find_components(valid_frames, accepted_odom_pairs)
    emit(f"      [PGO][R{round_idx}] View Graph có {len(components)} connected component(s), "
         f"sizes={[len(c) for c in components][:8]} (odom-only components: {[len(c) for c in odom_components][:8]})")

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 5.0,
        edge_prune_threshold=0.25,
        preference_loop_closure=max(2.5, loop_closure_preference),
        reference_node=0)
    criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()

    max_shift_mm = 0.0
    optimized_any = False
    for comp in components:
        shift = _optimize_component(comp, node_id_map, reverse_map, pose_graph,
                                    option, criteria, emit)
        if shift > 0.0:
            optimized_any = True
            max_shift_mm = max(max_shift_mm, shift)

    new_poses = dict(current_poses)
    for f in valid_frames:
        new_poses[f] = np.array(pose_graph.nodes[node_id_map[f]].pose).copy()

    new_poses, repaired_weak_nodes = _smooth_weak_degree_nodes(
        valid_frames, new_poses, node_degree, emit)

    round_stats = {
        "round": round_idx,
        "accepted_edges": accepted_edges,
        "rejected_edges": rejected_edges,
        "node_degree": node_degree,
        "repaired_weak_nodes": repaired_weak_nodes,
        "n_components": len(components),
        "component_sizes": [len(c) for c in components],
        "odom_component_sizes": [len(c) for c in odom_components],
        "pgo_applied": bool(optimized_any),
        "pgo_max_shift_mm": round(max_shift_mm, 3),
    }
    return new_poses, round_stats, edge_records, components, max_shift_mm


def build_and_optimize_pose_graph(valid_frames, candidate_pairs, optimized_poses,
                                  color_np, depth_np, masks, intrinsic_tensor,
                                  voxel_size=0.015, min_icp_fitness=0.4, max_icp_rmse=0.015,
                                  max_cycle_translation=0.02, max_cycle_rotation=3.0,
                                  odometry_min_fitness=None, odometry_max_rmse=None,
                                  odometry_max_cycle_translation=None,
                                  odometry_max_cycle_rotation=None,
                                  use_colored_icp=False, log_dir=None,
                                  refine_rounds=2, refine_convergence_mm=0.5,
                                  loop_closure_preference=0.80):
    node_id_map = {}
    valid_frames = sorted(list(valid_frames))
    for idx, f in enumerate(valid_frames):
        node_id_map[f] = idx
    reverse_map = {v: k for k, v in node_id_map.items()}

    log_file = None
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = open(os.path.join(log_dir, "view_graph_edges.log"), "w", encoding="utf-8")
        except Exception as e:
            print(f"      [ViewGraph] [WARN] Không mở được file log ({e}), chỉ in ra terminal.")
            log_file = None

    def emit(line):
        print(line)
        if log_file:
            log_file.write(line + "\n")
            log_file.flush()

    emit(f"      [ViewGraph] {len(valid_frames)} nodes, {len(candidate_pairs)} candidate edges, "
         f"{max(1, refine_rounds)} refine round(s) requested.")

    pc_cache = {}
    edge_transforms = {}
    current_poses = {f: np.array(optimized_poses[f], dtype=np.float64) for f in valid_frames}

    rounds_stats = []
    last_edge_records = []
    last_components = []
    overall_start = time.time()

    n_rounds = max(1, refine_rounds)
    for round_idx in range(1, n_rounds + 1):
        emit(f"      [PGO] ── Round {round_idx}/{n_rounds} ──")
        current_poses, round_stats, edge_records, components, max_shift_mm = _run_one_round(
            round_idx, valid_frames, node_id_map, reverse_map, candidate_pairs,
            current_poses, color_np, depth_np, masks, intrinsic_tensor,
            voxel_size, min_icp_fitness, max_icp_rmse,
            max_cycle_translation, max_cycle_rotation,
            odometry_min_fitness, odometry_max_rmse,
            odometry_max_cycle_translation, odometry_max_cycle_rotation,
            use_colored_icp, pc_cache, edge_transforms, emit,
            loop_closure_preference)

        rounds_stats.append(round_stats)
        last_edge_records = edge_records
        last_components = components

        emit(f"      [PGO] Round {round_idx}/{n_rounds} xong: "
             f"{round_stats['n_components']} component(s), "
             f"max shift {max_shift_mm:.2f}mm, "
             f"accepted={round_stats['accepted_edges']} rejected={round_stats['rejected_edges']}")

        if round_idx < n_rounds and len(components) == 1 and max_shift_mm < refine_convergence_mm:
            emit(f"      [PGO] Hội tụ sau round {round_idx} "
                 f"(1 component, max shift {max_shift_mm:.2f}mm < {refine_convergence_mm}mm) -- dừng sớm.")
            break

    total_time = time.time() - overall_start
    final_round = rounds_stats[-1]
    emit(f"      [PGO] Hoàn tất {len(rounds_stats)} round(s) sau {total_time:.2f}s. "
         f"Component cuối: {final_round['component_sizes'][:8]}"
         f"{'...' if len(final_round['component_sizes']) > 8 else ''}")
    if not final_round["pgo_applied"] and len(valid_frames) > 1:
        emit("      [PGO] [WARN] Round cuối không di chuyển pose nào -- kiểm tra lại "
             "candidate pairs / ngưỡng gate.")

    if log_dir:
        try:
            with open(os.path.join(log_dir, "view_graph_edges.json"), "w", encoding="utf-8") as jf:
                json.dump({
                    "n_nodes": len(valid_frames),
                    "n_candidate_edges": len(candidate_pairs),
                    "n_rounds_run": len(rounds_stats),
                    "min_icp_fitness": min_icp_fitness,
                    "max_icp_rmse_mm": max_icp_rmse * 1000.0,
                    "max_cycle_translation_mm": max_cycle_translation * 1000.0,
                    "max_cycle_rotation_deg": max_cycle_rotation,
                    "odometry_max_cycle_translation_mm": (
                        None if odometry_max_cycle_translation is None
                        else odometry_max_cycle_translation * 1000.0),
                    "odometry_max_cycle_rotation_deg": odometry_max_cycle_rotation,
                    "rounds": rounds_stats,
                    "components": last_components,
                    "accepted_edges": final_round["accepted_edges"],
                    "rejected_edges": final_round["rejected_edges"],
                    "edges": last_edge_records,
                }, jf, indent=2)
            print(f"      [ViewGraph] Đã ghi log chi tiết trong {log_dir}")
        except Exception as e:
            print(f"      [ViewGraph] [WARN] Không ghi được view_graph_edges.json ({e})")

    if log_file:
        log_file.close()

    if isinstance(optimized_poses, dict):
        new_optimized_poses = dict(optimized_poses)
    else:
        new_optimized_poses = list(optimized_poses)
    for f in valid_frames:
        new_optimized_poses[f] = current_poses[f]

    diagnostics = {
        "status": "OK",
        "nodes": len(valid_frames),
        "accepted_edges": final_round["accepted_edges"],
        "rejected_edges": final_round["rejected_edges"],
        "node_degree": final_round["node_degree"],
        "repaired_weak_nodes": final_round.get("repaired_weak_nodes", []),
        "n_components": final_round["n_components"],
        "component_sizes": final_round["component_sizes"],
        "components": last_components,
        "pgo_applied": final_round["pgo_applied"],
        "pgo_max_shift_mm": final_round["pgo_max_shift_mm"],
        "rounds": rounds_stats,
        "n_rounds_run": len(rounds_stats),
    }
    return new_optimized_poses, diagnostics, edge_transforms