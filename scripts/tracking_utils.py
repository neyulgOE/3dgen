from geometry_utils import *
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

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def _run_cotracker_recovery(dataset_dir, n_points, min_inliers, inlier_thr_m, optimized_poses):
    import subprocess
    print("   -> Running CoTracker Offline Recovery (Phase 8)...")

    _scripts_dir = os.path.dirname(os.path.abspath(__file__))

    sparse_script = os.path.join(_scripts_dir, "sparse_tracker.py")
    cmd_sparse = [
        sys.executable, sparse_script, dataset_dir,
        "--n-points", str(n_points)
    ]
    print(f"      Executing: {' '.join(cmd_sparse)}")
    res1 = subprocess.run(cmd_sparse, cwd=_scripts_dir)
    if res1.returncode != 0:
        print(f"      [WARN] sparse_tracker failed with code {res1.returncode}")
        return {}
        
    pose_script = os.path.join(_scripts_dir, "pose_recovery.py")
    cmd_pose = [
        sys.executable, pose_script, dataset_dir,
        "--min-inliers", str(min_inliers),
        "--inlier-thr-mm", str(inlier_thr_m * 1000.0)
    ]
    print(f"      Executing: {' '.join(cmd_pose)}")
    res2 = subprocess.run(cmd_pose, cwd=_scripts_dir)
    if res2.returncode != 0:
        print(f"      [WARN] pose_recovery failed with code {res2.returncode}")
        return {}
        
    rec_path = os.path.join(dataset_dir, "recovered_poses.json")
    if not os.path.exists(rec_path):
        return {}
        
    with open(rec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    recovered = {}
    for fstr, info in data.get("frames", {}).items():
        target_fidx = int(fstr)
        anchor_fidx = info["anchor_frame"]
        T_anchor_to_target = np.array(info["T_anchor_to_target"])
        
        if 0 <= anchor_fidx < len(optimized_poses):
            T_target_to_anchor = np.linalg.inv(T_anchor_to_target)
            T_c2w_anchor = optimized_poses[anchor_fidx]
            T_c2w_target = T_c2w_anchor @ T_target_to_anchor
            recovered[target_fidx] = T_c2w_target
            
    return recovered

def select_motion_keyframes(indices, poses, max_views, min_translation_m, min_rotation_deg,
                            max_frame_gap=15):
    if not indices:
        return []
    candidates = [indices[0]]
    last = indices[0]
    last_kept_pos = 0
    for pos, index in enumerate(indices[1:], start=1):
        relative = np.linalg.inv(poses[last]) @ poses[index]
        moved = np.linalg.norm(relative[:3, 3]) >= min_translation_m
        turned = rotation_angle_deg(relative) >= min_rotation_deg
        stalled = (pos - last_kept_pos) >= max_frame_gap
        if moved or turned or stalled:
            candidates.append(index)
            last = index
            last_kept_pos = pos

    if candidates[-1] != indices[-1]:
        candidates.append(indices[-1])
    if len(candidates) <= max_views:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, max_views).round().astype(int)
    return [candidates[p] for p in np.unique(positions)]

def compute_feature_metrics(img1, img2, mask1, mask2):
    orb = cv2.ORB_create(nfeatures=2000)
    m1 = (mask1 * 255).astype(np.uint8) if mask1 is not None else None
    m2 = (mask2 * 255).astype(np.uint8) if mask2 is not None else None
    kp1, des1 = orb.detectAndCompute(img1, m1)
    kp2, des2 = orb.detectAndCompute(img2, m2)
    num_features = len(kp1)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return num_features, 0, 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    num_matches = len(matches)
    if num_matches < 4:
        return num_features, num_matches, 0, 0.0
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    _, inlier_mask = cv2.findEssentialMat(
        pts1, pts2, focal=615.0, pp=(320.0, 240.0),
        method=cv2.RANSAC, prob=0.999, threshold=1.0)
    num_inliers = int(np.sum(inlier_mask)) if inlier_mask is not None else 0
    inlier_ratio = round(num_inliers / num_features, 4) if num_features > 0 else 0.0
    return num_features, num_matches, num_inliers, inlier_ratio

def refine_pose_against_scene_map(frame_index, initial_pose_guess, scene_history, color_np, depth_np, masks, intrinsic_tensor, voxel_size=0.0025, object_only=False, use_colored_icp=False):
    """Refine camera→world pose against recent full-scene clouds using Tensor ICP."""
    if len(scene_history) < 2:
        return initial_pose_guess, {"used": False, "reason": "scene_map_warmup"}
        
    source = frame_point_cloud(frame_index, color_np, depth_np, masks, intrinsic_tensor,
                               object_only=object_only).voxel_down_sample(voxel_size)
                               
    target = None
    for cloud in scene_history[-10:]:
        if target is None:
            target = cloud.clone()
        else:
            target = target.append(cloud)
    if target is None:
        return initial_pose_guess, {"used": False, "reason": "scene_map_warmup"}
    target = target.voxel_down_sample(voxel_size)
    
    if len(source.point.positions) < 100 or len(target.point.positions) < 300:
        return initial_pose_guess, {"used": False, "reason": "too_few_scene_points"}
        
    source.estimate_normals(max_nn=40, radius=voxel_size * 3.0)
    target.estimate_normals(max_nn=40, radius=voxel_size * 3.0)
    
    if use_colored_icp:
        if "colors" in source.point and source.point.colors.dtype == o3d.core.Dtype.UInt8:
            source.point.colors = source.point.colors.to(o3d.core.Dtype.Float32) / 255.0
        if "colors" in target.point and target.point.colors.dtype == o3d.core.Dtype.UInt8:
            target.point.colors = target.point.colors.to(o3d.core.Dtype.Float32) / 255.0
        icp_method = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    else:
        icp_method = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        
    initial_tensor = o3d.core.Tensor(initial_pose_guess, o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
    icp_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=45)
    
    # Tinh chỉnh cục bộ quanh initial_pose_guess (đã có từ RGB-D odometry):
    # KHÔNG dùng bán kính 9x (135mm) vì sẽ làm ICP trượt dọc theo mặt bàn/mặt hộp
    # dẫn tới bước nhảy 100mm như ở frame 206->207 và 258->259.
    coarse_to_fine = [voxel_size * 3.0, voxel_size * 2.0, voxel_size * 1.5]

    try:
        forward = None
        for corr_dist in coarse_to_fine:
            f_init = (forward.transformation if forward is not None else initial_tensor)
            forward = o3d.t.pipelines.registration.icp(
                source, target, max_correspondence_distance=corr_dist,
                init_source_to_target=f_init,
                estimation_method=icp_method,
                criteria=icp_criteria)
            if forward.fitness <= 0.0:
                break

        forward_mat = forward.transformation.cpu().numpy()
        backward = None
        for corr_dist in coarse_to_fine:
            b_init = (backward.transformation if backward is not None else
                      o3d.core.Tensor(np.linalg.inv(forward_mat), o3d.core.Dtype.Float64, O3D_DEVICE_CPU))
            backward = o3d.t.pipelines.registration.icp(
                target, source, max_correspondence_distance=corr_dist,
                init_source_to_target=b_init,
                estimation_method=icp_method,
                criteria=icp_criteria)
            if backward.fitness <= 0.0:
                break
            
        forward_mat = forward.transformation.cpu().numpy()
        backward_mat = backward.transformation.cpu().numpy()
        cycle_translation, cycle_rotation = inverse_consistency(forward_mat, backward_mat)
        
        info_mat = np.eye(6)
        is_degenerate = True
        try:
            info_tensor = o3d.t.pipelines.registration.get_information_matrix(
                source, target, coarse_to_fine[-1], forward.transformation)
            info_mat = info_tensor.cpu().numpy()
            is_degenerate = bool(np.linalg.eigvalsh(info_mat)[0] < 1e-4)
        except Exception:
            pass
            
        return forward_mat, {
            "used": True, "fitness": float(forward.fitness),
            "rmse_m": float(forward.inlier_rmse),
            "cycle_translation_m": cycle_translation, "cycle_rotation_deg": cycle_rotation,
            "information_matrix": info_mat.tolist(),
            "degenerate": is_degenerate
        }
    except RuntimeError as e:
        if "Singular 6x6 linear system" in str(e):
            return initial_pose_guess, {"used": False, "reason": "icp_singular_matrix_error"}
        raise e

def relocalize_submap(source, target, voxel_size, min_fitness, max_rmse, 
                      max_cycle_trans=0.015, max_cycle_rot=2.0, initial_guesses=None,
                      use_colored_icp=False, run_ransac=False, max_trans_drift=None):
    """Return a verified source→target transform, or None when unsafe.
    - Khi max_trans_drift != None (các cạnh temporal / local spatial): chỉ tinh chỉnh
      cục bộ quanh initial_guesses[0], chặn hiện tượng ICP trượt dọc mặt phẳng.
    - Khi max_trans_drift == None (các cạnh loop / cross-pass): cho phép tìm kiếm
      bán kính rộng + centroid fallback + RANSAC.
    """
    if len(source.point.positions) < 50 or len(target.point.positions) < 50:
        return None, {"reason": "too_few_points"}
    distance = voxel_size * 3.0
    
    primary_guesses = []
    if initial_guesses:
        primary_guesses.extend(initial_guesses)

    source.estimate_normals(max_nn=40, radius=voxel_size * 5.0)
    target.estimate_normals(max_nn=40, radius=voxel_size * 5.0)

    source_legacy = source.cpu().to_legacy()
    target_legacy = target.cpu().to_legacy()

    fallback_guesses = []
    # Chỉ dùng centroid fallback khi không khóa drift cục bộ (tức là cạnh loop / relocalize)
    if max_trans_drift is None:
        try:
            src_center = np.asarray(source_legacy.get_center())
            tgt_center = np.asarray(target_legacy.get_center())
            if initial_guesses:
                rot_centroid_guess = np.eye(4)
                R_g = np.asarray(initial_guesses[0], dtype=np.float64)[:3, :3]
                rot_centroid_guess[:3, :3] = R_g
                rot_centroid_guess[:3, 3] = tgt_center - (R_g @ src_center)
                fallback_guesses.append(rot_centroid_guess)
            centroid_guess = np.eye(4)
            centroid_guess[:3, 3] = tgt_center - src_center
            fallback_guesses.append(centroid_guess)
        except Exception:
            pass

    if use_colored_icp:
        if "colors" in source.point and source.point.colors.dtype == o3d.core.Dtype.UInt8:
            source.point.colors = source.point.colors.to(o3d.core.Dtype.Float32) / 255.0
        if "colors" in target.point and target.point.colors.dtype == o3d.core.Dtype.UInt8:
            target.point.colors = target.point.colors.to(o3d.core.Dtype.Float32) / 255.0
        icp_method = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    else:
        icp_method = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()
        
    icp_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
    if max_trans_drift is not None:
        # Bán kính hẹp cho cạnh cục bộ để không trượt dọc mặt bàn / vách hộp phẳng
        coarse_to_fine = [voxel_size * 3.0, voxel_size * 2.0, voxel_size * 1.5]
    else:
        coarse_to_fine = [voxel_size * 6.0, voxel_size * 3.5, voxel_size * 2.0]

    best_result = None
    best_diag = None

    def _try_guesses(guess_list, check_against_init=False):
        nonlocal best_result, best_diag
        for guess in guess_list:
            guess_tensor = o3d.core.Tensor(guess, o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
            try:
                forward = None
                for corr_dist in coarse_to_fine:
                    init = (forward.transformation if forward is not None else guess_tensor)
                    forward = o3d.t.pipelines.registration.icp(
                        source, target, max_correspondence_distance=corr_dist,
                        init_source_to_target=init,
                        estimation_method=icp_method,
                        criteria=icp_criteria)
                    if forward.fitness <= 0.0:
                        break

                forward_mat = forward.transformation.cpu().numpy()
                backward = None
                for corr_dist in coarse_to_fine:
                    b_init = (backward.transformation if backward is not None else
                              o3d.core.Tensor(np.linalg.inv(forward_mat), o3d.core.Dtype.Float64, O3D_DEVICE_CPU))
                    backward = o3d.t.pipelines.registration.icp(
                        target, source, max_correspondence_distance=corr_dist,
                        init_source_to_target=b_init,
                        estimation_method=icp_method,
                        criteria=icp_criteria)
                    if backward.fitness <= 0.0:
                        break
            except RuntimeError:
                continue

            backward_mat = backward.transformation.cpu().numpy()
            cycle_translation, cycle_rotation = inverse_consistency(forward_mat, backward_mat)

            if check_against_init and initial_guesses:
                delta_init = np.linalg.inv(initial_guesses[0]) @ forward_mat
                rot_diff = rotation_angle_deg(delta_init)
                trans_diff = float(np.linalg.norm(delta_init[:3, 3]))
                if rot_diff > 25.0:
                    continue
                if max_trans_drift is not None and trans_diff > max_trans_drift:
                    continue

            info_mat = np.eye(6)
            is_degenerate = True
            try:
                info_tensor = o3d.t.pipelines.registration.get_information_matrix(
                    source, target, coarse_to_fine[-1], forward.transformation)
                info_mat = info_tensor.cpu().numpy()
                eigvals = np.linalg.eigvalsh(info_mat)
                is_degenerate = bool(eigvals[0] < 1e-4 or (eigvals[-1] / max(eigvals[0], 1e-9)) > 1e5)
            except Exception:
                pass

            effective_rmse_limit = max_rmse
            if forward.fitness >= 0.78 and cycle_translation <= 0.0035 and cycle_rotation <= 0.6:
                effective_rmse_limit = max_rmse * 1.22

            effective_min_fitness = min_fitness
            if cycle_translation <= 0.006 and cycle_rotation <= 1.2 and forward.inlier_rmse <= max_rmse:
                effective_min_fitness = min(min_fitness, max(0.40, min_fitness * 0.75))

            verified = (forward.fitness >= effective_min_fitness and
                        forward.inlier_rmse <= effective_rmse_limit and
                        cycle_translation <= max_cycle_trans and
                        cycle_rotation <= max_cycle_rot)

            diagnostics = {
                "fitness": round(float(forward.fitness), 4),
                "rmse_mm": round(float(forward.inlier_rmse * 1000.0), 3),
                "cycle_translation_mm": round(cycle_translation * 1000.0, 3),
                "cycle_rotation_deg": round(cycle_rotation, 3),
                "information_matrix": info_mat.tolist(),
                "degenerate": is_degenerate,
                "transformation": np.asarray(forward_mat).tolist(),
                "reason": "verified" if verified else "quality_gate_failed"
            }

            if verified:
                return forward_mat, diagnostics

            if best_result is None or forward.fitness > best_result.fitness:
                best_result = forward
                best_diag = diagnostics
        return None, None

    # 1. Thử initial_guesses trước tiên
    mat, diag = _try_guesses(primary_guesses, check_against_init=(not run_ransac or max_trans_drift is not None))
    if mat is not None:
        return mat, diag

    # 2. Chỉ thử fallback_guesses khi không bị khóa max_trans_drift
    if fallback_guesses:
        mat, diag = _try_guesses(fallback_guesses, check_against_init=not run_ransac)
        if mat is not None:
            return mat, diag

    # 3. Lazy RANSAC cho các cạnh đóng vòng xa
    if run_ransac and max_trans_drift is None:
        try:
            source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                source_legacy, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
            target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                target_legacy, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))

            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_legacy, target_legacy, source_fpfh, target_fpfh, True, distance,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(25000, 0.999))
            mat, diag = _try_guesses([ransac.transformation], check_against_init=False)
            if mat is not None:
                return mat, diag
        except Exception as e:
            print(f"      [relocalize_submap] RANSAC/FPFH skipped ({e})")

    return None, best_diag

def visual_reference_candidates(query, references, color_np, descriptor_cache, limit=3, min_score=12, max_distance=52):
    orb = descriptor_cache["orb"]
    _, query_desc = orb.detectAndCompute(color_np[query], None)
    if query_desc is None:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ranked = []
    for reference in references:
        ref_desc = descriptor_cache["descriptors"].get(reference)
        if ref_desc is None:
            _, ref_desc = orb.detectAndCompute(color_np[reference], None)
            descriptor_cache["descriptors"][reference] = ref_desc
        if ref_desc is None:
            continue
        matches = matcher.match(query_desc, ref_desc)
        good = [match for match in matches if match.distance <= max_distance]
        ranked.append((len(good), reference))
    return [reference for score, reference in sorted(ranked, reverse=True)[:limit] if score >= min_score]