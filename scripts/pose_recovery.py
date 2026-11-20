import os
import json
import argparse
import numpy as np
import cv2
import sys

def kabsch_umeyama(P, Q, weights=None):
    """
    Finds (R, t) such that Q = R @ P + t using Kabsch algorithm.
    P, Q: (N, 3) arrays of corresponding points.
    weights: (N,) optional weights for points.
    """
    assert len(P) == len(Q)
    if len(P) == 0:
        return np.eye(3), np.zeros(3), 0.0

    if weights is None:
        weights = np.ones(len(P))
    else:
        weights = np.asarray(weights)
        if weights.sum() == 0:
            weights = np.ones(len(P))
            
    weights = weights / weights.sum()

    p_bar = np.average(P, axis=0, weights=weights)
    q_bar = np.average(Q, axis=0, weights=weights)
    P_centered = P - p_bar
    Q_centered = Q - q_bar

    # Weighted covariance matrix
    H = (P_centered * weights[:, np.newaxis]).T @ Q_centered
    
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U) * np.linalg.det(Vt)
    d = 1.0 if d > 0 else -1.0

    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    t = q_bar - R @ p_bar
    
    residuals = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
    weighted_residual = np.average(residuals, weights=weights)
    return R, t, float(weighted_residual)

def ransac_se3(P, Q, weights=None, n_iters=256, inlier_threshold_m=0.008, min_sample=4):
    N = len(P)
    if N < min_sample:
        return None, None, None

    best_inliers = np.zeros(N, bool)
    best_count   = 0
    rng          = np.random.default_rng(0)

    for _ in range(n_iters):
        idx = rng.choice(N, min_sample, replace=False)
        try:
            # unweighted kabsch for random sample
            R_tmp, t_tmp, _ = kabsch_umeyama(P[idx], Q[idx])
        except np.linalg.LinAlgError:
            continue
        residuals = np.linalg.norm((R_tmp @ P.T).T + t_tmp - Q, axis=1)
        inliers   = residuals < inlier_threshold_m
        count     = inliers.sum()
        if count > best_count:
            best_count   = count
            best_inliers = inliers

    if best_count < min_sample:
        return None, None, None

    try:
        w_inliers = weights[best_inliers] if weights is not None else None
        R, t, _ = kabsch_umeyama(P[best_inliers], Q[best_inliers], weights=w_inliers)
    except np.linalg.LinAlgError:
        return None, None, None
        
    return R, t, best_inliers

def se3_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T

def load_intrinsics(dataset_dir):
    path = os.path.join(dataset_dir, "intrinsics.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        return (float(d.get("fx", 615.0)), float(d.get("fy", 615.0)),
                float(d.get("cx", 320.0)), float(d.get("cy", 240.0)))
    return 615.0, 615.0, 320.0, 240.0

def rotation_angle_deg(R):
    cos_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))

def compute_information_matrix(P_transformed, weights=None):
    N = len(P_transformed)
    if weights is None:
        weights = np.ones(N)
    
    W_sum = weights.sum()
    if W_sum == 0: 
        return np.eye(6), np.ones(6), True
    
    weights = weights / W_sum
    
    H_tt = np.eye(3)
    mu_p = np.average(P_transformed, axis=0, weights=weights)
    mu_p_skew = np.array([
        [0, -mu_p[2], mu_p[1]],
        [mu_p[2], 0, -mu_p[0]],
        [-mu_p[1], mu_p[0], 0]
    ])
    H_tr = -mu_p_skew
    H_rt = mu_p_skew
    
    p_sq = np.sum(P_transformed**2, axis=1)
    sum_w_psq = np.sum(weights * p_sq)
    cov_p = (P_transformed * weights[:, None]).T @ P_transformed
    
    H_rr = np.eye(3) * sum_w_psq - cov_p
    
    H = np.zeros((6, 6))
    H[:3, :3] = H_tt
    H[:3, 3:] = H_tr
    H[3:, :3] = H_rt
    H[3:, 3:] = H_rr
    
    H = H * W_sum
    eigvals = np.linalg.eigvalsh(H)
    degenerate = bool(eigvals[0] < 1e-4)
    return H, eigvals, degenerate

def compute_robust_residual(residuals, c=0.05):
    if len(residuals) == 0: return 999.0
    loss = (c**2 / 2.0) * np.log(1.0 + (residuals / c)**2)
    return float(np.mean(loss))

def evaluate_registration(P, Q, p_tgt_2d, n_valid, fx, fy, cx, cy, weights=None, min_inliers=10, inlier_thr_m=0.008, ransac_iters=256):
    # 1. Forward T_anchor_to_target (P -> Q)
    R_AT, t_AT, inliers_AT = ransac_se3(P, Q, weights=weights, n_iters=ransac_iters, inlier_threshold_m=inlier_thr_m)
    if R_AT is None:
        return {"status": "REJECTED", "reason": "RANSAC failed forward"}
    T_AT = se3_matrix(R_AT, t_AT)
    inlier_count = int(inliers_AT.sum())
    inlier_ratio = inlier_count / n_valid

    # 2. Backward T_target_to_anchor (Q -> P)
    R_TA, t_TA, inliers_TA = ransac_se3(Q, P, weights=weights, n_iters=ransac_iters, inlier_threshold_m=inlier_thr_m)
    if R_TA is None:
        return {"status": "REJECTED", "reason": "RANSAC failed backward"}
    T_TA = se3_matrix(R_TA, t_TA)

    # 3. Cycle Consistency
    T_cycle_anchor = T_TA @ T_AT
    T_cycle_target = T_AT @ T_TA
    
    cycle_trans_anchor = np.linalg.norm(T_cycle_anchor[:3, 3])
    cycle_rot_anchor   = rotation_angle_deg(T_cycle_anchor[:3, :3])
    cycle_trans_target = np.linalg.norm(T_cycle_target[:3, 3])
    cycle_rot_target   = rotation_angle_deg(T_cycle_target[:3, :3])

    # 4. Reprojection Error
    Q_pred = (R_AT @ P[inliers_AT].T).T + t_AT
    x_pred = (Q_pred[:, 0] * fx / Q_pred[:, 2]) + cx
    y_pred = (Q_pred[:, 1] * fy / Q_pred[:, 2]) + cy
    uv_pred = np.stack([x_pred, y_pred], axis=1)
    
    errors = np.linalg.norm(uv_pred - p_tgt_2d[inliers_AT], axis=1)
    err_mean = float(np.mean(errors)) if len(errors) > 0 else 999.0
    err_median = float(np.median(errors)) if len(errors) > 0 else 999.0
    err_p95 = float(np.percentile(errors, 95)) if len(errors) > 0 else 999.0

    # 5. Translation, Rotation, Residual, Degeneracy
    trans_m = np.linalg.norm(t_AT)
    rot_deg = rotation_angle_deg(R_AT)
    
    Q_pred_inliers = Q_pred
    Q_inliers = Q[inliers_AT]
    W_inliers = weights[inliers_AT] if weights is not None else None
    
    res_inliers = np.linalg.norm(Q_pred_inliers - Q_inliers, axis=1)
    residual_rms = float(np.linalg.norm(res_inliers).mean()) if len(inliers_AT) > 0 else 999.0
    robust_res = compute_robust_residual(res_inliers)
    
    # Information matrix & degeneracy
    info_mat, eigvals, is_degen = compute_information_matrix(Q_pred_inliers, W_inliers)

    # 6. Categorization
    status = "REJECTED"
    if (inlier_count >= int(min_inliers * 1.2) and inlier_ratio >= 0.5 and
        err_mean < 8.0 and cycle_trans_anchor < 0.020 and cycle_rot_anchor < 2.0):
        status = "RECOVERED"
    elif (inlier_count >= min_inliers and inlier_ratio >= 0.4 and
          err_mean < 16.0 and cycle_trans_anchor < 0.050 and cycle_rot_anchor < 5.0):
        status = "RECOVERED_WEAK"

    return {
        "status": status,
        "R": R_AT.tolist(),
        "t": t_AT.tolist(),
        "T_anchor_to_target": T_AT.tolist(),
        
        "num_valid_depth": n_valid,
        "num_inliers": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "inliers": inliers_AT.tolist(),
        
        "translation_mm": round(trans_m * 1000, 2),
        "rotation_deg": round(rot_deg, 3),
        
        "reprojection_error_mean_px": round(err_mean, 2),
        "reprojection_error_median_px": round(err_median, 2),
        "reprojection_error_p95_px": round(err_p95, 2),
        
        "cycle_translation_anchor_mm": round(cycle_trans_anchor * 1000, 2),
        "cycle_rotation_anchor_deg": round(cycle_rot_anchor, 3),
        "cycle_translation_target_mm": round(cycle_trans_target * 1000, 2),
        "cycle_rotation_target_deg": round(cycle_rot_target, 3),
        
        "residual_rms_mm": round(residual_rms * 1000, 3),
        "robust_residual": round(robust_res, 6),
        
        "degenerate": is_degen,
        "hessian_eigenvalues": eigvals.tolist(),
        "information_matrix": info_mat.tolist()
    }

def recover_pose_for_frame(target_t_idx, anchor_t_idx, tracks_3d, tracks_2d, visibility, 
                           intrinsics, min_vis=0.5, ransac_iters=256, inlier_thr_m=0.008, 
                           min_inliers=10):
    fx, fy, cx, cy = intrinsics
    N = tracks_3d.shape[0]
    
    vis_ok = (visibility[:, anchor_t_idx] > min_vis) & (visibility[:, target_t_idx] > min_vis)
    anc_ok = ~np.isnan(tracks_3d[:, anchor_t_idx, 0])
    tgt_ok = ~np.isnan(tracks_3d[:, target_t_idx, 0])
    valid  = vis_ok & anc_ok & tgt_ok

    P = tracks_3d[valid, anchor_t_idx, :]
    Q = tracks_3d[valid, target_t_idx, :]
    p_tgt_2d = tracks_2d[valid, target_t_idx, :]
    
    n_valid = len(P)
    if n_valid < min_inliers:
        return {"status": "REJECTED", "reason": f"only {n_valid} valid correspondences"}

    weights = visibility[valid, anchor_t_idx] * visibility[valid, target_t_idx]
    
    res = evaluate_registration(P, Q, p_tgt_2d, n_valid, fx, fy, cx, cy, weights=weights,
                                min_inliers=min_inliers, inlier_thr_m=inlier_thr_m, ransac_iters=ransac_iters)
    res["num_tracks"] = N
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--ransac-iters", type=int, default=256)
    ap.add_argument("--inlier-thr-mm", type=float, default=8.0)
    ap.add_argument("--min-inliers", type=int, default=10)
    ap.add_argument("--min-vis", type=float, default=0.5)
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    print(f"Dataset: {dataset_dir}")

    intrinsics = load_intrinsics(dataset_dir)
    
    win_path = os.path.join(dataset_dir, "windows.json")
    if not os.path.exists(win_path):
        print(f"[ERROR] {win_path} not found.")
        return
    with open(win_path) as f:
        windows = json.load(f)

    tracks_path  = os.path.join(dataset_dir, "cotracker_tracks.npz")
    tracks3d_path = os.path.join(dataset_dir, "cotracker_tracks_3d.npz")
    data_2d = np.load(tracks_path,  allow_pickle=True)
    data_3d = np.load(tracks3d_path, allow_pickle=True)

    recovered = {}
    all_stats = []

    for idx, w in enumerate(windows):
        wk = f"win{idx}"
        anchor_frame = w["anchor_frame"]
        frames = w["frames"]
        
        try:
            tracks_3d = data_3d[f"{wk}_3d"]
            tracks_2d = data_2d[f"{wk}_2d"]
            visibility = data_2d[f"{wk}_vis"]
            saved_frames = data_3d[f"{wk}_frames"].tolist()
        except KeyError:
            print(f"[WARN] Data for {wk} not found.")
            continue
            
        anchor_t_idx = saved_frames.index(anchor_frame)
        print(f"\n── Window {idx+1}/{len(windows)}: Anchor {anchor_frame} ──")

        window_recovered = 0
        failed = len(frames) - 1
        
        for target_frame in frames:
            if target_frame == anchor_frame:
                continue
                
            target_t_idx = saved_frames.index(target_frame)
            res = recover_pose_for_frame(
                target_t_idx, anchor_t_idx, tracks_3d, tracks_2d, visibility, 
                intrinsics, min_vis=args.min_vis, ransac_iters=args.ransac_iters, 
                inlier_thr_m=args.inlier_thr_mm/1000.0, min_inliers=args.min_inliers
            )
            
            res["anchor_frame"] = anchor_frame
            res["target_frame"] = target_frame
            res["window"] = wk
            
            status = res.get("status", "REJECTED")
            
            if status in ["RECOVERED", "RECOVERED_WEAK"]:
                recovered[target_frame] = res
                window_recovered += 1
                
                print(f"   [{status}] frame {target_frame:03d}: "
                      f"inliers={res['inlier_ratio']*100:.0f}%  "
                      f"t={res['translation_mm']}mm  "
                      f"rot={res['rotation_deg']}°  "
                      f"reproj={res['reprojection_error_median_px']}px  "
                      f"cyc_t={res['cycle_translation_anchor_mm']}mm "
                      f"cyc_R={res['cycle_rotation_anchor_deg']}°")
            else:
                if "RANSAC" in res.get('reason', ''):
                    print(f"   [REJECTED] frame {target_frame:03d}: {res.get('reason')}")
                else:
                    print(f"   [REJECTED] frame {target_frame:03d}: {res.get('reason', 'failed criteria')} | "
                          f"in_n={res.get('num_inliers', 0)} "
                          f"in_ratio={res.get('inlier_ratio', 0):.2f} "
                          f"err_mean={res.get('reprojection_error_mean_px', 999):.1f}px "
                          f"cyc_t_mm={res.get('cycle_translation_anchor_mm', 999):.2f} "
                          f"cyc_R_deg={res.get('cycle_rotation_anchor_deg', 999):.2f}")
                
        all_stats.append({
            "window": wk,
            "failed": failed,
            "recovered": window_recovered
        })

    out_path = os.path.join(dataset_dir, "recovered_poses.json")
    with open(out_path, "w") as f:
        json.dump({"recovered_count": len(recovered), "frames": recovered}, f, indent=2)

    print(f"\n✅  Recovered poses: {len(recovered)} frames → {out_path}")

if __name__ == "__main__":
    main()
