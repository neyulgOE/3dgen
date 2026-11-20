import numpy as np
import cv2


def generate_candidate_pairs(valid_frames, optimized_poses, color_np, 
                             max_temporal=5, spatial_dist_thr=0.18, spatial_angle_thr=35.0, 
                             num_visual_loops=3, include_non_temporal=True, submaps=None):
    """
    Sinh Candidate pairs tổng quát (không phụ thuộc tốc độ quét hay tổng số frame):
    - Phân biệt Local Spatial (cùng lượt quét) và Cross-Pass Loop (quay lại góc nhìn cũ
      sau khi đã xoay >= 90 độ quanh vật thể) dựa trên tổng góc xoay tích lũy dọc quỹ đạo.
    """
    pairs = {}
    valid_frames = sorted(list(valid_frames))
    if len(valid_frames) < 2:
        return []

    # Tính tổng góc xoay tích lũy dọc theo quỹ đạo để nhận diện chính xác khi nào
    # camera đã đi qua các mặt khác rồi vòng lại (thay vì dùng ngưỡng frame gap cố định)
    cum_rot_deg = np.zeros(len(valid_frames), dtype=np.float64)
    for k in range(1, len(valid_frames)):
        R_prev = optimized_poses[valid_frames[k - 1]][:3, :3]
        R_curr = optimized_poses[valid_frames[k]][:3, :3]
        cos_step = float(np.clip((np.trace(R_prev.T @ R_curr) - 1.0) * 0.5, -1.0, 1.0))
        step_deg = float(np.degrees(np.arccos(cos_step)))
        cum_rot_deg[k] = cum_rot_deg[k - 1] + min(step_deg, 25.0)

    descriptors = {}
    bf = None
    if include_non_temporal and num_visual_loops > 0:
        orb = cv2.ORB_create(nfeatures=1000)
        for i in valid_frames:
            _, descriptors[i] = orb.detectAndCompute(color_np[i], None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    nominal_obj_dist = 0.25  # Khoảng cách tâm nhìn danh định (m)

    for idx_i, i in enumerate(valid_frames):
        pose_i = optimized_poses[i]
        pos_i = pose_i[:3, 3]
        z_i = pose_i[:3, 2]
        look_i = pos_i + nominal_obj_dist * z_i
        
        local_spatial_scores = []
        cross_spatial_scores = []
        visual_scores = []
        
        for idx_j, j in enumerate(valid_frames):
            if i <= j:
                continue
            
            gap = idx_i - idx_j
            # 1. Temporal
            if gap <= max_temporal:
                if (i, j) not in pairs:
                    pairs[(i, j)] = 'temporal'
                continue

            if not include_non_temporal:
                continue
                
            pose_j = optimized_poses[j]
            pos_j = pose_j[:3, 3]
            z_j = pose_j[:3, 2]
            look_j = pos_j + nominal_obj_dist * z_j

            dist = float(np.linalg.norm(pos_i - pos_j))
            look_dist = float(np.linalg.norm(look_i - look_j))
            cos_angle = float(np.clip(np.dot(z_i, z_j), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            
            traj_rot_span = float(cum_rot_deg[idx_i] - cum_rot_deg[idx_j])

            # 2a. Local Spatial: camera chưa xoay quá 90 độ dọc theo quỹ đạo (cùng một lượt lia)
            if traj_rot_span < 90.0:
                if dist < spatial_dist_thr and angle_deg < spatial_angle_thr:
                    local_spatial_scores.append((dist, j))
                    continue
            # 2b. Cross-Pass Spatial: camera đã xoay >= 90 độ quanh vật thể rồi quay lại cùng hướng nhìn
            else:
                if (dist < max(spatial_dist_thr, 0.35) or look_dist < 0.25) and angle_deg < 30.0:
                    score_key = angle_deg * 0.01 + min(dist, look_dist) * 0.2
                    cross_spatial_scores.append((score_key, j))
                    continue
                
            # 3. Visual Loop candidates
            if bf is not None and descriptors.get(i) is not None and descriptors.get(j) is not None:
                if len(descriptors[i]) > 0 and len(descriptors[j]) > 0:
                    matches = bf.match(descriptors[i], descriptors[j])
                    if len(matches) > 15:
                        visual_scores.append((len(matches), j))
                        
        local_spatial_scores.sort(key=lambda x: x[0])
        for _, j in local_spatial_scores[:3]:
            if (i, j) not in pairs:
                pairs[(i, j)] = 'spatial'

        cross_spatial_scores.sort(key=lambda x: x[0])
        for _, j in cross_spatial_scores[:5]:
            if (i, j) not in pairs:
                pairs[(i, j)] = 'spatial'

        visual_scores = sorted(visual_scores, key=lambda x: x[0], reverse=True)
        for _, j in visual_scores[:num_visual_loops]:
            if (i, j) not in pairs:
                pairs[(i, j)] = 'loop'

    # 4. Cặp khép vòng HEAD <-> TAIL
    if len(valid_frames) >= 16:
        head_frames = valid_frames[:min(8, len(valid_frames) // 4)]
        tail_frames = valid_frames[-min(8, len(valid_frames) // 4):]
        for ti in tail_frames[::2]:
            for hj in head_frames[::2]:
                if ti > hj and (ti, hj) not in pairs:
                    pairs[(ti, hj)] = 'loop'

    # 5. Cầu nối giữa các submap liền kề
    if submaps and len(submaps) > 1:
        non_empty = [sorted(sm) for sm in submaps if len(sm) > 0]
        for k in range(len(non_empty) - 1):
            sm_a = non_empty[k]
            sm_b = non_empty[k + 1]
            tail_a = sm_a[-min(6, len(sm_a)):]
            head_b = sm_b[:min(6, len(sm_b))]
            for b_idx in head_b:
                for a_idx in tail_a:
                    src, tgt = max(a_idx, b_idx), min(a_idx, b_idx)
                    if src > tgt and (src, tgt) not in pairs:
                        pairs[(src, tgt)] = 'submap_bridge'

    return sorted([(i, j, t) for (i, j), t in pairs.items()])