# -*- coding: utf-8 -*-
"""
Full-Scene Feature Tracker dùng SuperPoint + LightGlue (GPU) và 3D Kabsch-RANSAC.
Thay thế hoàn toàn ORB + FPFH/RANSAC cũ.
"""
import gc
import os as _os

# Bộ bảo vệ chống lệch TSC gây crash ApproximateClock.cpp:17 trên CPU AMD Ryzen (Ubuntu 22.04)
_aff = _os.sched_getaffinity(0) if hasattr(_os, "sched_getaffinity") else None
if _aff:
    try:
        _os.sched_setaffinity(0, {min(_aff)})
    except Exception:
        pass
import torch
if _aff:
    try:
        _os.sched_setaffinity(0, _aff)
    except Exception:
        pass

import cv2
import numpy as np


def _rotation_angle_deg(T: np.ndarray) -> float:
    tr = float(np.trace(T[:3, :3]))
    cos_val = np.clip((tr - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def _kabsch_rigid(pts_src: np.ndarray, pts_tgt: np.ndarray) -> np.ndarray:
    """Tính ma trận biến đổi cứng T (4x4) sao cho pts_tgt ≈ R @ pts_src + t."""
    mu_s = pts_src.mean(axis=0)
    mu_t = pts_tgt.mean(axis=0)
    X = pts_src - mu_s
    Y = pts_tgt - mu_t
    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_t - R @ mu_s
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def solve_rigid_ransac_3d(
    pts_src: np.ndarray,
    pts_tgt: np.ndarray,
    inlier_threshold: float = 0.015,
    max_iters: int = 400,
    min_inliers: int = 12,
    seed: int = 42,
):
    """
    Giải phép biến đổi cứng 3D-3D bằng Kabsch-RANSAC (vectorized).
    Trả về: (success, T_src_to_tgt, inlier_mask, inlier_rmse)
    """
    n = len(pts_src)
    if n < max(6, min_inliers):
        return False, np.eye(4, dtype=np.float64), np.zeros(n, dtype=bool), 1e9

    rng = np.random.default_rng(seed)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0
    best_T = np.eye(4, dtype=np.float64)

    for _ in range(max_iters):
        idx = rng.choice(n, size=3, replace=False)
        p_s = pts_src[idx]
        p_t = pts_tgt[idx]

        # Loại nhanh các bộ 3 điểm bị suy biến (gần thẳng hàng hoặc lệch khoảng cách cạnh)
        d_s01 = np.linalg.norm(p_s[1] - p_s[0])
        d_t01 = np.linalg.norm(p_t[1] - p_t[0])
        if d_s01 < 0.02 or abs(d_s01 - d_t01) > inlier_threshold * 1.5:
            continue
        d_s12 = np.linalg.norm(p_s[2] - p_s[1])
        d_t12 = np.linalg.norm(p_t[2] - p_t[1])
        if d_s12 < 0.02 or abs(d_s12 - d_t12) > inlier_threshold * 1.5:
            continue

        T_cand = _kabsch_rigid(p_s, p_t)
        transformed = (pts_src @ T_cand[:3, :3].T) + T_cand[:3, 3]
        err = np.linalg.norm(transformed - pts_tgt, axis=1)
        inliers = err < inlier_threshold
        cnt = int(inliers.sum())

        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
            best_T = T_cand
            if cnt > 0.8 * n:
                break

    if best_count < min_inliers:
        return False, np.eye(4, dtype=np.float64), best_inliers, 1e9

    # Tinh chỉnh lại bằng toàn bộ inliers
    T_refined = _kabsch_rigid(pts_src[best_inliers], pts_tgt[best_inliers])
    transformed = (pts_src @ T_refined[:3, :3].T) + T_refined[:3, 3]
    err = np.linalg.norm(transformed - pts_tgt, axis=1)
    final_inliers = err < inlier_threshold
    if int(final_inliers.sum()) < min_inliers:
        return False, T_refined, final_inliers, 1e9

    T_refined = _kabsch_rigid(pts_src[final_inliers], pts_tgt[final_inliers])
    final_err = np.linalg.norm(
        ((pts_src[final_inliers] @ T_refined[:3, :3].T) + T_refined[:3, 3]) - pts_tgt[final_inliers],
        axis=1,
    )
    rmse = float(np.sqrt(np.mean(final_err ** 2)))
    return True, T_refined, final_inliers, rmse


class LightGlueFeatureTracker:
    """
    Trích xuất đặc trưng SuperPoint (cache trên CPU để tiết kiệm VRAM)
    và so khớp bằng LightGlue trên RTX 4060.
    Mặc định chạy Full-Scene (không dùng mask) để khóa tư thế 6-DOF trên nền tĩnh.
    """

    def __init__(
        self,
        max_keypoints: int = 1024,
        depth_scale: float = 1000.0,
        min_depth_m: float = 0.08,
        max_depth_m: float = 1.60,
        device: str = None,
    ):
        from lightglue import LightGlue, SuperPoint

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.depth_scale = float(depth_scale)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", depth_confidence=0.9, width_confidence=0.95).eval().to(self.device)

        # Cache đặc trưng trên CPU RAM (rất nhẹ, ~0.2MB/frame) để không chiếm 8GB VRAM
        self._feat_cache = {}

    @torch.inference_mode()
    def extract_and_cache(self, frame_id: int, rgb_image: np.ndarray):
        """Trích xuất SuperPoint 1 lần duy nhất cho frame_id (Full-Scene RGB)."""
        if frame_id in self._feat_cache:
            return self._feat_cache[frame_id]

        if rgb_image.ndim == 2:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
        elif rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]

        img_t = torch.from_numpy(rgb_image).permute(2, 0, 1).float().div_(255.0).to(self.device)
        feats = self.extractor.extract(img_t)

        # Đưa về CPU để giữ VRAM luôn thoáng
        feats_cpu = {k: v.detach().cpu() for k, v in feats.items()}
        self._feat_cache[frame_id] = feats_cpu
        return feats_cpu

    def _lift_keypoints_3d(
        self,
        uv: np.ndarray,
        depth_raw: np.ndarray,
        K: np.ndarray,
        mask: np.ndarray = None,
    ):
        """Nâng tọa độ 2D (u, v) lên 3D bằng median 3x3 trên ảnh độ sâu để chống nhiễu D435f."""
        H, W = depth_raw.shape[:2]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        n = len(uv)
        pts3d = np.zeros((n, 3), dtype=np.float64)
        valid = np.zeros(n, dtype=bool)

        for i in range(n):
            u = int(round(uv[i, 0]))
            v = int(round(uv[i, 1]))
            if u < 1 or u >= W - 1 or v < 1 or v >= H - 1:
                continue
            if mask is not None and mask[v, u] == 0:
                continue

            patch = depth_raw[v - 1 : v + 2, u - 1 : u + 2]
            nz = patch[patch > 0]
            if nz.size == 0:
                continue
            z = float(np.median(nz)) / self.depth_scale
            if z < self.min_depth_m or z > self.max_depth_m:
                continue

            x = (uv[i, 0] - cx) * z / fx
            y = (uv[i, 1] - cy) * z / fy
            pts3d[i] = (x, y, z)
            valid[i] = True

        return pts3d, valid

    @torch.inference_mode()
    def match_frames(
        self,
        src_id: int,
        tgt_id: int,
        rgb_src: np.ndarray,
        depth_src: np.ndarray,
        rgb_tgt: np.ndarray,
        depth_tgt: np.ndarray,
        K: np.ndarray,
        mask_src: np.ndarray = None,
        mask_tgt: np.ndarray = None,
        ransac_thresh_m: float = 0.018,
        min_inliers: int = 16,
    ):
        """
        So khớp 2 khung hình bằng SuperPoint + LightGlue -> Nâng 3D -> Kabsch-RANSAC.
        Trả về dict: {ok, T, num_matches, num_inliers, inlier_ratio, rmse}
        """
        f0_cpu = self.extract_and_cache(src_id, rgb_src)
        f1_cpu = self.extract_and_cache(tgt_id, rgb_tgt)

        f0 = {k: v.to(self.device) for k, v in f0_cpu.items()}
        f1 = {k: v.to(self.device) for k, v in f1_cpu.items()}

        pred = self.matcher({"image0": f0, "image1": f1})
        matches0 = pred["matches0"][0].detach().cpu().numpy()
        valid_m = matches0 > -1
        idx0 = np.where(valid_m)[0]
        idx1 = matches0[valid_m]

        num_matches = len(idx0)
        if num_matches < min_inliers:
            return {
                "ok": False,
                "T": np.eye(4, dtype=np.float64),
                "num_matches": num_matches,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "rmse": 1e9,
            }

        kpts0 = f0_cpu["keypoints"][0].numpy()[idx0]
        kpts1 = f1_cpu["keypoints"][0].numpy()[idx1]

        pts0, v0 = self._lift_keypoints_3d(kpts0, depth_src, K, mask=mask_src)
        pts1, v1 = self._lift_keypoints_3d(kpts1, depth_tgt, K, mask=mask_tgt)
        both = v0 & v1

        if int(both.sum()) < min_inliers:
            return {
                "ok": False,
                "T": np.eye(4, dtype=np.float64),
                "num_matches": num_matches,
                "num_inliers": 0,
                "inlier_ratio": 0.0,
                "rmse": 1e9,
            }

        ok, T_est, inliers, rmse = solve_rigid_ransac_3d(
            pts0[both],
            pts1[both],
            inlier_threshold=ransac_thresh_m,
            max_iters=450,
            min_inliers=min_inliers,
        )
        num_inliers = int(inliers.sum())
        inlier_ratio = float(num_inliers) / max(1, int(both.sum()))

        return {
            "ok": bool(ok and inlier_ratio >= 0.25),
            "T": T_est,
            "num_matches": num_matches,
            "num_inliers": num_inliers,
            "inlier_ratio": inlier_ratio,
            "rmse": rmse,
        }

    def release(self):
        """Giải phóng hoàn toàn VRAM RTX 4060 sau khi hoàn tất Tracking/PGO."""
        self._feat_cache.clear()
        if hasattr(self, "extractor"):
            del self.extractor
        if hasattr(self, "matcher"):
            del self.matcher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Alias giữ tương thích tên lớp cũ nếu có script khác import FeatureTracker
FeatureTracker = LightGlueFeatureTracker
