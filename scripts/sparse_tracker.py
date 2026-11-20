"""
sparse_tracker.py  –  Phase 5: Offline CoTracker
=================================================
Reads a TMVisionary dataset (rgb/, depth_raw/, tracking_health.csv,
tracking_report.json) and runs CoTracker3 on the SLAM-failure windows
identified by health_state == 'LOST' or 'WARNING'.

Outputs per-dataset:
  cotracker_tracks.npz    – 2D tracks  (N_query, T, 2)
  cotracker_tracks_3d.npz – 3D tracks  (N_query, T, 3)  [in camera frame of anchor]
  cotracker_vis_winXX.mp4 – visualization video (with --vis)

Usage:
  python sparse_tracker.py <dataset_dir> [--vis] [--full] [--device cuda]
"""

import argparse
import json
import os
# Khóa tạm vào 1 core khi nạp PyTorch để tránh lỗiTSC lệch giữa các core (ApproximateClock.cpp:17)
_orig_affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
if _orig_affinity:
    try:
        os.sched_setaffinity(0, {min(_orig_affinity)})
    except Exception:
        pass
import torch
if _orig_affinity:
    try:
        os.sched_setaffinity(0, _orig_affinity)
    except Exception:
        pass
import sys

import cv2
import numpy as np



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_intrinsics(dataset_dir):
    path = os.path.join(dataset_dir, "intrinsics.json")
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        fx = float(d.get("fx", 615.0))
        fy = float(d.get("fy", 615.0))
        cx = float(d.get("cx", 320.0))
        cy = float(d.get("cy", 240.0))
    else:
        fx = fy = 615.0; cx = 320.0; cy = 240.0
    return fx, fy, cx, cy


def read_rgb_frames(dataset_dir, frame_indices, filenames=None):
    """Load BGR frames for the given SLAM frame indices.

    Args:
        dataset_dir: dataset root directory.
        frame_indices: list of SLAM frame indices to load.
        filenames: ordered list of filenames as written by slam_object_builder
            (from frame_filenames.json).  When provided, `frame_index` i maps
            to `filenames[i]`, which is the actual on-disk filename.
            When None, falls back to the old sequential-name assumption.
    """
    rgb_dir = os.path.join(dataset_dir, "rgb")
    frames = []
    for i in frame_indices:
        if filenames and i < len(filenames):
            fn = filenames[i]
        else:
            fn = f"{i:06d}.png"
        path = os.path.join(rgb_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"RGB frame not found: {path}")
        frames.append(img)
    return frames


def read_depth_frame(dataset_dir, frame_index, filenames=None):
    """Load a depth frame by SLAM frame index, respecting filename map."""
    depth_dir = os.path.join(dataset_dir, "depth_raw")
    if filenames and frame_index < len(filenames):
        fn = filenames[frame_index]
    else:
        fn = f"{frame_index:06d}.png"
    path = os.path.join(depth_dir, fn)
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth frame not found: {path}")
    return depth.astype(np.float32)


def load_trusted_indices_and_nframes(dataset_dir):
    path = os.path.join(dataset_dir, "tracking_report.json")
    if not os.path.exists(path):
        return set(), 0
    with open(path) as f:
        r = json.load(f)
    trusted = set(r.get("selected_submap", []))
    # Prefer the explicit trusted_indices field (written by Phase 8 loop).
    if "trusted_indices" in r:
        trusted = set(r["trusted_indices"])

    # Derive n_frames from frame_filenames.json (most reliable: reflects the
    # actual downsampled frame count used during the SLAM run).
    fnames_path = os.path.join(dataset_dir, "frame_filenames.json")
    if os.path.exists(fnames_path):
        with open(fnames_path) as f:
            n_frames = len(json.load(f).get("filenames", []))
    else:
        # Fallback: infer from events in the tracking report
        events = r.get("events", [])
        n_frames = max(e.get("frame", 0) for e in events) + 1 if events else 0
    return trusted, n_frames


def load_frame_filenames(dataset_dir):
    """Return the ordered list of original RGB filenames that SLAM used.

    slam_object_builder.py writes this after temporal downsampling so that
    subprocesses can load the EXACT same frames by SLAM frame index, avoiding
    the index-mismatch bug that arises when the video is subsampled.
    """
    path = os.path.join(dataset_dir, "frame_filenames.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("filenames", [])
    return []  # Fallback: caller must use sequential filenames

def find_untrusted_windows(trusted_set, total_frames, min_window=8):
    """
    Find contiguous windows of frames NOT in trusted_set.
    Expands the window to include at least one trusted frame before (and after if possible)
    so CoTracker can anchor them.
    """
    untrusted = [f for f in range(total_frames) if f not in trusted_set]
    if not untrusted:
        return []
        
    windows = []
    rs = untrusted[0]
    re = untrusted[0]
    for f in untrusted[1:]:
        if f == re + 1:
            re = f
        else:
            windows.append((rs, re))
            rs = re = f
    windows.append((rs, re))
    
    expanded = []
    for ws, we in windows:
        start_anchor = ws
        for i in range(ws - 1, -1, -1):
            if i in trusted_set:
                start_anchor = i
                break
        end_anchor = we
        for i in range(we + 1, total_frames):
            if i in trusted_set:
                end_anchor = i
                break
        # Remove artificial padding. CoTracker works fine without it, 
        # and we MUST guarantee start_anchor is actually trusted!
        if start_anchor not in trusted_set:
            print(f"   [WARN] start_anchor {start_anchor} is not trusted!")
        # To avoid pose drift from long tracking chains, cap window to 15 frames.
        # The outer while-True loop in slam_object_builder.py will call us again
        # after each hop, progressively recovering the full failure zone.
        if end_anchor - start_anchor + 1 > 15:
            end_anchor = start_anchor + 14
            print(f"   [INFO] Window {start_anchor}-{end_anchor+14} capped to 15 frames (hop-based recovery).")
        expanded.append((start_anchor, end_anchor))
        break  # Process only ONE window per invocation for iterative recovery
    return expanded


def select_query_points(anchor_bgr, anchor_depth, anchor_mask,
                        n_points=128, min_depth_mm=50):
    if anchor_mask is None:
        anchor_mask = np.ones(anchor_bgr.shape[:2], dtype=np.uint8) * 255
    else:
        anchor_mask = (anchor_mask > 0).astype(np.uint8) * 255

    gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Texture/Gradient
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = np.sqrt(grad_x**2 + grad_y**2)
    if texture.max() > 0: texture /= texture.max()

    # 2. Canny Edge
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edge_score = (edges > 0).astype(np.float32)

    # 3. Depth Gradient
    valid_depth = (anchor_depth > min_depth_mm).astype(np.float32)
    depth_smooth = cv2.GaussianBlur(anchor_depth, (5,5), 0)
    d_grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    d_grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    depth_grad = np.sqrt(d_grad_x**2 + d_grad_y**2) * valid_depth
    if depth_grad.max() > 0: depth_grad /= depth_grad.max()

    # 4. Silhouette
    silhouette = np.zeros_like(gray, dtype=np.float32)
    contours, _ = cv2.findContours(anchor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(silhouette, contours, -1, 1.0, 3)

    # Total Score S(p)
    score_map = (1.0 * edge_score) + (0.8 * depth_grad) + (0.3 * silhouette) + (0.5 * texture)
    valid_mask = (anchor_mask > 0) & (anchor_depth > min_depth_mm)
    score_map[~valid_mask] = 0.0

    # Grid sampling (10x10 cells)
    H, W = gray.shape
    grid_size_x = W // 10
    grid_size_y = H // 10
    
    all_cell_pts = []
    # Collect top K points per cell
    K = max(1, n_points // (10 * 10) + 1)
    
    for r in range(10):
        for c in range(10):
            r0, r1 = r * grid_size_y, (r+1) * grid_size_y
            c0, c1 = c * grid_size_x, (c+1) * grid_size_x
            if r == 9: r1 = H
            if c == 9: c1 = W
            
            cell_score = score_map[r0:r1, c0:c1]
            ys, xs = np.where(cell_score > 0.05)
            if len(ys) == 0: continue
            
            scores = cell_score[ys, xs]
            sort_idx = np.argsort(scores)[::-1]
            
            for i in range(min(K, len(sort_idx))):
                idx = sort_idx[i]
                all_cell_pts.append((scores[idx], c0 + xs[idx], r0 + ys[idx]))
                
    # Sort all collected cell best points by score
    all_cell_pts.sort(key=lambda x: x[0], reverse=True)
    
    # If we still need more points, fallback to global top points
    if len(all_cell_pts) < n_points:
        ys, xs = np.where(score_map > 0.05)
        scores = score_map[ys, xs]
        sort_idx = np.argsort(scores)[::-1]
        
        selected_set = {(c, r) for _, c, r in all_cell_pts}
        for idx in sort_idx:
            if len(all_cell_pts) >= n_points:
                break
            pt = (xs[idx], ys[idx])
            if pt not in selected_set:
                all_cell_pts.append((scores[idx], pt[0], pt[1]))
                selected_set.add(pt)
                
    selected = all_cell_pts[:n_points]
    if len(selected) < 4:
        return np.empty((0, 2), dtype=np.float32)
        
    pts = [[c, r] for v, c, r in selected]
    return np.array(pts, dtype=np.float32)


def lift_to_3d(tracks_2d, depth_map_dict, frame_indices, fx, fy, cx, cy):
    """tracks_2d: (N, T, 2) -> tracks_3d: (N, T, 3) in metres, NaN where invalid."""
    N, T, _ = tracks_2d.shape
    out = np.full((N, T, 3), np.nan, dtype=np.float32)
    for t, fidx in enumerate(frame_indices):
        depth = depth_map_dict.get(fidx)
        if depth is None:
            continue
        xi = np.clip(tracks_2d[:, t, 0].round().astype(int), 0, depth.shape[1] - 1)
        yi = np.clip(tracks_2d[:, t, 1].round().astype(int), 0, depth.shape[0] - 1)
        z_mm = depth[yi, xi]
        valid = z_mm > 50
        z = z_mm[valid] / 1000.0
        x = (tracks_2d[valid, t, 0] - cx) * z / fx
        y = (tracks_2d[valid, t, 1] - cy) * z / fy
        out[np.where(valid), t, :] = np.stack([x, y, z], axis=1)
    return out


def visualize_tracks(frames_bgr, tracks_2d, out_path, fps=10):
    h, w = frames_bgr[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    N, T, _ = tracks_2d.shape
    rng = np.random.default_rng(42)
    colors = [tuple(int(c) for c in rng.integers(60, 255, 3)) for _ in range(N)]
    for t, frame in enumerate(frames_bgr):
        img = frame.copy()
        for n in range(N):
            x, y = int(tracks_2d[n, t, 0]), int(tracks_2d[n, t, 1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), 3, colors[n], -1)
                if t > 0:
                    px, py = int(tracks_2d[n, t-1, 0]), int(tracks_2d[n, t-1, 1])
                    if 0 <= px < w and 0 <= py < h:
                        cv2.line(img, (px, py), (x, y), colors[n], 1)
        # Label frame
        state = ""
        writer.write(img)
    writer.release()
    print(f"   Visualization saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Offline CoTracker for SLAM failure recovery")
    ap.add_argument("dataset_dir")
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-points", type=int, default=128)
    ap.add_argument("--vis",  action="store_true", help="Write visualization MP4")
    ap.add_argument("--full", action="store_true", help="Track all frames, not just failures")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir
    device      = torch.device(args.device)
    print(f"Dataset : {dataset_dir}")
    print(f"Device  : {device}")

    # Load model
    print("Loading CoTracker3 (offline) …")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                               trust_repo=True)
    cotracker = cotracker.to(device).eval()
    print("   Ready.")

    fx, fy, cx, cy = load_intrinsics(dataset_dir)
    trusted_set, n_frames = load_trusted_indices_and_nframes(dataset_dir)
    filenames = load_frame_filenames(dataset_dir)

    if n_frames == 0:
        if filenames:
            n_frames = len(filenames)
        else:
            rgb_dir  = os.path.join(dataset_dir, "rgb")
            n_frames = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])

    # Validate: trusted indices must not exceed n_frames
    trusted_set = {f for f in trusted_set if f < n_frames}
        
    print(f"   Frames: {n_frames}   Trusted SLAM frames: {len(trusted_set)}")

    # Determine windows
    if args.full or not trusted_set:
        windows = [(0, n_frames - 1)]
    else:
        windows = find_untrusted_windows(trusted_set, n_frames)
        if not windows:
            print("   SLAM healthy throughout — nothing to recover.")
            print("   Use --full to run CoTracker anyway.")
            return

    print(f"\n   Failure windows ({len(windows)}):")
    for ws, we in windows:
        n_untrusted = sum(1 for f in range(ws, we+1) if f not in trusted_set)
        print(f"      {ws:03d}–{we:03d}  ({we-ws+1} frames)  Untrusted={n_untrusted}")

    # Load mask cache
    recon_dir  = os.path.join(dataset_dir, "reconstruction")
    mask_cache = {}
    def get_mask(idx):
        if idx in mask_cache:
            return mask_cache[idx]
        p = os.path.join(recon_dir, f"{idx:06d}_mask.png")
        if os.path.exists(p):
            mask_cache[idx] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            return mask_cache[idx]
        return None

    all_results = []
    exported_windows = []

    for win_idx, (ws, we) in enumerate(windows):
        print(f"\n── Window {win_idx+1}/{len(windows)}: frames {ws}–{we} ──")
        fidxs = list(range(ws, we + 1))
        T     = len(fidxs)
        
        if T > 75:
            print(f"   [WARN] Window too large for offline CoTracker ({T} > 75). Skipping to avoid OOM.")
            continue

        # Use the TRUSTED frame at the start of the window as anchor.
        # This is always ws (find_untrusted_windows guarantees ws is trusted).
        # Anchoring on a trusted frame means CoTracker inherits a known-good
        # pose, minimising drift across the failure window.
        anchor_frame = ws
        exported_windows.append({"anchor_frame": anchor_frame, "frames": fidxs})

        frames_bgr = read_rgb_frames(dataset_dir, fidxs, filenames=filenames)
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        video_np   = np.stack(frames_rgb).astype(np.float32)          # (T,H,W,3)
        video_t    = torch.from_numpy(video_np).permute(0,3,1,2)      # (T,3,H,W)
        video_t    = video_t.unsqueeze(0).to(device)                   # (1,T,3,H,W)

        anchor_depth = read_depth_frame(dataset_dir, anchor_frame, filenames=filenames)
        anchor_mask  = get_mask(anchor_frame)
        qpts = select_query_points(frames_bgr[0], anchor_depth, anchor_mask,
                                   n_points=args.n_points)
        N = len(qpts)
        if N == 0:
            print("   No query points — skipping.")
            continue
        print(f"   Query points: {N} on frame {anchor_frame}")

        queries_t = torch.zeros((1, N, 3), device=device)
        queries_t[0, :, 1] = torch.from_numpy(qpts[:, 0])
        queries_t[0, :, 2] = torch.from_numpy(qpts[:, 1])

        with torch.no_grad():
            pred_tracks, pred_vis = cotracker(video_t, queries=queries_t)

        tracks  = pred_tracks[0].cpu().numpy().transpose(1, 0, 2)  # (N, T, 2)
        vis_arr = pred_vis[0].cpu().numpy().transpose(1, 0)         # (N, T)

        mean_vis = vis_arr.mean()
        print(f"   Mean visibility: {mean_vis:.3f}  ({vis_arr.sum():.0f} / {N*T})")

        depths = {f: read_depth_frame(dataset_dir, f, filenames=filenames) for f in fidxs}
        tracks_3d = lift_to_3d(tracks, depths, fidxs, fx, fy, cx, cy)
        valid_3d  = (~np.isnan(tracks_3d[:, :, 0])).sum()
        print(f"   Valid 3D lifts : {valid_3d} / {N*T}")
        
        # Calculate track confidence (V1: visibility + depth validity + smoothness)
        depth_validity = (~np.isnan(tracks_3d[:, :, 0])).astype(np.float32)
        velocity = np.zeros((N, T), dtype=np.float32)
        velocity[:, 1:] = np.linalg.norm(tracks[:, 1:, :] - tracks[:, :-1, :], axis=2)
        max_vel = 20.0 # jump of > 20 pixels per frame is likely a tracking failure
        smoothness = 1.0 - np.clip(velocity / max_vel, 0.0, 1.0)
        track_confidence = vis_arr * depth_validity * smoothness
        mean_conf = track_confidence.mean()
        print(f"   Mean confidence: {mean_conf:.3f}")

        all_results.append({
            "window":        (ws, we),
            "frame_indices": fidxs,
            "anchor_frame":  anchor_frame,
            "queries_2d":    qpts,
            "tracks_2d":     tracks,
            "visibility":    vis_arr,
            "track_confidence": track_confidence,
            "tracks_3d":     tracks_3d,
        })

        if args.vis:
            vis_path = os.path.join(dataset_dir, f"cotracker_vis_win{win_idx:02d}.mp4")
            visualize_tracks(frames_bgr, tracks, vis_path)

    if not all_results:
        print("\nNo results to save.")
        return

    out_2d = os.path.join(dataset_dir, "cotracker_tracks.npz")
    out_3d = os.path.join(dataset_dir, "cotracker_tracks_3d.npz")
    np.savez(out_2d,
             **{f"win{i}_2d":     r["tracks_2d"]     for i, r in enumerate(all_results)},
             **{f"win{i}_vis":    r["visibility"]     for i, r in enumerate(all_results)},
             **{f"win{i}_conf":   r["track_confidence"] for i, r in enumerate(all_results)},
             **{f"win{i}_frames": np.array(r["frame_indices"])
                                                     for i, r in enumerate(all_results)})
    np.savez(out_3d,
             **{f"win{i}_3d":      r["tracks_3d"]    for i, r in enumerate(all_results)},
             **{f"win{i}_queries": r["queries_2d"]   for i, r in enumerate(all_results)},
             **{f"win{i}_frames":  np.array(r["frame_indices"])
                                                     for i, r in enumerate(all_results)})
    print(f"\n✅  2D tracks: {out_2d}")
    print(f"✅  3D tracks: {out_3d}")

    print("\n── Summary ──")
    for i, r in enumerate(all_results):
        ws, we = r["window"]
        N, T, _ = r["tracks_2d"].shape
        valid_3d = (~np.isnan(r["tracks_3d"][:,:,0])).sum()
        print(f"  Win {i+1}: {ws:03d}–{we:03d}  N={N} T={T}  "
              f"vis={r['visibility'].mean():.3f} conf={r['track_confidence'].mean():.3f} valid3d={valid_3d}/{N*T}")

    # Export explicit window definitions so pose_recovery.py doesn't have to guess
    win_path = os.path.join(dataset_dir, "windows.json")
    with open(win_path, "w") as f:
        json.dump(exported_windows, f, indent=2)
    print(f"✅  Windows metadata: {win_path}")


if __name__ == "__main__":
    main()
