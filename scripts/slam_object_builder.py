from dataset_io import *
from segmentation_utils import *
from segmentation_utils import _find_prompt_frame
from geometry_utils import *
from tracking_utils import *
from tracking_utils import _run_cotracker_recovery

import os
import sys
import argparse
import math
import numpy as np
import open3d as o3d
import cv2
import gc
import torch
import warnings
from ultralytics import YOLOWorld
from sam2.build_sam import build_sam2_video_predictor
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
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

def _resolve_compute_device():
    try:
        if o3d.core.cuda.device_count() > 0:
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return o3d.core.Device("CPU:0")


O3D_DEVICE_GPU = _resolve_compute_device()
O3D_DEVICE_CPU = o3d.core.Device("CPU:0")
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def export_debug_stage(dataset_dir, stage_name, indices, poses, color_np, depth_np,
                        masks, intrinsic_tensor, voxel=0.003):
    if not indices:
        print(f"      [Debug Stage] '{stage_name}': rỗng, bỏ qua export.")
        return None
    try:
        cloud = submap_cloud(list(indices), poses, color_np, depth_np, masks,
                             intrinsic_tensor, voxel, object_only=True)
    except Exception as e:
        print(f"      [Debug Stage] '{stage_name}': lỗi khi dựng cloud ({e}), bỏ qua.")
        return None

    n_points = len(cloud.point.positions) if hasattr(cloud, "point") else 0
    if n_points == 0:
        print(f"      [Debug Stage] '{stage_name}': cloud rỗng, bỏ qua export.")
        return None

    debug_dir = os.path.join(dataset_dir, "reconstruction", "debug_stages")
    os.makedirs(debug_dir, exist_ok=True)
    out_path = os.path.join(debug_dir, f"{stage_name}.ply")
    try:
        cloud_legacy = cloud.cpu().to_legacy() if hasattr(cloud, "cpu") else cloud
        o3d.io.write_point_cloud(out_path, cloud_legacy)
        print(f"      [Debug Stage] '{stage_name}': {len(indices)} frames, "
              f"{n_points:,} điểm -> {out_path}")
    except Exception as e:
        print(f"      [Debug Stage] '{stage_name}': lỗi khi ghi file ({e}), bỏ qua.")
        return None
    return out_path


def keep_geometrically_attached_components(mesh, voxel_size):
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) <= 1:
        return mesh, {"components": len(counts), "kept": len(counts),
                      "removed_triangles": 0, "attachment_mm": 0.0}

    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    main = int(np.argmax(counts))
    main_vertex_ids = np.unique(triangles[labels == main].reshape(-1))
    main_vertices = vertices[main_vertex_ids]
    if len(main_vertices) < 3:
        return mesh, {"components": len(counts), "kept": 1,
                      "removed_triangles": int(len(triangles) - counts[main]),
                      "attachment_mm": 0.0}

    diagonal = float(np.linalg.norm(main_vertices.max(axis=0) - main_vertices.min(axis=0)))
    attachment = max(10.0 * float(voxel_size), 0.08 * diagonal)
    min_triangles = max(40, int(0.0015 * int(counts[main])))
    tree = cKDTree(main_vertices)
    keep_components = {main}
    component_distances = {}
    component_sizes = {}

    for component, triangle_count in enumerate(counts):
        if component == main or triangle_count < min_triangles:
            continue
        component_sizes[int(component)] = int(triangle_count)
        ids = np.unique(triangles[labels == component].reshape(-1))
        if len(ids) == 0:
            continue
        distances, _ = tree.query(vertices[ids], k=1)
        q05 = float(np.percentile(distances, 5.0))
        component_distances[int(component)] = round(q05 * 1000.0, 3)
        if q05 <= attachment:
            keep_components.add(component)

    keep_mask = np.isin(labels, list(keep_components))
    removed = int(np.count_nonzero(~keep_mask))
    mesh.remove_triangles_by_mask(~keep_mask)
    mesh.remove_unreferenced_vertices()
    return mesh, {
        "components": len(counts),
        "kept": len(keep_components),
        "removed_triangles": removed,
        "attachment_mm": round(attachment * 1000.0, 3),
        "component_q05_mm": component_distances,
        "component_triangles": component_sizes,
    }


def close_nonplanar_mesh_boundaries(mesh, voxel_size):
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    if len(triangles) == 0:
        return mesh, {"closed": False, "reason": "empty_mesh"}
    edges = np.sort(np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]],
                               triangles[:, [2, 0]])), axis=1)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    if len(boundary_edges) < 3:
        return mesh, {"closed": False, "reason": "no_boundary"}

    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for start in adjacency:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        loops.append(component)

    mesh_diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    nonplanar_limit = max(4.0 * float(voxel_size), 0.03 * mesh_diagonal)
    loop_reports, has_scan_loss_boundary = [], False
    for loop in loops:
        if len(loop) < 8:
            continue
        points = vertices[loop]
        singular_values = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        rms = float(singular_values[-1] / np.sqrt(len(points)))
        extent = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
        loop_reports.append({"vertices": len(loop), "rms_mm": round(rms * 1000.0, 3),
                             "extent_mm": round(extent * 1000.0, 3)})
        if rms > nonplanar_limit and extent > 0.25 * mesh_diagonal:
            has_scan_loss_boundary = True

    if not has_scan_loss_boundary:
        return mesh, {"closed": False, "reason": "only_planar_or_small_boundaries",
                      "loops": loop_reports, "boundary_edges": len(boundary_edges)}
    try:
        import vtk
        import tempfile
        with tempfile.TemporaryDirectory(prefix="tmvisionary_hole_fill_") as temp_dir:
            source_path = os.path.join(temp_dir, "source.ply")
            filled_path = os.path.join(temp_dir, "filled.ply")
            o3d.io.write_triangle_mesh(source_path, mesh, write_ascii=False)
            reader = vtk.vtkPLYReader(); reader.SetFileName(source_path); reader.Update()
            clean = vtk.vtkCleanPolyData(); clean.SetInputConnection(reader.GetOutputPort())
            filler = vtk.vtkFillHolesFilter(); filler.SetInputConnection(clean.GetOutputPort())
            filler.SetHoleSize(max(2.0 * mesh_diagonal, 0.10)); filler.Update()
            triangulate = vtk.vtkTriangleFilter(); triangulate.SetInputConnection(filler.GetOutputPort()); triangulate.Update()
            writer = vtk.vtkPLYWriter(); writer.SetFileName(filled_path)
            writer.SetInputConnection(triangulate.GetOutputPort()); writer.SetFileTypeToBinary(); writer.Write()
            filled = o3d.io.read_triangle_mesh(filled_path)
        if len(filled.triangles) <= len(mesh.triangles):
            return mesh, {"closed": False, "reason": "triangulator_added_nothing", "loops": loop_reports}
        if mesh.has_vertex_colors() and len(filled.vertices):
            source_vertices = np.asarray(mesh.vertices)
            source_colors = np.asarray(mesh.vertex_colors)
            _, nearest = cKDTree(source_vertices).query(np.asarray(filled.vertices), k=1)
            filled.vertex_colors = o3d.utility.Vector3dVector(source_colors[nearest])
        filled.remove_degenerate_triangles()
        filled.remove_duplicated_triangles()
        filled.remove_duplicated_vertices()
        filled.remove_non_manifold_edges()
        filled.compute_vertex_normals()
        return filled, {"closed": True, "loops": loop_reports,
                        "boundary_edges_before": len(boundary_edges),
                        "triangles_added": len(filled.triangles) - len(mesh.triangles)}
    except Exception as error:
        return mesh, {"closed": False, "reason": f"vtk_unavailable_or_failed: {error}", "loops": loop_reports}


def filter_fusion_frames_by_depth_spread(indices, depth_np, masks):
    metrics = []
    for index in indices:
        mask = masks[index]
        if mask is None:
            continue
        values = depth_np[index][mask > 0]
        values = values[(values > 50) & (values < 2000)]
        if len(values) < 100:
            continue
        p05, p95 = np.percentile(values, [5, 95])
        metrics.append((index, float(p95 - p05), float(np.median(values))))
    if len(metrics) < 4:
        return indices, {"kept": len(indices), "rejected": [], "threshold_mm": None}
    spans = np.array([span for _, span, _ in metrics])
    medians = np.array([median for _, _, median in metrics])
    spread_median = float(np.median(spans))
    spread_mad = float(np.median(np.abs(spans - spread_median)))
    robust_limit = spread_median + 4.0 * 1.4826 * spread_mad
    scale_limit = 0.40 * float(np.median(medians))
    threshold = max(60.0, robust_limit, scale_limit)
    rejected = [index for index, span, _ in metrics if span > threshold]
    rejected_set = set(rejected)
    kept = [index for index in indices if index not in rejected_set]
    if len(kept) < 3:
        return indices, {"kept": len(indices), "rejected": [], "threshold_mm": round(threshold, 2),
                         "reason": "would_leave_fewer_than_three_frames"}
    return kept, {"kept": len(kept), "rejected": rejected, "threshold_mm": round(threshold, 2)}


def estimate_loop_closure(verified_indices, optimized_poses, color_np, depth_np, masks,
                           intrinsic_tensor, voxel, min_fitness, max_rmse,
                           max_cycle_translation, max_cycle_rotation, n_cluster=10):
    """Ước lượng sai số đóng vòng. Hỗ trợ cả trường hợp người dùng quay đúng 360°
    (TAIL khớp HEAD ở đầu video) lẫn trường hợp quay dư > 360° (TAIL khớp với
    một cụm ở nửa đầu video có cùng góc nhìn)."""
    if len(verified_indices) < 2 * n_cluster:
        return None, {"reason": "not_enough_frames",
                       "have": len(verified_indices), "need": 2 * n_cluster}

    tail_idx = verified_indices[-n_cluster:]
    tail_pose = optimized_poses[tail_idx[len(tail_idx) // 2]]
    tail_z = tail_pose[:3, 2]

    # Luôn thử cụm đầu tiên (HEAD = 0..n_cluster), và tìm thêm cụm có hướng nhìn
    # gần nhất với TAIL trong nửa đầu quỹ đạo (phòng trường hợp quay quá 360°).
    candidate_starts = [0]
    half_len = max(1, len(verified_indices) // 2)
    best_Start = 0
    best_cos = -1.0
    for s in range(0, half_len, max(1, n_cluster // 2)):
        mid_f = verified_indices[min(s + n_cluster // 2, len(verified_indices) - 1)]
        z_s = optimized_poses[mid_f][:3, 2]
        c = float(np.dot(tail_z, z_s))
        if c > best_cos:
            best_cos = c
            best_Start = s
    if best_Start not in candidate_starts:
        candidate_starts.insert(0, best_Start)

    lc_voxel = min(voxel, 0.006)
    tail_cloud = submap_cloud(tail_idx, optimized_poses, color_np, depth_np,
                              masks, intrinsic_tensor, lc_voxel, object_only=None)

    best_T = None
    best_diag = None
    for s in candidate_starts:
        head_idx = verified_indices[s:s + n_cluster]
        if len(head_idx) < 3:
            continue
        head_cloud = submap_cloud(head_idx, optimized_poses, color_np, depth_np,
                                  masks, intrinsic_tensor, lc_voxel, object_only=None)
        T_loop, diagnostics = relocalize_submap(
            tail_cloud, head_cloud, lc_voxel, min_fitness, max_rmse,
            max_cycle_translation, max_cycle_rotation, [np.eye(4)],
            run_ransac=True)
        if diagnostics is None:
            diagnostics = {"reason": "registration_failed_all_guesses"}
        diagnostics["head_frames"] = [head_idx[0], head_idx[-1]]
        diagnostics["tail_frames"] = [tail_idx[0], tail_idx[-1]]

        if T_loop is None and diagnostics.get("transformation") is not None:
            best_fitness = float(diagnostics.get("fitness", 0.0) or 0.0)
            best_rmse = float(diagnostics.get("rmse_mm", 999.0) or 999.0) / 1000.0
            cycle_t = float(diagnostics.get("cycle_translation_mm", 999.0) or 999.0) / 1000.0
            cycle_r = float(diagnostics.get("cycle_rotation_deg", 999.0) or 999.0)
            passed_gate = (
                (best_fitness >= 0.45 and best_rmse <= 0.0085) or
                (best_fitness >= 0.28 and best_rmse <= 0.0065 and cycle_t <= 0.012 and cycle_r <= 2.5)
            )
            if passed_gate:
                T_loop = np.array(diagnostics["transformation"], dtype=np.float64)
                diagnostics["reason"] = "verified_sharp_rmse"

        if T_loop is not None:
            translation_mm = float(np.linalg.norm(T_loop[:3, 3])) * 1000.0
            rotation_deg = float(np.linalg.norm(R.from_matrix(T_loop[:3, :3]).as_rotvec(degrees=True)))
            diagnostics.update({"accepted": True,
                                "translation_mm": round(translation_mm, 2),
                                "rotation_deg": round(rotation_deg, 3)})
            return T_loop, diagnostics

        if best_diag is None or (diagnostics.get("fitness", 0.0) or 0.0) > (best_diag.get("fitness", 0.0) or 0.0):
            best_diag = diagnostics

    if best_diag is None:
        best_diag = {"reason": "registration_failed_all_guesses", "accepted": False}
    else:
        best_diag["accepted"] = False
    return None, best_diag


def apply_loop_closure(optimized_poses, verified_indices, T_loop, anchor_start_frame=None):
    i0 = anchor_start_frame if anchor_start_frame is not None else verified_indices[0]
    i1 = verified_indices[-1]
    span = max(1, i1 - i0)
    rotvec_full = R.from_matrix(T_loop[:3, :3]).as_rotvec()
    trans_full = T_loop[:3, 3]

    corrected = [p.copy() for p in optimized_poses]
    for idx in verified_indices:
        if idx <= i0:
            continue
        alpha = float(np.clip((idx - i0) / span, 0.0, 1.0))
        delta = np.eye(4)
        delta[:3, :3] = R.from_rotvec(rotvec_full * alpha).as_matrix()
        delta[:3, 3] = trans_full * alpha
        corrected[idx] = delta @ optimized_poses[idx]
    return corrected


def apply_rigid_component_transform(optimized_poses, component_indices, transform):
    corrected = [p.copy() for p in optimized_poses]
    transform = np.asarray(transform, dtype=np.float64)
    for idx in component_indices:
        if 0 <= idx < len(corrected) and corrected[idx] is not None:
            corrected[idx] = transform @ corrected[idx]
    return corrected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--prompt", type=str,   default="cup")
    parser.add_argument("--output", type=str,   default="")
    parser.add_argument("--voxel",  type=float, default=0.0025,
                        help="TSDF voxel size (m), default 2.5mm")
    parser.add_argument("--global-optimize", action="store_true")
    parser.add_argument("--skip-slam",       action="store_true")
    parser.add_argument("--reuse-optimized-poses", action="store_true")
    parser.add_argument("--reuse-input-poses", action="store_true")
    parser.add_argument("--assume-turntable", action="store_true")
    parser.add_argument("--manual-turntable", action="store_true")
    parser.add_argument("--turntable-yaw-mode", choices=["tracked", "uniform"], default="tracked")
    parser.add_argument("--turntable-camera-motion", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--turntable-axis", type=float, nargs=3, default=(0.0, 1.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--turntable-pivot", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--turntable-max-step-deg", type=float, default=25.0)
    parser.add_argument("--turntable-yaw-step-deg", type=float, default=2.0)
    parser.add_argument("--turntable-voxel", type=float, default=0.0025)
    parser.add_argument("--turntable-max-correspondence", type=float, default=0.012)
    parser.add_argument("--turntable-min-fitness", type=float, default=0.35)
    parser.add_argument("--turntable-max-rmse", type=float, default=0.009)
    parser.add_argument("--turntable-min-feature-fitness", type=float, default=0.12)
    parser.add_argument("--turntable-max-tracking-frames", type=int, default=72)
    parser.add_argument("--sam2-model", type=str, default="sam2_hiera_small",
                        choices=["sam2_hiera_tiny", "sam2_hiera_small",
                                 "sam2_hiera_base_plus", "sam2_hiera_large"])
    parser.add_argument("--max-input-frames", type=int, default=1000)
    parser.add_argument("--max-views", "--max-tsdf-frames", dest="max_views", type=int, default=260,
                        help="Tăng từ 160 lên 260 để giữ chi tiết mảnh (quai cốc, góc hộp)")
    parser.add_argument("--max-step-translation", type=float, default=0.045)
    parser.add_argument("--max-step-rotation", type=float, default=15.0)
    parser.add_argument("--scene-tracking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-voxel", type=float, default=0.015)
    parser.add_argument("--scene-min-fitness", type=float, default=0.35)
    parser.add_argument("--scene-max-rmse", type=float, default=0.018)
    parser.add_argument("--scene-max-cycle-translation", type=float, default=0.025)
    parser.add_argument("--max-cycle-translation", type=float, default=0.012)
    parser.add_argument("--smooth-iters", type=int, default=20)
    parser.add_argument("--max-cycle-rotation", type=float, default=2.0)
    parser.add_argument("--max-consecutive-track-failures", type=int, default=3)
    parser.add_argument("--merge-submaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-voxel", type=float, default=0.006)
    parser.add_argument("--merge-min-fitness", type=float, default=0.55)
    parser.add_argument("--merge-max-rmse", type=float, default=0.008)
    parser.add_argument("--merge-max-rmse-ratio", type=float, default=0.75)
    parser.add_argument("--odometry-min-fitness", type=float, default=0.5)
    parser.add_argument("--odometry-max-rmse", type=float, default=None)
    parser.add_argument("--odometry-max-cycle-translation", type=float, default=0.025)
    parser.add_argument("--odometry-max-cycle-rotation", type=float, default=4.0)
    parser.add_argument("--pgo-refine-rounds", type=int, default=2)
    parser.add_argument("--pgo-refine-convergence-mm", type=float, default=0.5)
    parser.add_argument("--pgo-loop-closure-preference", type=float, default=0.80)
    parser.add_argument("--visual-loop-candidates", type=int, default=3)
    parser.add_argument("--pgo-non-temporal-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--relocalize-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-relocalize-frames", type=int, default=24)
    parser.add_argument("--loop-closure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loop-closure-cluster", type=int, default=10)
    parser.add_argument("--loop-closure-min-frames", type=int, default=20)
    parser.add_argument("--cotracker-recovery", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cotracker-n-points", type=int, default=96)
    parser.add_argument("--cotracker-min-inliers", type=int, default=8)
    parser.add_argument("--cotracker-inlier-thr-mm", type=float, default=10.0)
    parser.add_argument("--min-view-support", type=int, default=0, help="Hạ về 0 để không bị xóa các mảng biên")
    parser.add_argument("--keyframe-translation", type=float, default=0.005)
    parser.add_argument("--keyframe-rotation", type=float, default=2.0)
    parser.add_argument("--keyframe-max-gap", type=int, default=15)
    parser.add_argument("--min-mask-confidence", type=float, default=0.65)
    parser.add_argument("--tracking-report", type=str, default="")
    parser.add_argument("--keep-largest-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-filter", type=str, choices=["none", "soft", "strict"], default="strict")
    parser.add_argument("--fuse-depth-band-mm", type=float, default=0.0)
    parser.add_argument("--depth-discontinuity-mm", type=float, default=15.0)
    parser.add_argument("--skip-zero-degree-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flatten-planes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flatten-max-planes", type=int, default=6)
    parser.add_argument("--flatten-distance-mm", type=float, default=3.5)
    parser.add_argument("--flatten-normal-angle-deg", type=float, default=20.0)
    parser.add_argument("--flatten-min-ratio", type=float, default=0.02)
    parser.add_argument("--trim-support-plane", action=argparse.BooleanOptionalAction, default=False,
                        help="Tự động phát hiện mặt bàn đỡ dưới chân vật thể, cắt bỏ chân lem và bịt đáy phẳng.")
    parser.add_argument("--trim-support-margin-mm", type=float, default=4.0)
    parser.add_argument("--trim-support-dist-mm", type=float, default=4.0)
    parser.add_argument("--smooth-boundaries", action=argparse.BooleanOptionalAction, default=True,
                        help="Tự động làm mịn các vòng biên hở (như miệng cốc, cổ chai) để triệt tiêu răng cưa.")
    parser.add_argument("--boundary-smooth-iters", type=int, default=12)
    parser.add_argument("--boundary-smooth-alpha", type=float, default=0.5)
    parser.add_argument("--close-nonplanar-boundaries", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=0, help="Stop loading at this frame index (0 = load until end)")
    parser.add_argument("--auto-merge-split", action="store_true", default=False)
    parser.add_argument("--no-auto-merge-split", dest="auto_merge_split", action="store_false")
    parser.add_argument("--init-pose-file", type=str, default="")
    parser.add_argument("--depth-prefilter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-prefilter-smoothing", type=str, default="strong")
    parser.add_argument("--depth-prefilter-gapfill", type=str, default="normal")
    parser.add_argument("--depth-prefilter-normal", type=str, default="normal")

    args = parser.parse_args()

    # ── 1. Load dataset ────────────────────────────────────────────────────
    print("1. Loading dataset...")
    rgbd_images, color_np, depth_np, init_poses, loaded_filenames = load_dataset(
        args.dataset_dir, start_frame=args.start_frame, max_frames=args.max_input_frames,
        end_frame=args.end_frame)
    n_frames  = len(rgbd_images)
    intrinsic, intrinsic_tensor = load_intrinsics(args.dataset_dir)
    W, H = intrinsic.width, intrinsic.height
    print(f"   {n_frames} frames loaded.")
    if n_frames < 2:
        raise RuntimeError("Need at least two RGB-D frames to build a mesh.")

    _fnames_path = os.path.join(args.dataset_dir, "frame_filenames.json")
    with open(_fnames_path, "w", encoding="utf-8") as _f:
        json.dump({"filenames": loaded_filenames}, _f)
    print(f"   Frame filename map: {_fnames_path}")

    # ── 2. Handle segmentation (with caching to save VRAM/RAM) ────────────
    print("2. Segmenting target (checking cache first)...")
    recon_dir = os.path.join(args.dataset_dir, "reconstruction")
    os.makedirs(recon_dir, exist_ok=True)
    cached_mask_paths = [
        os.path.join(recon_dir, f"{os.path.splitext(fn)[0]}_mask.png")
        for fn in loaded_filenames
    ]
    all_cached = all(os.path.exists(p) for p in cached_mask_paths)

    if all_cached:
        print(f"   [Cache] Loading {n_frames} SAM2 masks from {recon_dir} (skipping SAM2)...")
        masks = []
        for p in cached_mask_paths:
            m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = (m > 127).astype(np.uint8)
                masks.append(m if np.count_nonzero(m) >= 300 else None)
            else:
                masks.append(None)
        valid = sum(m is not None for m in masks)
        print(f"   Cache: {valid}/{n_frames} frames have valid masks")
    else:
        print(f"   [Cache Miss] Loading AI Models for prompt '{args.prompt}'...")
        sam2_checkpoint = f"{args.sam2_model}.pt"
        config_map = {
            "sam2_hiera_tiny": "sam2_hiera_t.yaml",
            "sam2_hiera_small": "sam2_hiera_s.yaml",
            "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
            "sam2_hiera_large": "sam2_hiera_l.yaml"
        }
        sam2_config = config_map.get(args.sam2_model, "sam2_hiera_s.yaml")
        sam2_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint)

        try:
            yolo_model = YOLOWorld('yolov8s-world.pt')
            yolo_model.set_classes([args.prompt])
        except Exception:
            yolo_model = None

        masks = segment_all_frames_sam2(
            color_np, sam2_predictor, yolo_model,
            min_confidence=args.min_mask_confidence, min_px=300)
            
        del sam2_predictor
        if yolo_model is not None:
            del yolo_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for i, m in enumerate(masks):
            if m is not None:
                cv2.imwrite(cached_mask_paths[i], (m * 255).astype(np.uint8))

    # ── 2.5. Input Depth Pre-filtering ───────────────────────────────────────
    if args.depth_prefilter:
        print("2.5. Applying Depth Pre-filtering (Gap Filling, Surface Smoothing, Normal Check)...")
        from depth_filter import DepthPreFilter
        _K = intrinsic.intrinsic_matrix
        depth_prefilter = DepthPreFilter(
            surface_smoothing=args.depth_prefilter_smoothing,
            gap_filling=args.depth_prefilter_gapfill,
            noise_removal="strong",
            outlier_removal_level=2,
            normal_outlier_removal=args.depth_prefilter_normal,
            edge_artifact_removal=True,
            depth_min_mm=80.0,
            depth_max_mm=1200.0,
        )
        for _f_idx in range(n_frames):
            depth_np[_f_idx] = depth_prefilter.process_frame(
                depth_np[_f_idx],
                mask=masks[_f_idx],
                fx=float(_K[0, 0]), fy=float(_K[1, 1]),
                cx=float(_K[0, 2]), cy=float(_K[1, 2])
            )
        print(f"   Completed depth pre-filtering on {n_frames} frames.")

    # ── 3. Compute poses ──────────────────────────────────────────────────
    report_path = args.tracking_report or os.path.join(args.dataset_dir, "tracking_report.json")

    if args.reuse_optimized_poses or args.reuse_input_poses:
        pose_path = os.path.join(args.dataset_dir, "global_poses_optimized.json" if args.reuse_optimized_poses else "poses.json")
        if not os.path.isfile(pose_path):
            raise RuntimeError("A requested saved-pose file is missing: " + pose_path)
        with open(pose_path, "r", encoding="utf-8") as _pf:
            saved_poses = json.load(_pf)
        optimized_poses = []
        for i in range(n_frames):
            value = (saved_poses.get(str(i), saved_poses.get(i)) if args.reuse_optimized_poses
                     else saved_poses.get(f"cam_{i:03d}"))
            optimized_poses.append(np.asarray(value, dtype=np.float64).reshape(4, 4)
                                   if value is not None else init_poses[i])
        try:
            with open(report_path, "r", encoding="utf-8") as _rf:
                cached_report = json.load(_rf)
        except Exception:
            cached_report = {}
        cached_trusted = (cached_report.get("world_verified_indices_after_step7")
                          or cached_report.get("trusted_indices") or [])
        trusted_indices = sorted({int(i) for i in cached_trusted
                                  if 0 <= int(i) < n_frames and masks[int(i)] is not None})
        if len(trusted_indices) < 3:
            raise RuntimeError("Saved pose report contains fewer than 3 valid trusted frames.")
        submaps = [trusted_indices]
        verified_indices = trusted_indices
        world_verified_indices = set(trusted_indices)
        unanchored_frames = set()
        pgo_diagnostics = cached_report.get("pgo_diagnostics", {})

    elif args.manual_turntable:
        print("3. Manual turntable: estimating constrained 1-DoF object yaw...")
        valid_candidates = [i for i, mask in enumerate(masks) if mask is not None]
        if not valid_candidates:
            raise RuntimeError("No reliable target masks were found.")
        turn_start, turn_end = valid_candidates[0], valid_candidates[-1]
        estimated_pivot, frame_pivots, pivot_info = estimate_turntable_pivots(
            valid_candidates, depth_np, masks, intrinsic)
        pivot = np.asarray(args.turntable_pivot, dtype=np.float64) if args.turntable_pivot is not None else estimated_pivot
        if len(valid_candidates) > args.turntable_max_tracking_frames:
            positions = np.linspace(0, len(valid_candidates) - 1,
                                    args.turntable_max_tracking_frames).round().astype(int)
            valid_candidates = [valid_candidates[p] for p in np.unique(positions)]
        camera_poses = [np.eye(4) for _ in range(n_frames)]
        if args.turntable_yaw_mode == "uniform" and args.turntable_camera_motion:
            camera_poses, camera_trusted, _ = estimate_turntable_camera_poses(
                valid_candidates, rgbd_images, color_np, depth_np, masks, intrinsic_tensor, init_poses)
            valid_candidates = camera_trusted
        anchor = valid_candidates[0]
        anchor_cloud = frame_point_cloud(anchor, color_np, depth_np, masks, intrinsic_tensor).voxel_down_sample(
            args.turntable_voxel).cpu().to_legacy()
        if pivot is None:
            pivot = np.median(np.asarray(anchor_cloud.points), axis=0)
        axis = np.asarray(args.turntable_axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        optimized_poses = [np.eye(4) for _ in range(n_frames)]
        timeline_span = max(1, turn_end - turn_start)
        canonical_pivot = pivot
        world_axis = axis
        anchor_cloud_world = o3d.geometry.PointCloud(anchor_cloud)
        recent_clouds = [anchor_cloud_world]
        anchor_features = turntable_feature_cloud(anchor_cloud_world, canonical_pivot, world_axis)
        recent_feature_clouds = [anchor_features] if len(anchor_features.points) else []
        submaps, active = [], [anchor]
        for index in valid_candidates[1:]:
            expected_angle = 360.0 * (index - turn_start) / timeline_span
            source_pivot = frame_pivots.get(index, pivot)
            source_world = frame_point_cloud(index, color_np, depth_np, masks, intrinsic_tensor).voxel_down_sample(args.turntable_voxel).cpu().to_legacy()
            reference_cloud = o3d.geometry.PointCloud()
            for cloud in recent_clouds:
                reference_cloud += cloud
            feature_source = turntable_feature_cloud(source_world, source_pivot, world_axis)
            feature_reference = o3d.geometry.PointCloud()
            for cloud in recent_feature_clouds:
                feature_reference += cloud
            pose, _ = refine_timed_turntable_yaw(
                source_world, reference_cloud, feature_source, feature_reference,
                source_pivot, canonical_pivot, world_axis, expected_angle, args)
            if pose is None:
                continue
            optimized_poses[index] = pose
            active.append(index)
            aligned_cloud = o3d.geometry.PointCloud(source_world)
            aligned_cloud.transform(pose)
            recent_clouds.append(aligned_cloud)
            if len(recent_clouds) > 6:
                recent_clouds.pop(0)
        if active:
            submaps.append(active)
        trusted_indices = sorted(list(set(active)))
        world_verified_indices = set(trusted_indices)
        unanchored_frames = set()

    elif args.assume_turntable:
        mask_0 = masks[0] if masks[0] is not None else np.ones(color_np[0].shape[:2], dtype=bool)
        d_0 = depth_np[0].copy().astype(np.float32)
        d_0[~mask_0.astype(bool)] = 0
        pcd_0 = o3d.geometry.PointCloud.create_from_rgbd_image(
            o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(cv2.cvtColor(color_np[0], cv2.COLOR_BGR2RGB)),
                o3d.geometry.Image(d_0.astype(np.uint16)),
                depth_scale=1000.0, depth_trunc=2.5, convert_rgb_to_intensity=False),
            intrinsic
        )
        center = pcd_0.get_center()
        cx, cz = center[0], center[2]
        optimized_poses = []
        for i in range(n_frames):
            theta = (i / n_frames) * 2 * np.pi
            rot_mat = np.array([
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1]
            ])
            T_to_center = np.eye(4); T_to_center[0, 3] = -cx; T_to_center[2, 3] = -cz
            T_from_center = np.eye(4); T_from_center[0, 3] = cx; T_from_center[2, 3] = cz
            optimized_poses.append(T_from_center @ rot_mat @ T_to_center)
        trusted_indices = [i for i, mask in enumerate(masks) if mask is not None]
        world_verified_indices = set(trusted_indices)
        unanchored_frames = set()

    else:
        print("3. Computing verified handheld camera poses and submaps...")
        optimized_poses = [np.eye(4) for _ in range(n_frames)]
        valid_candidates = [i for i, mask in enumerate(masks) if mask is not None]
        if not valid_candidates:
            raise RuntimeError("No reliable target masks were found.")
        _init_world_pose = np.eye(4)
        if args.init_pose_file and os.path.isfile(args.init_pose_file):
            with open(args.init_pose_file, encoding="utf-8") as _pf:
                _init_world_pose = np.array(json.load(_pf)["last_world_pose"], dtype=np.float64)
        optimized_poses[valid_candidates[0]] = _init_world_pose
        submaps, active = [], [valid_candidates[0]]
        tracking_events = [{"frame": valid_candidates[0], "accepted": True,
                            "reason": "initial_anchor", "submap": 0}]
        anchor = valid_candidates[0]
        rejected = consecutive_failures = 0

        scene_history = []
        first_scene_cloud = frame_point_cloud(anchor, color_np, depth_np, masks, intrinsic_tensor,
                                              object_only=None).voxel_down_sample(args.scene_voxel)
        first_scene_cloud.transform(optimized_poses[anchor])
        scene_history.append(first_scene_cloud)

        for i in range(anchor + 1, n_frames):
            if masks[i] is None:
                continue
            frame_gap = max(1, i - anchor)
            prior_step = np.linalg.inv(init_poses[anchor]) @ init_poses[i]
            initial_tensor = o3d.core.Tensor(prior_step, o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
            
            try:
                forward_res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                    full_rgbd_image(i, color_np, depth_np, masks),
                    full_rgbd_image(anchor, color_np, depth_np, masks),
                    intrinsic_tensor, initial_tensor,
                    depth_scale=1000.0, depth_max=1.2)
            except RuntimeError as e:
                if "Singular 6x6 linear system" in str(e):
                    forward_res = type('obj', (object,), {'fitness': 0.0, 'transformation': initial_tensor})()
                else:
                    raise e
                
            ok_f = forward_res.fitness > 0.0
            forward = np.eye(4)
            info_f = np.zeros((6, 6))
            
            if ok_f:
                forward = forward_res.transformation.cpu().numpy()
                info_f_tensor = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
                    full_rgbd_image(i, color_np, depth_np, masks).depth,
                    full_rgbd_image(anchor, color_np, depth_np, masks).depth,
                    intrinsic_tensor, forward_res.transformation,
                    dist_threshold=0.05, depth_scale=1000.0, depth_max=1.2)
                info_f = info_f_tensor.cpu().numpy()

            ok_b, backward, info_b = False, np.eye(4), np.zeros((6, 6))
            if ok_f:
                inv_initial_tensor = o3d.core.Tensor(np.linalg.inv(forward), o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
                try:
                    backward_res = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                        full_rgbd_image(anchor, color_np, depth_np, masks),
                        full_rgbd_image(i, color_np, depth_np, masks),
                        intrinsic_tensor, inv_initial_tensor,
                        depth_scale=1000.0, depth_max=1.2)
                except RuntimeError as e:
                    if "Singular 6x6 linear system" in str(e):
                        backward_res = type('obj', (object,), {'fitness': 0.0, 'transformation': inv_initial_tensor})()
                    else:
                        raise e
                    
                ok_b = backward_res.fitness > 0.0
                if ok_b:
                    backward = backward_res.transformation.cpu().numpy()
                    info_b_tensor = o3d.t.pipelines.odometry.compute_odometry_information_matrix(
                        full_rgbd_image(anchor, color_np, depth_np, masks).depth,
                        full_rgbd_image(i, color_np, depth_np, masks).depth,
                        intrinsic_tensor, backward_res.transformation,
                        dist_threshold=0.05, depth_scale=1000.0, depth_max=1.2)
                    info_b = info_b_tensor.cpu().numpy()

            max_step_t = min(0.080, args.max_step_translation * frame_gap)
            max_step_r = min(25.0, args.max_step_rotation * frame_gap)
            plausible, translation, rotation = pose_is_plausible(forward, max_step_t, max_step_r)
            cycle_translation, cycle_rotation = inverse_consistency(forward, backward)
            sufficient_information = min(np.trace(info_f), np.trace(info_b)) >= 5e4
            consistent = (cycle_translation <= args.max_cycle_translation and
                          cycle_rotation <= args.max_cycle_rotation)
            # Kiểm tra xem VIO prior_step có mượt và hợp lệ không
            prior_plausible, prior_t, prior_r = pose_is_plausible(prior_step, max_step_t, max_step_r)

            # Chặn lỗi RGB-D odometry trượt dọc mặt phẳng (tổng quát hóa cho mọi frame):
            odom_prior_consistent = True
            if ok_f and prior_plausible:
                odom_vs_prior = np.linalg.inv(prior_step) @ forward
                odom_prior_consistent, _, _ = pose_is_plausible(
                    odom_vs_prior, max(0.018 * frame_gap, 0.5 * prior_t), 5.0 * frame_gap)

            direct_accepted = (ok_f and ok_b and plausible and odom_prior_consistent and
                               sufficient_information and consistent)

            if ok_f and plausible and odom_prior_consistent:
                initial_world_pose = optimized_poses[anchor] @ forward
            elif prior_plausible:
                initial_world_pose = optimized_poses[anchor] @ prior_step
            else:
                initial_world_pose = optimized_poses[anchor]

            scene_pose, scene_diag = refine_pose_against_scene_map(
                i, initial_world_pose, scene_history, color_np, depth_np, masks,
                intrinsic_tensor, args.scene_voxel, object_only=None) if args.scene_tracking else (initial_world_pose, {"used": False})
            scene_accepted = False
            if scene_diag.get("used", False):
                scene_relative = np.linalg.inv(optimized_poses[anchor]) @ scene_pose
                scene_plausible, _, _ = pose_is_plausible(scene_relative, max_step_t, max_step_r)
                # Chặn lỗi ICP trượt dọc theo mặt phẳng nền (như tại frame 206->207 nhảy 108mm):
                # scene_pose không được lệch quá 22mm hoặc 6 độ so với initial_world_pose!
                drift_from_init = np.linalg.inv(initial_world_pose) @ scene_pose
                init_consistent, _, _ = pose_is_plausible(drift_from_init, 0.022 * frame_gap, 6.0 * frame_gap)
                scene_cycle_limit = getattr(args, 'scene_max_cycle_translation', args.max_cycle_translation * 2.0)
                scene_accepted = (scene_plausible and init_consistent and
                                  scene_diag["fitness"] >= args.scene_min_fitness and
                                  scene_diag["rmse_m"] <= args.scene_max_rmse and
                                  scene_diag["cycle_translation_m"] <= scene_cycle_limit and
                                  scene_diag["cycle_rotation_deg"] <= args.max_cycle_rotation * 2.0)
            accepted = scene_accepted or direct_accepted

            if not accepted:
                # Fallback an toàn: nếu VIO prior_step (từ poses.json) mượt và hợp lệ,
                # giữ liên kết chuỗi bằng prior_step thay vì cắt đứt hoặc cho nhảy 100mm
                if prior_plausible and frame_gap <= 2:
                    optimized_poses[i] = optimized_poses[anchor] @ prior_step
                    active.append(i)
                    anchor = i
                    consecutive_failures = 0
                    continue

                rejected += 1
                consecutive_failures += 1
                if consecutive_failures >= args.max_consecutive_track_failures:
                    submaps.append(active)
                    optimized_poses[i] = (optimized_poses[anchor] @ prior_step) if prior_plausible else optimized_poses[anchor].copy()
                    active = [i]
                    anchor = i
                    consecutive_failures = 0
                continue

            optimized_poses[i] = scene_pose if scene_accepted else initial_world_pose
            active.append(i)
            accepted_scene_cloud = frame_point_cloud(
                i, color_np, depth_np, masks, intrinsic_tensor, object_only=None).voxel_down_sample(args.scene_voxel)
            accepted_scene_cloud.transform(optimized_poses[i])
            scene_history.append(accepted_scene_cloud)
            if len(scene_history) > 15:
                scene_history.pop(0)
            anchor = i
            consecutive_failures = 0

        submaps.append(active)
        print(f"      Submaps: {[len(submap) for submap in submaps]}; rejected={rejected}")

        for sm_id, sm in enumerate(submaps):
            export_debug_stage(args.dataset_dir, f"part{sm_id + 1}_raw", sm,
                               optimized_poses, color_np, depth_np, masks,
                               intrinsic_tensor, voxel=args.merge_voxel)

        # ── STEP 5 & 6: Submap PGO Full-Scene (SuperPoint + LightGlue + ICP) ──
        print("   -> (STEP 5-6) Chạy Submap PGO Full-Scene (SuperPoint + LightGlue + ICP)...")
        from pose_graph_optimizer import build_and_optimize_pose_graph

        valid_frames = set()
        for sm in submaps:
            valid_frames.update(sm)
        valid_indices = sorted(list(valid_frames))

        optimized_poses, pgo_diagnostics = build_and_optimize_pose_graph(
            valid_frames=valid_indices,
            rgb_list=color_np,
            depth_list=depth_np,
            init_poses=optimized_poses,
            intrinsic_o3d=intrinsic,
            submap_size=10,
            voxel_size=0.008,
            depth_scale=1000.0,
            depth_trunc=1.5,
            log_dir=recon_dir,
        )

        trusted_indices = valid_indices
        world_verified_indices = set(valid_indices)
        unanchored_frames = set()

        # ── STEP 7: Kiểm tra kết quả khép vòng 360° ─────────────────────────────
        accepted_loops = int(pgo_diagnostics.get("accepted_loop_edges", 0))
        if accepted_loops > 0:
            print(f"   -> (STEP 7) ✅ SubmapPGO đã khép vòng 360° với {accepted_loops} cạnh loop (bỏ qua CoTracker3).")
        elif args.loop_closure and len(trusted_indices) >= args.loop_closure_min_frames:
            view_graph_voxel_lc = max(args.merge_voxel, 0.015)
            effective_max_rmse = min(max(args.merge_max_rmse, args.merge_max_rmse_ratio * view_graph_voxel_lc), args.scene_max_rmse)
            _n_lc_cluster = min(args.loop_closure_cluster, len(trusted_indices) // 3)
            if _n_lc_cluster >= 3:
                print(f"   -> (STEP 7) [Fallback Global Loop Closure] Kiểm tra khép vòng trên "
                      f"{len(trusted_indices)} frames (cluster={_n_lc_cluster})...")
                _T_global_loop, _glc_diag = estimate_loop_closure(
                    trusted_indices,
                    optimized_poses,
                    color_np, depth_np, masks,
                    intrinsic_tensor,
                    voxel=view_graph_voxel_lc,
                    min_fitness=args.merge_min_fitness * 0.78,
                    max_rmse=effective_max_rmse * 1.25,
                    max_cycle_translation=args.max_cycle_translation * 2.0,
                    max_cycle_rotation=args.max_cycle_rotation * 2.5,
                    n_cluster=_n_lc_cluster,
                )
                if _T_global_loop is not None:
                    _head_start = _glc_diag.get("head_frames", [trusted_indices[0]])[0]
                    print(f"      [Global Loop Closure] ✅ Khép vòng 360° thành công "
                          f"(khớp với đoạn frame {_glc_diag.get('head_frames')}): "
                          f"bù sai số t={_glc_diag.get('translation_mm', 0):.1f}mm, "
                          f"r={_glc_diag.get('rotation_deg', 0):.2f}° dọc quỹ đạo.")
                    optimized_poses = apply_loop_closure(
                        optimized_poses, trusted_indices, _T_global_loop,
                        anchor_start_frame=_head_start)
                else:
                    print(f"      [Global Loop Closure] Bỏ qua: {_glc_diag.get('reason', '?')} "
                          f"(fitness={_glc_diag.get('fitness', 'N/A')}, "
                          f"rmse={_glc_diag.get('rmse_mm', 'N/A')}mm)")
                trusted_indices = sorted(list(valid_frames))

        try:
            with open(report_path, "r", encoding="utf-8") as _rf:
                _rep = json.load(_rf)
        except Exception:
            _rep = {}
        _rep["trusted_indices"] = trusted_indices
        _rep["world_verified_indices_after_step7"] = sorted(list(world_verified_indices))
        _rep["pgo_diagnostics"] = pgo_diagnostics
        with open(report_path, "w", encoding="utf-8") as _rf:
            json.dump(_rep, _rf, indent=2)

    export_debug_stage(args.dataset_dir, "final_strict_pool",
                       sorted(world_verified_indices) if 'world_verified_indices' in locals() else trusted_indices,
                       optimized_poses, color_np, depth_np, masks,
                       intrinsic_tensor, voxel=args.merge_voxel)

    if len(trusted_indices) < 3:
        raise RuntimeError("Tracking is not stable enough: fewer than 3 trusted views.")

    # ── 4. Estimate depth range ────────────────────────────────────────────
    print("4. Estimating object depth range...")
    sample_depths = []

    slam_trusted_set = (world_verified_indices if 'world_verified_indices' in locals() and world_verified_indices
                         else set(trusted_indices))
    _unanchored = unanchored_frames if 'unanchored_frames' in locals() else set()
    if 'pgo_diagnostics' in locals() and 'node_degree' in pgo_diagnostics:
        _nd_map = pgo_diagnostics['node_degree']
        _all_degs = [int(_nd_map.get(f, _nd_map.get(str(f), 0))) for f in slam_trusted_set]
        _med_deg = float(np.median(_all_degs)) if _all_degs else 0.0
        _min_fuse_deg = 2
        _weak_deg_frames = {
            f for f in slam_trusted_set
            if int(_nd_map.get(f, _nd_map.get(str(f), 0))) < _min_fuse_deg
        }
        if _weak_deg_frames and len(slam_trusted_set - _weak_deg_frames) >= 12:
            print(f"   [strict] Excluding {len(_weak_deg_frames)} weakly-connected frames "
                  f"(degree < {_min_fuse_deg}, median={_med_deg:.1f}): {sorted(_weak_deg_frames)}")
            _unanchored = _unanchored | _weak_deg_frames

    if args.depth_filter == "strict" and slam_trusted_set:
        strict_pool = sorted(slam_trusted_set - _unanchored)
        fuse_indices = select_motion_keyframes(
            strict_pool, optimized_poses, args.max_views,
            args.keyframe_translation, args.keyframe_rotation,
            max_frame_gap=args.keyframe_max_gap)
        print(f"   [strict] Using SLAM-verified frames: {len(fuse_indices)}/{len(strict_pool)} "
              f"(excluded {len(_unanchored)} unanchored)")
    else:
        fuse_indices = select_motion_keyframes(
            trusted_indices, optimized_poses, args.max_views,
            args.keyframe_translation, args.keyframe_rotation,
            max_frame_gap=args.keyframe_max_gap)

    fuse_indices, _depth_spread_report = filter_fusion_frames_by_depth_spread(
        fuse_indices, depth_np, masks)

    samples_per_frame = max(300, 30000 // max(1, len(fuse_indices)))
    for i in fuse_indices:
        m = masks[i]
        if m is None:
            continue
        d_valid = depth_np[i][m > 0]
        d_valid = d_valid[d_valid > 0]
        if len(d_valid) > 50:
            if len(d_valid) > samples_per_frame:
                d_sample = np.random.choice(d_valid, samples_per_frame, replace=False)
            else:
                d_sample = d_valid
            sample_depths.extend(d_sample.tolist())

    if sample_depths:
        arr = np.array(sample_depths)
        lo  = max(50,  int(np.percentile(arr, 3)  - 30))
        hi  = min(3000, int(np.percentile(arr, 97) + 80))
        print(f"   Depth range: {lo}mm – {hi}mm")
    else:
        lo, hi = 50, 800

    # ── 5. TSDF integration với masked depth ──────────────────────────────
    print("5. TSDF integration (masked depth) with Tensor API...")
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=['tsdf', 'weight', 'color'],
        attr_dtypes=[o3d.core.Dtype.Float32, o3d.core.Dtype.Float32, o3d.core.Dtype.Float32],
        attr_channels=[[1], [1], [3]],
        voxel_size=args.voxel,
        block_resolution=16,
        block_count=20000,
        device=O3D_DEVICE_GPU
    )

    integrated, skipped = 0, 0

    for fusion_number, i in enumerate(fuse_indices, start=1):
        if 'pgo_diagnostics' in locals() and 'node_degree' in pgo_diagnostics:
            _node_degree = pgo_diagnostics['node_degree']
            _deg = _node_degree.get(i, _node_degree.get(str(i), 0))
            if _deg == 0 and args.skip_zero_degree_frames and len(fuse_indices) > 8:
                skipped += 1
                continue

        mask = masks[i]
        if mask is None:
            skipped += 1
            continue

        obj_px = int(np.count_nonzero(mask))
        if obj_px < 300:
            skipped += 1
            continue

        depth_m = depth_np[i].copy().astype(np.float32)
        d_in_mask = depth_m[mask > 0]
        d_valid   = d_in_mask[(d_in_mask > 50) & (d_in_mask < 2000)]
        if len(d_valid) < 100:
            skipped += 1
            continue

        frame_median = float(np.median(d_valid))
        if args.fuse_depth_band_mm > 0:
            half_band = float(args.fuse_depth_band_mm)
        else:
            _p5, _p95 = np.percentile(d_valid, [5, 95])
            _extent = float(_p95 - _p5)
            half_band = float(np.clip(0.75 * _extent + 25.0, 40.0, 150.0))
        lower_cut = max(50.0, frame_median - half_band)
        upper_cut = min(2000.0, frame_median + half_band)

        depth_m[mask == 0] = 0
        depth_m[depth_m < lower_cut] = 0
        depth_m[depth_m > upper_cut] = 0

        if args.depth_discontinuity_mm > 0:
            _valid = (depth_m > 0).astype(np.uint8)
            _k = np.ones((3, 3), np.uint8)
            _filled = np.where(depth_m > 0, depth_m, np.float32(1e5))
            _dmin = cv2.erode(_filled, _k)
            _dmax = cv2.dilate(depth_m, _k)
            _span = _dmax - _dmin
            depth_m[(_valid > 0) & (_span > args.depth_discontinuity_mm)] = 0

        frame_lo = int(lower_cut)
        frame_hi = int(upper_cut)

        if np.count_nonzero(depth_m) < 200:
            skipped += 1
            continue

        color_t = o3d.core.Tensor(cv2.cvtColor(color_np[i], cv2.COLOR_BGR2RGB), o3d.core.Dtype.UInt8, O3D_DEVICE_GPU)
        depth_t = o3d.core.Tensor(depth_m.astype(np.uint16), o3d.core.Dtype.UInt16, O3D_DEVICE_GPU)
        t_color = o3d.t.geometry.Image(color_t)
        t_depth = o3d.t.geometry.Image(depth_t)
        depth_max_m = frame_hi / 1000.0 + 0.02

        extrinsic = o3d.core.Tensor(np.linalg.inv(optimized_poses[i]), o3d.core.Dtype.Float64, O3D_DEVICE_CPU)
        frustum_block_coords = volume.compute_unique_block_coordinates(t_depth, intrinsic_tensor, extrinsic, 1000.0, depth_max_m)
        volume.integrate(frustum_block_coords, t_depth, t_color, intrinsic_tensor, extrinsic, 1000.0, depth_max_m)

        integrated += 1

        if fusion_number % 4 == 0 or fusion_number == len(fuse_indices):
            print(f"   [{fusion_number:2d}/{len(fuse_indices)}] integrated={integrated} "
                  f"(frame median={int(frame_median)}mm range=[{int(frame_lo)},{int(frame_hi)}]mm)")

    print(f"   Done: integrated={integrated} skipped={skipped}")

    # ── 6. Extract Mesh from TSDF (Marching Cubes) ────────────────────────
    print("\n6. Extracting mesh from TSDF (Marching Cubes)...")
    mesh_t = volume.extract_triangle_mesh()
    mesh = mesh_t.to_legacy()
    mesh.compute_vertex_normals()
    n_raw = len(mesh.triangles)
    print(f"   Raw TSDF mesh: {len(mesh.vertices):,} vertices, {n_raw:,} triangles")

    if n_raw == 0:
        print("[ERROR] TSDF mesh rỗng!")
        sys.exit(1)

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # KHÔNG gọi mesh.orient_triangles() vì Marching Cubes đã có hướng chuẩn theo gradient TSDF
    mesh.compute_vertex_normals()

    required_view_support = 0 if args.manual_turntable else args.min_view_support
    if required_view_support > 0:
        before_support_filter = len(mesh.triangles)
        mesh, support = filter_mesh_by_view_support(
            mesh, fuse_indices, optimized_poses, depth_np, masks, intrinsic,
            required_view_support, args.voxel * 3.0)
        print(f"   Multi-view support filter: {before_support_filter:,} -> {len(mesh.triangles):,} triangles "
              f"(minimum {required_view_support} views)")
        if len(mesh.triangles) == 0:
            mesh = volume.extract_triangle_mesh().to_legacy()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()

    # ── 7. Component filter ─────────────────────────────────────────────────
    if args.keep_largest_component:
        print("\n7. Filtering disconnected mesh components...")
        try:
            _labels, cn, _ = mesh.cluster_connected_triangles()
            cn = np.asarray(cn)
            if len(cn) > 0:
                before_components = len(cn)
                before_triangles = len(mesh.triangles)
                mesh, component_report = keep_geometrically_attached_components(mesh, args.voxel)
                print(f"   Components: {before_components}; kept {component_report['kept']} "
                      f"geometrically attached component(s), removed "
                      f"{component_report['removed_triangles']:,}/{before_triangles:,} triangles.")
        except Exception as e:
            print(f"   [WARN] cluster filter: {e}")

    # ── 8. Taubin Smoothing ───────────────────────────────────────────────
    print("\n8. Taubin smoothing...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=getattr(args, 'smooth_iters', 20))
    mesh.compute_vertex_normals()
    print(f"   Final mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    # ── 8.5 Planar flattening ────────────────────────────────────────────
    if args.flatten_planes:
        print("\n8.5. Flattening large planar surfaces (RANSAC plane snap)...")
        mesh, _flatten_reports = flatten_planar_regions(
            mesh,
            max_planes=args.flatten_max_planes,
            ransac_distance_m=args.flatten_distance_mm / 1000.0,
            normal_angle_thresh_deg=args.flatten_normal_angle_deg,
            min_cluster_ratio=args.flatten_min_ratio,
        )
        if _flatten_reports:
            for _r in _flatten_reports:
                print(f"    Plane {_r['plane']}: {_r['vertices']:,} vertices corrected, "
                      f"rms={_r['rms_correction_mm']}mm max={_r['max_correction_mm']}mm")

    if args.trim_support_plane:
        print("\n8.6. Trimming table support plane & capping flat base...")
        from geometry_utils import trim_and_cap_support_plane
        mesh, _trim_report = trim_and_cap_support_plane(
            mesh, cut_margin_mm=args.trim_support_margin_mm, ransac_dist_mm=args.trim_support_dist_mm)
        if _trim_report.get("trimmed"):
            print(f"    Support plane: trimmed {_trim_report.get('trimmed_vertices', 0):,} table vertices, "
                  f"capped base = {_trim_report.get('capped')}")

    if args.smooth_boundaries:
        print("\n8.7. Smoothing open boundary loops (rim & edge regularization)...")
        from geometry_utils import smooth_open_boundary_loops
        mesh, _smooth_report = smooth_open_boundary_loops(
            mesh, iterations=args.boundary_smooth_iters, alpha=args.boundary_smooth_alpha)
        if _smooth_report.get("loops", 0) > 0:
            print(f"    Boundary smoothing: {_smooth_report['loops']} loop(s) regularized.")

    if args.close_nonplanar_boundaries:
        print("\n8.8. Closing non-planar scan-loss boundaries...")
        mesh, _closure_report = close_nonplanar_mesh_boundaries(mesh, args.voxel)

    # ── 9. Save ───────────────────────────────────────────────────────────
    out = (args.output if args.output
           else os.path.join(args.dataset_dir, "scanned_object_slam_mesh.ply"))
    o3d.io.write_triangle_mesh(out, mesh, write_ascii=False)
    print(f"\n✅ Saved Mesh: {out}")
    print(f"   Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")

    # ── 10. Export Poses for MVS Texturing ────────────────────────────────
    if not (args.reuse_optimized_poses or args.reuse_input_poses):
        poses_out = os.path.join(args.dataset_dir, "global_poses_optimized.json")
        exported_poses = {}
        for idx, p in enumerate(optimized_poses):
            exported_poses[idx] = p.tolist() if isinstance(p, np.ndarray) else p
        with open(poses_out, "w", encoding="utf-8") as f:
            json.dump(exported_poses, f, indent=2)
        print(f"✅ Saved Optimized Poses: {poses_out}")


if __name__ == "__main__":
    main()