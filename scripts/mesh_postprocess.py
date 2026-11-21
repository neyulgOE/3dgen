# -*- coding: utf-8 -*-
"""
Hậu xử lý lưới 3D (Mesh-to-CAD Post-Processing) cho TMVisionary 3D Studio.
Chạy độc lập trên file .ply / .obj / .stl đã sinh ra (thuần Open3D + NumPy + SciPy):
  1. Làm sạch cấu trúc manifold & giữ thành phần liên thông chính.
  2. Làm mịn giữ cạnh bằng Bilateral Normal Filtering (Sun et al. / Zheng et al.).
  3. Phân loại tự động vòng biên hở: Lỗ thủng (hole) | Đáy (base) | Miệng hở (opening).
  4. Vá lỗ cục bộ bằng Delaunay 2D + Chia lưới + Laplacian Fairing tiếp tuyến cong.
  5. Bịt đáy phẳng bằng Constrained-style Delaunay 2D lưới đều.
  6. (Tùy chọn) Tạo độ dày thành vỏ (Solidify) cho vật rỗng miệng hở (cốc/bát).
  7. (Tùy chọn) Giảm đa giác giữ hình dáng (Quadric Decimation) xuất CAD (.obj/.stl/.ply).
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── 1. Tiện ích hình học & Kiểm tra đa giác 2D ──────────────────────────────

def _parse_up_axis(axis_str: str) -> np.ndarray:
    mapping = {
        "+x": [1.0, 0.0, 0.0], "-x": [-1.0, 0.0, 0.0],
        "+y": [0.0, 1.0, 0.0], "-y": [0.0, -1.0, 0.0],
        "+z": [0.0, 0.0, 1.0], "-z": [0.0, 0.0, -1.0],
    }
    key = axis_str.strip().lower()
    if key not in mapping:
        raise ValueError(f"Trục up không hợp lệ: {axis_str}. Chọn một trong {list(mapping.keys())}")
    return np.array(mapping[key], dtype=np.float64)


def _points_in_polygon_2d(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-casting vector hóa: kiểm tra mảng điểm (M, 2) có nằm trong đa giác kín (K, 2) không."""
    x = pts[:, 0]
    y = pts[:, 1]
    px = poly[:, 0]
    py = poly[:, 1]
    n = len(poly)
    inside = np.zeros(len(pts), dtype=bool)
    j = n - 1
    for i in range(n):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def _clean_mesh_topology(mesh: o3d.geometry.TriangleMesh, merge_eps: float = 1e-5) -> o3d.geometry.TriangleMesh:
    if merge_eps > 0:
        mesh.merge_close_vertices(merge_eps)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def _extract_directed_boundary_loops(triangles: np.ndarray):
    """Trích xuất các vòng biên hở có hướng từ lưới tam giác.
    Cạnh biên (u -> v) là nửa cạnh chỉ xuất hiện 1 lần trên toàn bộ lưới.
    """
    edge_count = defaultdict(int)
    directed_edges = []
    for a, b, c in triangles:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (u, v) if u < v else (v, u)
            edge_count[key] += 1
            directed_edges.append((u, v, key))

    next_map = defaultdict(list)
    for u, v, key in directed_edges:
        if edge_count[key] == 1:
            next_map[u].append(v)

    visited_edges = set()
    loops = []
    for start_u, nbrs in list(next_map.items()):
        for first_v in nbrs:
            if (start_u, first_v) in visited_edges:
                continue
            loop = [start_u]
            u, v = start_u, first_v
            visited_edges.add((u, v))
            while True:
                if v == start_u:
                    break
                loop.append(v)
                cands = [nxt for nxt in next_map.get(v, []) if (v, nxt) not in visited_edges]
                if not cands:
                    break
                nxt = cands[0]
                visited_edges.add((v, nxt))
                u, v = v, nxt
            if len(loop) >= 3 and v == start_u:
                loops.append(loop)
    return loops


def _boundary_vertex_set(triangles: np.ndarray) -> set:
    edges = np.sort(
        np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    b_edges = unique_edges[counts == 1]
    return set(np.unique(b_edges).tolist())


# ── 2. Làm mịn giữ cạnh: Bilateral Normal Filtering ─────────────────────────

def bilateral_normal_filter(
    mesh: o3d.geometry.TriangleMesh,
    normal_iters: int = 5,
    vertex_iters: int = 10,
    sigma_angle_deg: float = 28.0,
    sigma_spatial_mult: float = 2.0,
    preserve_boundary: bool = True,
) -> o3d.geometry.TriangleMesh:
    """Làm mịn bề mặt bảo toàn góc cạnh (Bilateral Normal Filtering + Sun's Vertex Update).
    - Triệt tiêu gợn sóng nhiễu D435f trên cả mặt cong (thân cốc) lẫn mặt phẳng (hộp).
    - Tự động ngắt trọng số qua các cạnh gập lớn hơn ~sigma_angle_deg để giữ sắc cạnh.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts = len(verts)
    n_tris = len(tris)
    if n_tris < 20:
        return mesh

    # Dựng lân cận mặt theo chung đỉnh (1-ring face neighborhood)
    v_to_f = [[] for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(tris):
        v_to_f[a].append(fi)
        v_to_f[b].append(fi)
        v_to_f[c].append(fi)

    # Tính cạnh trung bình để định tỷ lệ sigma không gian
    e01 = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
    e12 = np.linalg.norm(verts[tris[:, 2]] - verts[tris[:, 1]], axis=1)
    e20 = np.linalg.norm(verts[tris[:, 0]] - verts[tris[:, 2]], axis=1)
    mean_edge = float(np.median(np.concatenate([e01, e12, e20])))
    sigma_s = max(mean_edge * sigma_spatial_mult, 1e-4)
    # Ngưỡng chênh lệch vector pháp tuyến: ||n_i - n_j|| = 2 * sin(theta / 2)
    sigma_r = 2.0 * np.sin(np.radians(sigma_angle_deg) * 0.5)
    inv_2s2 = 1.0 / (2.0 * sigma_s * sigma_s)
    inv_2r2 = 1.0 / (2.0 * sigma_r * sigma_r)

    # Dựng bảng cặp (fi, fj) cho lân cận mặt để vector hóa trên NumPy
    src_list = []
    dst_list = []
    for fi, (a, b, c) in enumerate(tris):
        nbrs = set(v_to_f[a]) | set(v_to_f[b]) | set(v_to_f[c])
        nbrs.discard(fi)
        if nbrs:
            src_list.extend([fi] * len(nbrs))
            dst_list.extend(nbrs)

    src_idx = np.asarray(src_list, dtype=np.int64)
    dst_idx = np.asarray(dst_list, dtype=np.int64)

    # Tính tâm mặt, diện tích và pháp tuyến ban đầu
    v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    centroids = (v0 + v1 + v2) / 3.0
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(2.0 * areas[:, None], 1e-12)

    # Khoảng cách không gian giữa các tâm mặt lân cận (tính 1 lần)
    diff_c = centroids[src_idx] - centroids[dst_idx]
    w_spatial = np.exp(-np.sum(diff_c * diff_c, axis=1) * inv_2s2) * np.maximum(areas[dst_idx], 1e-10)

    # Bước A: Lọc song phương trên vector pháp tuyến mặt
    for _ in range(normal_iters):
        diff_n = normals[src_idx] - normals[dst_idx]
        w_range = np.exp(-np.sum(diff_n * diff_n, axis=1) * inv_2r2)
        w = w_spatial * w_range

        weighted_n = normals[dst_idx] * w[:, None]
        accum = np.zeros_like(normals)
        np.add.at(accum, src_idx, weighted_n)
        # Cộng thêm chính nó để tránh suy biến nếu mặt cô lập
        accum += normals * np.maximum(areas[:, None], 1e-10)
        norms = np.linalg.norm(accum, axis=1, keepdims=True)
        normals = accum / np.maximum(norms, 1e-12)

    # Bước B: Cập nhật vị trí đỉnh khớp theo pháp tuyến đã lọc (Sun et al., 2007)
    b_verts = _boundary_vertex_set(tris) if preserve_boundary else set()
    is_boundary = np.zeros(n_verts, dtype=bool)
    if b_verts:
        is_boundary[list(b_verts)] = True

    # Chuẩn bị cặp (vertex_id, face_id)
    vf_v = tris.reshape(-1)
    vf_f = np.repeat(np.arange(n_tris, dtype=np.int64), 3)
    v_deg = np.bincount(vf_v, minlength=n_verts).astype(np.float64)
    v_deg = np.maximum(v_deg, 1.0)

    for _ in range(vertex_iters):
        c_cur = (verts[tris[:, 0]] + verts[tris[:, 1]] + verts[tris[:, 2]]) / 3.0
        n_f = normals[vf_f]
        c_f = c_cur[vf_f]
        v_p = verts[vf_v]
        # Hình chiếu (c_f - v_p) lên pháp tuyến n_f
        proj = np.sum((c_f - v_p) * n_f, axis=1)
        disp_pair = n_f * proj[:, None]

        disp_sum = np.zeros_like(verts)
        np.add.at(disp_sum, vf_v, disp_pair)
        step = disp_sum / v_deg[:, None]
        # Giảm biên độ dịch chuyển ở đỉnh biên hở để giữ nguyên kích thước miệng cốc
        if preserve_boundary:
            step[is_boundary] *= 0.25
        verts += step

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


# ── 2.5. Vector Planar Edge Snapping (Cạnh giao 2 mặt phẳng & Góc 3 mặt) ────

def _solve_plane_intersection_line(n1: np.ndarray, d1: float, n2: np.ndarray, d2: float, ref_pt: np.ndarray):
    """Giải đường giao tuyến 3D của 2 mặt phẳng n1·p + d1 = 0 và n2·p + d2 = 0.
    Trả về (p0, l_unit) với p0 là điểm trên giao tuyến gần ref_pt nhất.
    """
    l_vec = np.cross(n1, n2)
    l_norm = float(np.linalg.norm(l_vec))
    if l_norm < 1e-6:
        return None, None
    l_unit = l_vec / l_norm

    # Hệ 3x3 chốt vị trí dọc theo l_unit tại đúng mặt phẳng qua ref_pt
    A = np.vstack([n1, n2, l_unit])
    b = np.array([-d1, -d2, float(np.dot(l_unit, ref_pt))], dtype=np.float64)
    try:
        p0 = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return p0, l_unit


def _is_vertex_move_safe(vi: int, new_pos: np.ndarray, verts: np.ndarray, tris: np.ndarray, v_to_tris: list) -> bool:
    """Kiểm tra dịch chuyển đỉnh vi -> new_pos không làm đảo chiều pháp tuyến hay suy biến tam giác kề."""
    for fi in v_to_tris[vi]:
        a, b, c = int(tris[fi, 0]), int(tris[fi, 1]), int(tris[fi, 2])
        p0, p1, p2 = verts[a], verts[b], verts[c]
        old_n = np.cross(p1 - p0, p2 - p0)
        old_len = float(np.linalg.norm(old_n))
        if old_len < 1e-12:
            continue

        q0 = new_pos if a == vi else p0
        q1 = new_pos if b == vi else p1
        q2 = new_pos if c == vi else p2
        new_n = np.cross(q1 - q0, q2 - q0)
        new_len = float(np.linalg.norm(new_n))
        if new_len < 1e-10:
            return False
        if float(np.dot(old_n, new_n)) / (old_len * new_len) < 0.20:
            return False
    return True


def snap_planar_intersections(
    mesh: o3d.geometry.TriangleMesh,
    max_planes: int = 6,
    snap_radius_mm: float = 3.0,
    min_dihedral_deg: float = 25.0,
    max_dihedral_deg: float = 155.0,
    adjacency_dist_mm: float = 12.0,
):
    """Tách các mặt phẳng thật (tự bỏ qua mặt cong/trụ), giải giao tuyến cạnh
    l = n1 x n2 và đỉnh góc 3 mặt, sau đó kéo sắc các đỉnh dải chuyển tiếp
    trong phạm vi <= snap_radius_mm (bảo vệ bo góc fillet thật).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from geometry_utils import flatten_planar_regions

    mesh, planes = flatten_planar_regions(
        mesh,
        max_planes=max_planes,
        blend_alpha=0.90,
        include_plane_data=True,
    )
    if len(planes) < 2:
        return mesh, {
            "planes_detected": len(planes),
            "edges_snapped": [],
            "corners_snapped": 0,
            "total_verts_snapped": 0,
        }

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    n_verts = len(verts)

    adj = [set() for _ in range(n_verts)]
    v_to_tris = [[] for _ in range(n_verts)]
    for fi, (a, b, c) in enumerate(tris):
        ai, bi, ci = int(a), int(b), int(c)
        adj[ai].update((bi, ci))
        adj[bi].update((ai, ci))
        adj[ci].update((ai, bi))
        v_to_tris[ai].append(fi)
        v_to_tris[bi].append(fi)
        v_to_tris[ci].append(fi)

    # Phân biệt đỉnh lõi mặt phẳng (khóa cố định) vs đỉnh biên / chuyển tiếp (được phép snap)
    plane_core_mask = np.zeros(n_verts, dtype=bool)
    plane_trees = []
    plane_sets = []
    for p in planes:
        v_ids = np.asarray(p["vertex_indices"], dtype=np.int64)
        v_set = set(v_ids.tolist())
        plane_sets.append(v_set)
        plane_trees.append(cKDTree(verts[v_ids]))
        for vi in v_ids:
            if all(nb in v_set for nb in adj[int(vi)]):
                plane_core_mask[int(vi)] = True

    snap_radius_m = float(snap_radius_mm) / 1000.0
    adj_dist_m = float(adjacency_dist_mm) / 1000.0
    seam_band_m = max(snap_radius_m * 2.0, 0.006)

    adjacent_pairs = set()
    edge_reports = []
    total_snapped = 0

    # 1. Giải giao tuyến cho từng cặp mặt phẳng kề nhau
    for i in range(len(planes)):
        ni = np.asarray(planes[i]["normal"], dtype=np.float64)
        di = float(planes[i]["d"])
        pts_i = verts[planes[i]["vertex_indices"]]

        for j in range(i + 1, len(planes)):
            nj = np.asarray(planes[j]["normal"], dtype=np.float64)
            dj = float(planes[j]["d"])

            cos_ang = float(np.clip(np.dot(ni, nj), -1.0, 1.0))
            dihedral_deg = float(np.degrees(np.arccos(cos_ang)))
            if not (min_dihedral_deg <= dihedral_deg <= max_dihedral_deg):
                continue

            # Kiểm tra 2 cụm mặt phẳng có kề nhau trong không gian không
            dists_ij, _ = plane_trees[j].query(pts_i, k=1)
            near_i = pts_i[dists_ij <= adj_dist_m]
            if len(near_i) < 10:
                continue

            adjacent_pairs.add((i, j))
            ref_pt = near_i.mean(axis=0)
            p0, l_unit = _solve_plane_intersection_line(ni, di, nj, dj, ref_pt)
            if p0 is None:
                continue

            # Chọn đỉnh ứng viên nằm trong dải chuyển tiếp giữa 2 mặt phẳng
            dist_pi = np.abs(verts @ ni + di)
            dist_pj = np.abs(verts @ nj + dj)
            cand_ids = np.nonzero((~plane_core_mask) & (dist_pi <= seam_band_m) & (dist_pj <= seam_band_m))[0]
            if len(cand_ids) == 0:
                continue

            # Chỉ giữ đỉnh thực sự nằm gần ranh giới 2 cụm i và j
            di_c, _ = plane_trees[i].query(verts[cand_ids], k=1)
            dj_c, _ = plane_trees[j].query(verts[cand_ids], k=1)
            cand_ids = cand_ids[(di_c <= adj_dist_m) & (dj_c <= adj_dist_m)]

            snapped_this_edge = 0
            for vi in cand_ids:
                v = verts[int(vi)]
                v_proj = p0 + float(np.dot(v - p0, l_unit)) * l_unit
                move_dist = float(np.linalg.norm(v_proj - v))
                if move_dist > snap_radius_m or move_dist < 1e-6:
                    continue
                if _is_vertex_move_safe(int(vi), v_proj, verts, tris, v_to_tris):
                    verts[int(vi)] = v_proj
                    snapped_this_edge += 1

            total_snapped += snapped_this_edge
            edge_reports.append({
                "pair": f"P{i}-P{j}",
                "dihedral_deg": round(dihedral_deg, 2),
                "snapped_verts": snapped_this_edge,
            })

    # 2. Giải đỉnh góc 3 mặt phẳng đôi một kề nhau
    corners_snapped = 0
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            if (i, j) not in adjacent_pairs:
                continue
            for k in range(j + 1, len(planes)):
                if (i, k) not in adjacent_pairs or (j, k) not in adjacent_pairs:
                    continue
                ni = np.asarray(planes[i]["normal"], dtype=np.float64)
                nj = np.asarray(planes[j]["normal"], dtype=np.float64)
                nk = np.asarray(planes[k]["normal"], dtype=np.float64)
                A = np.vstack([ni, nj, nk])
                if abs(float(np.linalg.det(A))) < 0.15:
                    continue
                b = np.array([-float(planes[i]["d"]), -float(planes[j]["d"]), -float(planes[k]["d"])], dtype=np.float64)
                try:
                    corner = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                dists_corner = np.linalg.norm(verts - corner[None, :], axis=1)
                near_c = np.nonzero((~plane_core_mask) & (dists_corner <= snap_radius_m))[0]
                if len(near_c) == 0:
                    continue
                # Kéo đỉnh gần góc nhất lên đúng tọa độ góc CAD
                best_vi = int(near_c[int(np.argmin(dists_corner[near_c]))])
                if _is_vertex_move_safe(best_vi, corner, verts, tris, v_to_tris):
                    verts[best_vi] = corner
                    corners_snapped += 1
                    total_snapped += 1

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh = _clean_mesh_topology(mesh)
    return mesh, {
        "planes_detected": len(planes),
        "edges_snapped": edge_reports,
        "corners_snapped": corners_snapped,
        "total_verts_snapped": total_snapped,
    }


# ── 3. Phân loại vòng biên hở (Lỗ thủng / Đáy / Miệng hở) ──────────────────

def classify_boundary_loops(
    mesh: o3d.geometry.TriangleMesh,
    up_vec: np.ndarray,
    max_hole_extent_ratio: float = 0.18,
    min_major_loop_ratio: float = 0.30,
    preserve_through_holes: bool = True,
):
    """Phân loại tất cả các vòng biên hở thành:
      - 'base': vòng biên lớn nằm ở đáy tiếp xúc mặt bàn (thấp nhất theo up_vec).
      - 'opening': vòng biên lớn nằm ở đỉnh/miệng (như miệng cốc, không bịt đặc).
      - 'aperture': khe rỗng xuyên thấu (như lỗ quai cốc), tuyệt đối không vá bít.
      - 'hole': lỗ thủng cục bộ trên bề mặt cần vá mượt.
      - 'large_open': biên hở lớn không rõ đáy/miệng.
    """
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    v_normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    loops = _extract_directed_boundary_loops(tris)
    if not loops:
        return []

    diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    heights = verts @ up_vec
    h_min, h_max = float(heights.min()), float(heights.max())
    h_span = max(h_max - h_min, 1e-6)

    # Bước 1: Trích đặc trưng hình học cho từng vòng biên
    classified = []
    for lid, loop in enumerate(loops):
        pts = verts[loop]
        n_loop = v_normals[loop]
        centroid = pts.mean(axis=0)
        extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        extent_ratio = extent / max(diag, 1e-6)
        h_loop = pts @ up_vec
        h_mean_norm = float((h_loop.mean() - h_min) / h_span)
        h_spread_norm = float((h_loop.max() - h_loop.min()) / h_span)

        # Pháp tuyến mặt phẳng PCA của vòng biên
        centered = pts - centroid
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        plane_normal = vt[2]

        # Độ đồng hướng pháp tuyến đỉnh & độ thẳng hàng với pháp tuyến mặt phẳng lỗ
        mean_v_normal = n_loop.mean(axis=0)
        normal_coherence = float(np.linalg.norm(mean_v_normal))
        mean_v_normal_unit = mean_v_normal / max(normal_coherence, 1e-8)
        normal_plane_align = float(np.mean(np.abs(n_loop @ plane_normal)))

        if extent_ratio >= min_major_loop_ratio and h_mean_norm <= 0.18 and h_spread_norm <= 0.25:
            kind = "base"
        elif extent_ratio >= min_major_loop_ratio and h_mean_norm >= 0.78 and h_spread_norm <= 0.30:
            kind = "opening"
        elif extent_ratio <= max_hole_extent_ratio:
            # Nếu pháp tuyến đỉnh tỏa xung quanh hoặc vuông góc mặt phẳng lỗ -> mép khe/ống xuyên thấu
            if preserve_through_holes and (normal_coherence < 0.45 or normal_plane_align < 0.45):
                kind = "aperture"
            else:
                kind = "hole"
        else:
            kind = "large_open"

        classified.append({
            "id": lid,
            "loop": loop,
            "kind": kind,
            "vertices": len(loop),
            "extent_mm": round(extent * 1000.0, 2),
            "extent_ratio": round(extent_ratio, 3),
            "height_norm": round(h_mean_norm, 3),
            "normal_coherence": round(normal_coherence, 3),
            "normal_plane_align": round(normal_plane_align, 3),
            "_centroid": centroid,
            "_extent": extent,
            "_plane_normal": plane_normal,
            "_mean_v_normal": mean_v_normal_unit,
        })

    # Bước 2: Bắt cặp 2 vòng biên song song đối diện nhau (mặt trước & sau của khe quai cốc)
    if preserve_through_holes:
        cand_ids = [i for i, item in enumerate(classified) if item["kind"] == "hole"]
        paired = set()
        for a_idx in range(len(cand_ids)):
            i = cand_ids[a_idx]
            ci = classified[i]
            for b_idx in range(a_idx + 1, len(cand_ids)):
                j = cand_ids[b_idx]
                cj = classified[j]

                # Khoảng cách 2 tâm nhỏ (độ dày quai/thành thường 3..35mm)
                dist_m = float(np.linalg.norm(ci["_centroid"] - cj["_centroid"]))
                max_ext = max(ci["_extent"], cj["_extent"], 1e-6)
                min_ext = min(ci["_extent"], cj["_extent"])
                if dist_m > min(0.035, max_ext * 1.35):
                    continue
                # Kích thước 2 lỗ tương đương nhau
                if (min_ext / max_ext) < 0.45:
                    continue
                # 2 mặt phẳng lỗ gần song song & nằm đối diện dọc theo trục pháp tuyến mặt phẳng
                plane_parallel = abs(float(np.dot(ci["_plane_normal"], cj["_plane_normal"])))
                if plane_parallel < 0.75:
                    continue
                delta = cj["_centroid"] - ci["_centroid"]
                delta_unit = delta / max(dist_m, 1e-8)
                axial_align = abs(float(np.dot(delta_unit, ci["_plane_normal"])))
                # Pháp tuyến vỏ của 2 mặt hướng ngược chiều nhau
                opp_normals = float(np.dot(ci["_mean_v_normal"], cj["_mean_v_normal"]))
                if axial_align >= 0.55 and opp_normals < -0.25:
                    paired.add(i)
                    paired.add(j)

        for idx in paired:
            classified[idx]["kind"] = "aperture"

    # Dọn các khóa nội bộ trước khi trả về
    for item in classified:
        for k in ("_centroid", "_extent", "_plane_normal", "_mean_v_normal"):
            item.pop(k, None)

    return classified


# ── 4. Vá lỗ cục bộ + Làm mượt tiếp tuyến cong (Fairing) ────────────────────

def _triangulate_loop_2d(pts_3d: np.ndarray, target_edge: float, add_interior_grid: bool = True):
    """Chiếu đa giác biên 3D lên mặt phẳng tốt nhất (PCA), rải điểm nội bộ và chia lưới Delaunay 2D.
    Trả về (new_interior_pts_3d, tri_indices_local, plane_origin, plane_normal).
    Chỉ số 0..K-1 tương ứng đúng thứ tự đỉnh trên vòng biên; K.. tương ứng điểm nội bộ mới tạo.
    """
    k = len(pts_3d)
    origin = pts_3d.mean(axis=0)
    centered = pts_3d - origin
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u_vec, v_vec, n_vec = vt[0], vt[1], vt[2]

    poly_2d = np.stack([centered @ u_vec, centered @ v_vec], axis=1)

    # Đảm bảo thứ tự góc chuẩn trên 2D
    all_2d = [poly_2d]
    interior_3d = []

    if add_interior_grid and target_edge > 1e-5:
        x_min, y_min = poly_2d.min(axis=0)
        x_max, y_max = poly_2d.max(axis=0)
        gx = np.arange(x_min + target_edge * 0.7, x_max - target_edge * 0.5, target_edge)
        gy = np.arange(y_min + target_edge * 0.7, y_max - target_edge * 0.5, target_edge)
        if len(gx) > 0 and len(gy) > 0:
            xx, yy = np.meshgrid(gx, gy)
            grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
            # Chỉ giữ điểm nằm trong đa giác và cách biên ít nhất 0.45 * target_edge
            inside = _points_in_polygon_2d(grid_pts, poly_2d)
            cand = grid_pts[inside]
            if len(cand) > 0:
                b_tree = cKDTree(poly_2d)
                dists, _ = b_tree.query(cand, k=1)
                cand = cand[dists >= 0.45 * target_edge]
            if len(cand) > 0:
                all_2d.append(cand)
                p3 = origin + np.outer(cand[:, 0], u_vec) + np.outer(cand[:, 1], v_vec)
                interior_3d = p3

    pts_2d_all = np.vstack(all_2d)
    if len(pts_2d_all) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), origin, n_vec

    try:
        tri = Delaunay(pts_2d_all)
    except Exception:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), origin, n_vec

    simplices = tri.simplices
    # Lọc chỉ giữ các tam giác có trọng tâm nằm hoàn toàn trong đa giác biên
    tri_centers = pts_2d_all[simplices].mean(axis=1)
    keep = _points_in_polygon_2d(tri_centers, poly_2d)
    valid_tris = simplices[keep]

    if len(interior_3d) == 0:
        interior_3d = np.zeros((0, 3), dtype=np.float64)
    return interior_3d, valid_tris, origin, n_vec


def _align_patch_winding(patch_tris: np.ndarray, loop_global_ids: list) -> np.ndarray:
    """Đảm bảo chiều quấn (winding) của miếng vá khớp với lưới gốc.
    Biên hở gốc đi theo hướng (u -> v), do đó tam giác vá chung cạnh đó phải đi theo hướng ngược lại (v -> u).
    """
    if len(patch_tris) == 0:
        return patch_tris
    k = len(loop_global_ids)
    # Các cạnh hướng mong muốn cho miếng vá là (i+1 -> i) trong chỉ số local
    desired = set(((i + 1) % k, i) for i in range(k))
    same_dir = 0
    opp_dir = 0
    for a, b, c in patch_tris:
        for u, v in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if (u, v) in desired:
                same_dir += 1
            elif (v, u) in desired:
                opp_dir += 1
    if opp_dir > same_dir:
        patch_tris = patch_tris[:, [0, 2, 1]]
    return patch_tris


def fill_local_holes_with_fairing(
    mesh: o3d.geometry.TriangleMesh,
    classified_loops: list,
    fairing_iters: int = 25,
):
    """Vá các lỗ thủng cục bộ ('hole') rồi chạy Laplacian Fairing khóa cứng đỉnh gốc
    để miếng vá uốn cong mượt theo bề mặt xung quanh.
    """
    hole_loops = [info for info in classified_loops if info["kind"] == "hole"]
    if not hole_loops:
        return mesh, {"holes_filled": 0, "triangles_added": 0}

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    tris = np.asarray(mesh.triangles, dtype=np.int64).copy()
    has_colors = mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64).copy() if has_colors else None

    orig_v_count = len(verts)
    e01 = np.linalg.norm(verts[tris[:, 1]] - verts[tris[:, 0]], axis=1)
    target_edge = float(np.median(e01)) * 1.15

    new_verts_list = [verts]
    new_colors_list = [colors] if has_colors else None
    new_tris_list = [tris]
    current_v_offset = orig_v_count
    holes_filled = 0
    tris_added = 0

    for info in hole_loops:
        loop = info["loop"]
        pts_loop = verts[loop]
        interior_3d, local_tris, _, _ = _triangulate_loop_2d(
            pts_loop, target_edge=target_edge, add_interior_grid=True
        )
        if len(local_tris) == 0:
            continue

        local_tris = _align_patch_winding(local_tris, loop)
        k = len(loop)
        local_to_global = np.empty(k + len(interior_3d), dtype=np.int64)
        local_to_global[:k] = loop
        if len(interior_3d) > 0:
            local_to_global[k:] = np.arange(current_v_offset, current_v_offset + len(interior_3d))
            new_verts_list.append(interior_3d)
            if has_colors:
                # Nội suy màu theo đỉnh biên gần nhất
                _, nn = cKDTree(pts_loop).query(interior_3d, k=1)
                new_colors_list.append(colors[np.asarray(loop)[nn]])
            current_v_offset += len(interior_3d)

        global_patch_tris = local_to_global[local_tris]
        new_tris_list.append(global_patch_tris)
        holes_filled += 1
        tris_added += len(global_patch_tris)

    if holes_filled == 0:
        return mesh, {"holes_filled": 0, "triangles_added": 0}

    all_verts = np.vstack(new_verts_list)
    all_tris = np.vstack(new_tris_list)

    # Laplacian Fairing chỉ áp dụng lên các đỉnh nội bộ mới tạo (idx >= orig_v_count)
    n_new_verts = len(all_verts) - orig_v_count
    if n_new_verts > 0 and fairing_iters > 0:
        adj = defaultdict(set)
        for a, b, c in all_tris:
            if a >= orig_v_count or b >= orig_v_count or c >= orig_v_count:
                adj[int(a)].update((int(b), int(c)))
                adj[int(b)].update((int(a), int(c)))
                adj[int(c)].update((int(a), int(b)))
        for _ in range(fairing_iters):
            updated = all_verts[orig_v_count:].copy()
            for vi in range(orig_v_count, len(all_verts)):
                nbrs = adj.get(vi)
                if nbrs:
                    updated[vi - orig_v_count] = all_verts[list(nbrs)].mean(axis=0)
            all_verts[orig_v_count:] = updated

    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(all_tris)
    if has_colors:
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.vstack(new_colors_list))
    mesh = _clean_mesh_topology(mesh)
    return mesh, {"holes_filled": holes_filled, "triangles_added": tris_added}


# ── 5. Bịt đáy phẳng bằng Delaunay 2D lưới đều ──────────────────────────────

def cap_base_delaunay(
    mesh: o3d.geometry.TriangleMesh,
    classified_loops: list,
    up_vec: np.ndarray,
):
    """Ép phẳng các đỉnh biên đáy lên mặt phẳng đỡ và bịt đáy bằng lưới tam giác Delaunay 2D đều."""
    base_loops = [info for info in classified_loops if info["kind"] == "base"]
    if not base_loops:
        return mesh, {"capped": False, "reason": "no_base_loop_detected"}

    # Chọn vòng biên đáy lớn nhất
    base_info = max(base_loops, key=lambda x: x["vertices"])
    loop = base_info["loop"]

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    tris = np.asarray(mesh.triangles, dtype=np.int64).copy()
    has_colors = mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64).copy() if has_colors else None

    pts_loop = verts[loop]
    # Fit mặt phẳng đáy và ép phẳng nhẹ các đỉnh biên đáy lên mặt phẳng đó
    origin = pts_loop.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_loop - origin, full_matrices=False)
    n_base = vt[2]
    # Đảm bảo pháp tuyến mặt đáy hướng xuống dưới (ngược chiều up_vec)
    if np.dot(n_base, up_vec) > 0:
        n_base = -n_base

    # Chiếu các đỉnh biên đáy lên mặt phẳng đáy
    signed_d = (pts_loop - origin) @ n_base
    verts[loop] = pts_loop - np.outer(signed_d, n_base)
    pts_loop_flat = verts[loop]

    edge_lens = np.linalg.norm(np.roll(pts_loop_flat, -1, axis=0) - pts_loop_flat, axis=1)
    target_edge = max(float(np.median(edge_lens)) * 1.3, 0.0025)

    interior_3d, local_tris, _, _ = _triangulate_loop_2d(
        pts_loop_flat, target_edge=target_edge, add_interior_grid=True
    )
    if len(local_tris) == 0:
        return mesh, {"capped": False, "reason": "delaunay_failed"}

    # Kiểm tra hướng pháp tuyến tam giác đáy: phải hướng ra ngoài (cùng chiều n_base)
    pts_comb = np.vstack([pts_loop_flat, interior_3d]) if len(interior_3d) > 0 else pts_loop_flat
    v0 = pts_comb[local_tris[:, 0]]
    v1 = pts_comb[local_tris[:, 1]]
    v2 = pts_comb[local_tris[:, 2]]
    tri_n = np.cross(v1 - v0, v2 - v0)
    if float(np.mean(tri_n @ n_base)) < 0:
        local_tris = local_tris[:, [0, 2, 1]]

    orig_v_count = len(verts)
    k = len(loop)
    local_to_global = np.empty(k + len(interior_3d), dtype=np.int64)
    local_to_global[:k] = loop

    if len(interior_3d) > 0:
        local_to_global[k:] = np.arange(orig_v_count, orig_v_count + len(interior_3d))
        all_verts = np.vstack([verts, interior_3d])
        if has_colors:
            base_col = np.median(colors[loop], axis=0)
            all_colors = np.vstack([colors, np.tile(base_col, (len(interior_3d), 1))])
    else:
        all_verts = verts
        all_colors = colors

    global_base_tris = local_to_global[local_tris]
    all_tris = np.vstack([tris, global_base_tris])

    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(all_tris)
    if has_colors:
        mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)
    mesh = _clean_mesh_topology(mesh)
    return mesh, {
        "capped": True,
        "triangles_added": int(len(global_base_tris)),
        "base_up_normal": (-n_base).tolist(),
    }


# ── 6. Tạo độ dày thành (Solidify / Shell-to-Solid cho cốc/bát hở miệng) ────

def solidify_open_shell(mesh: o3d.geometry.TriangleMesh, thickness_mm: float = 3.0):
    """Biến lưới vỏ mỏng có miệng hở (như cốc, bát) thành khối kín (watertight) có độ dày thành."""
    if thickness_mm <= 0:
        return mesh, {"solidified": False}

    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    has_colors = mesh.has_vertex_colors()
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_colors else None

    n_v = len(verts)
    thickness_m = thickness_mm / 1000.0

    # Vỏ trong dịch ngược pháp tuyến và đảo chiều tam giác
    inner_verts = verts - normals * thickness_m
    inner_tris = tris[:, [0, 2, 1]] + n_v

    # Khâu kín tất cả các vòng biên hở (vành miệng cốc)
    loops = _extract_directed_boundary_loops(tris)
    rim_tris = []
    for loop in loops:
        k = len(loop)
        for i in range(k):
            u = loop[i]
            v = loop[(i + 1) % k]
            u_in = u + n_v
            v_in = v + n_v
            rim_tris.append([u, v_in, v])
            rim_tris.append([u, u_in, v_in])

    all_verts = np.vstack([verts, inner_verts])
    if rim_tris:
        all_tris = np.vstack([tris, inner_tris, np.asarray(rim_tris, dtype=np.int64)])
    else:
        all_tris = np.vstack([tris, inner_tris])

    solid = o3d.geometry.TriangleMesh()
    solid.vertices = o3d.utility.Vector3dVector(all_verts)
    solid.triangles = o3d.utility.Vector3iVector(all_tris)
    if has_colors:
        solid.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, colors]))
    solid = _clean_mesh_topology(solid)
    return solid, {
        "solidified": True,
        "thickness_mm": thickness_mm,
        "rim_loops_stitched": len(loops),
        "is_watertight": bool(solid.is_watertight()),
    }


# ── 7. Chuẩn hóa hệ tọa độ CAD ──────────────────────────────────────────────

def align_mesh_to_cad_frame(
    mesh: o3d.geometry.TriangleMesh,
    up_axis_vec: np.ndarray,
    scale_to_mm: bool = False,
):
    """Chuẩn hóa lưới từ hệ tọa độ camera OpenCV sang hệ CAD chuẩn:
      - Trục đứng xoay trùng +Z ([0, 0, 1]) theo pháp tuyến đáy (hoặc up_vec).
      - Mặt đáy đặt phẳng tại Z = 0.
      - Tâm ngang (X, Y) đưa về gốc (0, 0).
      - (Tùy chọn) Nhân 1000.0 để đổi đơn vị từ mét sang milimét.
    """
    src = np.asarray(up_axis_vec, dtype=np.float64)
    src_norm = float(np.linalg.norm(src))
    if src_norm < 1e-8:
        src = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        src = src / src_norm

    dst = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))

    if dot > 1.0 - 1e-9:
        R = np.eye(3, dtype=np.float64)
    elif dot < -1.0 + 1e-9:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        axis = np.cross(src, dst)
        axis = axis / float(np.linalg.norm(axis))
        angle = float(np.arccos(dot))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        R = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    verts_rot = verts @ R.T

    # Đưa tâm bounding-box ngang (X, Y) về (0, 0) và đáy (Z_min) về 0
    x_center = 0.5 * (float(verts_rot[:, 0].min()) + float(verts_rot[:, 0].max()))
    y_center = 0.5 * (float(verts_rot[:, 1].min()) + float(verts_rot[:, 1].max()))
    z_min = float(verts_rot[:, 2].min())
    verts_rot[:, 0] -= x_center
    verts_rot[:, 1] -= y_center
    verts_rot[:, 2] -= z_min

    scale = 1000.0 if scale_to_mm else 1.0
    if scale != 1.0:
        verts_rot *= scale

    mesh.vertices = o3d.utility.Vector3dVector(verts_rot)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    ext = verts_rot.max(axis=0) - verts_rot.min(axis=0)
    unit = "mm" if scale_to_mm else "m"
    return mesh, {
        "aligned": True,
        "tilt_corrected_deg": round(float(np.degrees(np.arccos(dot))), 2),
        "unit": unit,
        "bbox": [round(float(v), 2) for v in ext],
    }


# ── 8. CLI Entry Point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hậu xử lý Mesh-to-CAD (Làm mịn giữ cạnh, vá lỗ cong, bịt đáy, Solidify)")
    parser.add_argument("input_mesh", help="Đường dẫn file lưới đầu vào (.ply / .obj / .stl)")
    parser.add_argument("-o", "--output", default="", help="Đường dẫn file đầu ra (mặc định thêm _cad.ply)")
    parser.add_argument("--up-axis", default="-y", choices=["+x", "-x", "+y", "-y", "+z", "-z"],
                        help="Trục hướng lên trên của vật thể (OpenCV/RealSense mặc định là -y)")
    parser.add_argument("--bilateral-iters", type=int, default=5, help="Số vòng lọc pháp tuyến song phương")
    parser.add_argument("--vertex-iters", type=int, default=10, help="Số vòng cập nhật đỉnh khớp pháp tuyến")
    parser.add_argument("--edge-angle-deg", type=float, default=28.0, help="Ngưỡng góc giữ cạnh sắc (độ)")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True,
                        help="Tự động vá lỗ thủng cục bộ kèm làm mượt độ cong")
    parser.add_argument("--max-hole-ratio", type=float, default=0.18,
                        help="Tỷ lệ độ trải tối đa so với đường chéo vật để coi là lỗ thủng cục bộ")
    parser.add_argument("--preserve-through-holes", action=argparse.BooleanOptionalAction, default=True,
                        help="Bảo vệ khe rỗng xuyên thấu (như lỗ quai cốc) khỏi bị vá nhầm thành lỗ thủng")
    parser.add_argument("--cap-base", action=argparse.BooleanOptionalAction, default=True,
                        help="Tự động bịt đáy phẳng bằng Delaunay 2D")
    parser.add_argument("--solidify-mm", type=float, default=0.0,
                        help="Tạo độ dày thành vỏ kín khối (mm) cho cốc/bát hở miệng (0 = tắt)")
    parser.add_argument("--target-triangles", type=int, default=0,
                        help="Giảm số tam giác về mức mục tiêu bằng Quadric Decimation (0 = giữ nguyên)")
    parser.add_argument("--align-cad", action=argparse.BooleanOptionalAction, default=False,
                        help="Xoay trục đứng về +Z, đặt đáy tại Z=0, đưa tâm (X, Y) về (0, 0)")
    parser.add_argument("--export-mm", action=argparse.BooleanOptionalAction, default=False,
                        help="Nhân tọa độ x1000 sang đơn vị mm khi xuất file CAD (.stl/.obj)")
    parser.add_argument("--snap-planar-edges", action=argparse.BooleanOptionalAction, default=False,
                        help="Làm phẳng mặt phẳng thật và kéo sắc cạnh giao (n1 x n2), tự bỏ qua vật cong")
    parser.add_argument("--snap-radius-mm", type=float, default=3.0,
                        help="Bán kính kéo sắc cạnh tối đa (mm) để bảo vệ bo góc fillet thật (mặc định 3.0)")
    args = parser.parse_args()

    t0 = time.time()
    if not os.path.isfile(args.input_mesh):
        raise FileNotFoundError(f"Không tìm thấy file lưới: {args.input_mesh}")

    mesh = o3d.io.read_triangle_mesh(args.input_mesh)
    if len(mesh.triangles) == 0:
        raise RuntimeError("File lưới rỗng (0 tam giác).")

    print(f"[PostProcess] Đã nạp: {args.input_mesh} "
          f"({len(mesh.vertices):,} đỉnh, {len(mesh.triangles):,} tam giác)")
    mesh = _clean_mesh_topology(mesh)
    up_vec = _parse_up_axis(args.up_axis)

    # Bước 1: Làm mịn giữ cạnh (Bilateral Normal Filtering)
    if args.bilateral_iters > 0:
        print(f"1. Bilateral Normal Filtering (normal_iters={args.bilateral_iters}, "
              f"vertex_iters={args.vertex_iters}, edge_angle={args.edge_angle_deg}°)...")
        mesh = bilateral_normal_filter(
            mesh,
            normal_iters=args.bilateral_iters,
            vertex_iters=args.vertex_iters,
            sigma_angle_deg=args.edge_angle_deg,
            preserve_boundary=True,
        )

    # Bước 1.5: Vector Planar Edge Snapping (chỉ chạy trên mặt phẳng thật, tự bỏ qua Cốc)
    if args.snap_planar_edges:
        mesh, snap_rep = snap_planar_intersections(
            mesh,
            snap_radius_mm=args.snap_radius_mm,
        )
        print(f"1.5. Vector Planar Edge Snapping: {snap_rep}")

    # Bước 2: Phân loại biên hở
    loops_info = classify_boundary_loops(
        mesh, up_vec,
        max_hole_extent_ratio=args.max_hole_ratio,
        preserve_through_holes=args.preserve_through_holes,
    )
    print(f"2. Phân loại biên hở ({len(loops_info)} vòng):")
    for info in loops_info:
        print(f"   - Loop #{info['id']}: kind={info['kind']:10s} | verts={info['vertices']:4d} "
              f"| extent={info['extent_mm']:5.1f}mm ({info['extent_ratio']:.2f}) "
              f"| h_norm={info['height_norm']:.2f} | coh={info['normal_coherence']:.2f} "
              f"| align={info['normal_plane_align']:.2f}")

    # Bước 3: Vá lỗ cục bộ + Fairing cong
    if args.fill_holes:
        mesh, hole_rep = fill_local_holes_with_fairing(mesh, loops_info, fairing_iters=25)
        print(f"3. Vá lỗ cục bộ (Curvature Fairing): {hole_rep}")

    # Bước 4: Bịt đáy phẳng bằng Delaunay 2D
    cad_up_vec = up_vec.copy()
    if args.cap_base:
        loops_after = classify_boundary_loops(
            mesh, up_vec,
            max_hole_extent_ratio=args.max_hole_ratio,
            preserve_through_holes=args.preserve_through_holes,
        )
        mesh, base_rep = cap_base_delaunay(mesh, loops_after, up_vec)
        if base_rep.get("capped") and "base_up_normal" in base_rep:
            cad_up_vec = np.asarray(base_rep["base_up_normal"], dtype=np.float64)
        print(f"4. Bịt đáy phẳng (Delaunay 2D): {base_rep}")

    # Bước 5: Tạo độ dày thành cho vật hở miệng (tùy chọn)
    if args.solidify_mm > 0:
        mesh, sol_rep = solidify_open_shell(mesh, thickness_mm=args.solidify_mm)
        print(f"5. Tạo độ dày thành (Solidify): {sol_rep}")

    # Bước 6: Giảm đa giác tối ưu xuất CAD (tùy chọn)
    if args.target_triangles > 0 and len(mesh.triangles) > args.target_triangles:
        before = len(mesh.triangles)
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=args.target_triangles)
        mesh = _clean_mesh_topology(mesh)
        print(f"6. Quadric Decimation: {before:,} -> {len(mesh.triangles):,} tam giác")

    # Bước 7: Chuẩn hóa hệ tọa độ CAD (+Z up, đáy Z=0, tâm XY=0, tùy chọn mm)
    if args.align_cad or args.export_mm:
        mesh, cad_rep = align_mesh_to_cad_frame(
            mesh,
            up_axis_vec=cad_up_vec,
            scale_to_mm=args.export_mm,
        )
        print(f"7. Chuẩn hóa hệ CAD: {cad_rep}")

    out_path = args.output
    if not out_path:
        stem, _ = os.path.splitext(args.input_mesh)
        out_path = f"{stem}_cad.ply"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(out_path, mesh, write_ascii=False)
    print(f"✅ Hoàn tất trong {time.time() - t0:.2f}s -> {out_path} "
          f"({len(mesh.vertices):,} đỉnh, {len(mesh.triangles):,} tam giác, "
          f"watertight={mesh.is_watertight()})")


if __name__ == "__main__":
    main()
