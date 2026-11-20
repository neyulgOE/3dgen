import open3d as o3d
import numpy as np
import argparse
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Build 3D Mesh from Point Cloud using Poisson Surface Reconstruction")
    parser.add_argument("input_ply", help="Path to the input PLY file (e.g., final_optimized_model.ply)")
    parser.add_argument("--depth", type=int, default=9, help="Depth of the Poisson tree (higher is more detailed, default: 9)")
    parser.add_argument("--density-thresh", type=float, default=0.1, help="Quantile threshold to remove low density vertices (0.0 to 1.0)")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel size for downsampling (default: 0.0 = disabled)")
    parser.add_argument("--output", type=str, default="", help="Custom output path for the OBJ file")
    args = parser.parse_args()

    input_path = args.input_ply
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    if os.path.isdir(input_path):
        import glob
        ply_files = sorted(glob.glob(os.path.join(input_path, "cloud_*.ply")))
        if len(ply_files) == 0:
            print("Error: No cloud_*.ply found in directory!")
            sys.exit(1)
            
        print(f"1. Loading and Aligning {len(ply_files)} Point Clouds using ICP Registration...")
        pcd = o3d.io.read_point_cloud(ply_files[0])
        # Downsample initially for speed
        pcd = pcd.voxel_down_sample(voxel_size=1.0)
        
        for i in range(1, len(ply_files)):
            source = o3d.io.read_point_cloud(ply_files[i])
            source = source.voxel_down_sample(voxel_size=1.0)
            
            print(f"   - ICP Aligning {os.path.basename(ply_files[i])} to global model...")
            # Use Point-to-Point ICP with large correspondence distance (500mm) to fix severe VIO drift
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source, pcd, 500.0, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
            )
            source.transform(reg_p2p.transformation)
            pcd += source
            # Downsample accumulated cloud to prevent exponential growth
            pcd = pcd.voxel_down_sample(voxel_size=1.0)
            
        print(f"   - ICP Alignment complete. Total points: {len(pcd.points)}")
    else:
        print(f"1. Loading Point Cloud: {input_path}")
        pcd = o3d.io.read_point_cloud(input_path)
        
        if len(pcd.points) == 0:
            print("Error: Point cloud is empty!")
            sys.exit(1)

        print(f"   - Loaded {len(pcd.points)} points.")

    print("2. Pre-processing Point Cloud...")
    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"   - Downsampled to {len(pcd.points)} points using voxel size {args.voxel_size}.")
        
    # Loại bỏ điểm nhiễu (outliers) cực kỳ quan trọng để Poisson không tạo ra cái bong bóng khổng lồ
    print("   - Removing statistical outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    print(f"   - Kept {len(pcd.points)} clean points.")

    print("3. Estimating & Orienting Normals...")
    # Bán kính 50mm (5cm) vì đơn vị của Point Cloud đang là Milimet.
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=50.0, max_nn=30))
    # Đồng bộ hóa hướng pháp tuyến
    pcd.orient_normals_consistent_tangent_plane(k=30)

    print(f"4. Running Poisson Surface Reconstruction (Depth={args.depth})...")
    # Chạy nội suy Poisson. Hàm này trả về Mesh và mảng Density (mật độ điểm)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    print(f"   - Generated Raw Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")

    print(f"5. Filtering Low Density Artifacts (Threshold={args.density_thresh})...")
    # Thuật toán Poisson thường sinh ra các màng bọc ảo ở những vùng không có điểm.
    # Dựa vào mảng densities, ta cắt bỏ các đỉnh có mật độ điểm lân cận thấp.
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, args.density_thresh)
    vertices_to_remove = densities < density_threshold
    
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"   - Filtered Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")

    # Dọn dẹp lại Mesh sau khi cắt
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    if pcd.has_colors():
        print("   - Painting mesh with original point cloud colors...")
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_colors = []
        for v in mesh.vertices:
            _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
            mesh_colors.append(pcd.colors[idx[0]])
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(mesh_colors))

    # Khắc phục lỗi NaN: Xóa các bề mặt, góc cạnh rác do Poisson tạo ra trước khi làm mịn
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    print("6. Smoothing Mesh (Preserving Details)...")
    # Thay đổi thuật toán làm mịn: Taubin hay sinh lỗi NaN, dùng Simple Smooth sẽ ổn định 100%
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    mesh.compute_vertex_normals()

    # Xuất file (dùng định dạng PLY để giữ lại toàn bộ màu sắc RGB của Mesh)
    if args.output:
        output_obj = args.output
    else:
        output_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_obj = os.path.join(output_dir, f"{base_name}_mesh.ply")
    
    print(f"7. Saving Output to: {output_obj}")
    o3d.io.write_triangle_mesh(output_obj, mesh, write_ascii=False)
    print("✅ Done! Tái tạo bề mặt thành công.")

if __name__ == "__main__":
    main()
