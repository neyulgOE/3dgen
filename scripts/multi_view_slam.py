import open3d as o3d
import numpy as np
import copy
import argparse
from pathlib import Path
import sys
import json

def load_poses(directory):
    poses_file = Path(directory) / "poses.json"
    if not poses_file.exists():
        return None
    
    with open(poses_file, 'r') as f:
        data = json.load(f)
        
    poses = []
    # Assuming cam_0, cam_1, ...
    i = 0
    while f"cam_{i}" in data:
        matrix_flat = data[f"cam_{i}"]
        matrix = np.array(matrix_flat).reshape(4, 4)
        poses.append(matrix)
        i += 1
        
    return poses

def create_frustum_lineset(pose, scale=50.0, color=[0, 1, 0]):
    w = 0.5 * scale
    h = 0.3 * scale
    z = 1.0 * scale
    # Các đỉnh của hình chóp cụt tượng trưng cho camera
    points = [
        [0, 0, 0],
        [w, h, z],
        [-w, h, z],
        [-w, -h, z],
        [w, -h, z]
    ]
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]
    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    # Lật trục Z lại nếu C++ đang chiếu ngược hướng
    # line_set.transform(pose)
    # Tùy thuộc vào thiết lập trục tọa độ, có thể cần lật Z
    return line_set.transform(pose)

def load_point_clouds(directory, voxel_size=0.0):
    pcds = []
    filepaths = sorted(list(Path(directory).glob("cam_*.ply")))
    if not filepaths:
        print(f"No .ply files found in directory: {directory}")
        sys.exit(1)
        
    for filepath in filepaths:
        pcd = o3d.io.read_point_cloud(str(filepath))
        # Nếu voxel_size > 0, downsample để chạy nhanh hơn
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        pcds.append(pcd)
        print(f"Loaded {filepath.name} ({len(pcd.points)} points)")
    return pcds

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, init_transform=np.identity(4)):
    print("  -> Running Point-to-Plane ICP (coarse)...")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    
    print("  -> Running Point-to-Plane ICP (fine)...")
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
        
    return icp_fine.transformation, information_icp, icp_fine.inlier_rmse

def full_registration(pcds, max_correspondence_distance_coarse, max_correspondence_distance_fine, initial_poses=None):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    n_pcds = len(pcds)
    
    # Initialize PoseGraphNodes with the absolute poses from C++ Odometry (if available)
    for i in range(n_pcds):
        initial_pose = np.identity(4)
        if initial_poses is not None and i < len(initial_poses):
            initial_pose = initial_poses[i]
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(initial_pose))
    
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            print(f"Matching view {source_id} to view {target_id}...")
            
            # Calculate initial guess for the transformation FROM source TO target
            # Target * p_target = Source * p_source => p_target = (Target^-1 * Source) * p_source
            init_transform = np.identity(4)
            if initial_poses is not None and source_id < len(initial_poses) and target_id < len(initial_poses):
                source_pose = initial_poses[source_id]
                target_pose = initial_poses[target_id]
                init_transform = np.linalg.inv(target_pose) @ source_pose
                
            transformation_icp, information_icp, rmse = pairwise_registration(
                pcds[source_id], pcds[target_id], 
                max_correspondence_distance_coarse, 
                max_correspondence_distance_fine,
                init_transform)
            print(f"     RMSE: {rmse:.4f}")
            
            uncertain = True
            if target_id == source_id + 1:
                uncertain = False
                
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation_icp,
                                                         information_icp, uncertain=uncertain))
                
    return pose_graph

def main():
    parser = argparse.ArgumentParser(description="Multi-View Point Cloud Registration (ICP) using Open3D")
    parser.add_argument("--dir", type=str, required=True, help="Thư mục chứa các file .ply")
    parser.add_argument("--out", type=str, default="merged_output.ply", help="Tên file kết quả (mặc định: merged_output.ply)")
    parser.add_argument("--voxel", type=float, default=1.0, help="Kích thước Voxel để downsample (mm) - giảm để tăng độ chính xác (mặc định: 1.0)")
    parser.add_argument("--coarse", type=float, default=15.0, help="Khoảng cách tìm kiếm tối đa ban đầu (mm) (mặc định: 15.0)")
    parser.add_argument("--fine", type=float, default=3.0, help="Khoảng cách tìm kiếm tối đa chi tiết (mm) (mặc định: 3.0)")
    args = parser.parse_args()

    print("=== MULTI-VIEW SLAM (ICP) ===")
    pcds = load_point_clouds(args.dir, voxel_size=args.voxel)
    poses = load_poses(args.dir)
    if poses:
        print(f"Loaded {len(poses)} initial poses from poses.json")
    
    if len(pcds) < 2:
        print("At least 2 .ply files are required for registration.")
        sys.exit(1)
        
    print("\n--- Starting Pose Graph Optimization ---")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds, args.coarse, args.fine, poses)
        
        print("\n--- Optimizing Pose Graph (Global Optimization) ---")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=args.fine,
            edge_prune_threshold=0.25,
            preference_loop_closure=2.0,
            reference_node=0)
        
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)

    print("\n--- Creating combined point cloud ---")
    pcds_down = load_point_clouds(args.dir, voxel_size=args.voxel / 2.0)
    pcd_combined = o3d.geometry.PointCloud()
    
    for point_id in range(len(pcds_down)):
        print(f"Transforming view {point_id}...")
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds_down[point_id]
        
    print("\n--- Merging overlapping surfaces ---")
    pcd_combined = pcd_combined.voxel_down_sample(voxel_size=args.voxel / 2.0)
        
    print("\n--- Cleaning up point cloud ---")
    pcd_combined, ind = pcd_combined.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"\nSaved result to: {args.out} ({len(pcd_combined.points)} points)")
    o3d.io.write_point_cloud(args.out, pcd_combined)
    print("DONE!")
    
    print("\nOpening 3D viewer to show result...")
    vis_elements = [pcd_combined]
    
    # Thêm mô hình camera vào view
    for point_id in range(len(pcds_down)):
        # pose_graph.nodes[point_id].pose chính là pose của camera thứ i
        frustum = create_frustum_lineset(pose_graph.nodes[point_id].pose, scale=100.0)
        vis_elements.append(frustum)
        
    # Coordinate frame tại gốc 0,0,0
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    vis_elements.append(coord)
    
    o3d.visualization.draw_geometries(vis_elements, window_name="Multi-View ICP Result")

if __name__ == "__main__":
    main()
