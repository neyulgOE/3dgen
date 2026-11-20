import os
import sys
import glob
import json
import argparse
import numpy as np
import open3d as o3d
import gc
import csv
import copy
from scipy.spatial.transform import Rotation as R

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def load_intrinsics(dataset_dir):
    intr_file = os.path.join(dataset_dir, "intrinsics.json")
    if os.path.exists(intr_file):
        with open(intr_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}

def load_imu_data(dataset_dir):
    imu_file = os.path.join(dataset_dir, "imu.csv")
    gyro_data = []
    if not os.path.exists(imu_file):
        return gyro_data
        
    with open(imu_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and row[1] == 'gyro':
                t_ns = int(row[0])
                wx = float(row[2])
                wy = float(row[3])
                wz = float(row[4])
                gyro_data.append((t_ns, wx, wy, wz))
    gyro_data.sort(key=lambda x: x[0])
    return gyro_data

def integrate_imu(t1_ns, t2_ns, gyro_data):
    if not gyro_data or t1_ns == 0 or t2_ns == 0 or t1_ns >= t2_ns:
        return np.eye(3)
        
    relevant_gyro = [g for g in gyro_data if t1_ns <= g[0] <= t2_ns]
    if len(relevant_gyro) < 2:
        return np.eye(3)
        
    delta_R = np.eye(3)
    for i in range(1, len(relevant_gyro)):
        t_prev, wx_prev, wy_prev, wz_prev = relevant_gyro[i-1]
        t_curr, wx_curr, wy_curr, wz_curr = relevant_gyro[i]
        
        dt_sec = (t_curr - t_prev) * 1e-9
        if dt_sec <= 0:
            continue
        
        wx = (wx_prev + wx_curr) / 2.0
        wy = (wy_prev + wy_curr) / 2.0
        wz = (wz_prev + wz_curr) / 2.0
        
        rot_vec = np.array([wx, wy, wz]) * dt_sec
        r = R.from_rotvec(rot_vec).as_matrix()
        delta_R = delta_R @ r
        
    return delta_R

def load_timestamps(dataset_dir):
    ts_file = os.path.join(dataset_dir, "timestamps.csv")
    timestamps_dict = {}
    if os.path.exists(ts_file):
        with open(ts_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    timestamps_dict[row[0]] = int(row[1])
    return timestamps_dict

def _resolve_cam_key(filename, poses_data):
    stem = os.path.splitext(filename)[0]
    try:
        idx = int(stem)
        candidates = [f"cam_{idx:03d}", f"cam_{idx}", f"cam_{stem[-3:]}"]
    except ValueError:
        candidates = [f"cam_{stem}", f"cam_{stem[-3:]}"]
    for key in candidates:
        if key in poses_data:
            return key
    return None

def load_dataset(dataset_dir):
    color_dir = os.path.join(dataset_dir, "rgb")
    depth_raw_dir = os.path.join(dataset_dir, "depth_raw")
    
    if not os.path.exists(color_dir) or not os.path.exists(depth_raw_dir):
        print("Lỗi: Không tìm thấy thư mục rgb hoặc depth_raw.")
        sys.exit(1)
        
    poses_file = os.path.join(dataset_dir, "poses.json")
    if not os.path.exists(poses_file):
        print("Lỗi: Không tìm thấy poses.json")
        sys.exit(1)
        
    with open(poses_file, "r", encoding='utf-8') as f:
        poses_data = json.load(f)
        
    timestamps_dict = load_timestamps(dataset_dir)
        
    rgbd_images = []
    poses = []
    timestamps = []
    
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith(".png")])
    for filename in color_files:
        cam_key = _resolve_cam_key(filename, poses_data)
        if cam_key is None:
            continue
            
        file = os.path.join(color_dir, filename)
        if not os.path.exists(file):
            continue
            
        color_img = o3d.io.read_image(file)
        depth_raw_file = os.path.join(depth_raw_dir, filename)
        if not os.path.exists(depth_raw_file):
            continue
        depth_raw_img = o3d.io.read_image(depth_raw_file)
        rgbd_raw = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_raw_img, depth_scale=1000.0, depth_trunc=3.0, convert_rgb_to_intensity=False)
            
        pose_array = poses_data[cam_key]
        T_base_cam = np.array(pose_array, dtype=np.float64).reshape((4, 4))
        
        rgbd_images.append(rgbd_raw)
        poses.append(T_base_cam)
        timestamps.append(timestamps_dict.get(cam_key, 0))
        print(f"Loaded {cam_key} (from {filename}) - Odometry & Integration: raw")
        
    return rgbd_images, poses, timestamps

def extract_planes(pcd, n_planes=5, dist_thresh=0.006, min_inliers=50):
    planes = []
    remaining = pcd
    for _ in range(n_planes):
        if len(remaining.points) < min_inliers:
            break
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=dist_thresh, ransac_n=3, num_iterations=500)
        except Exception:
            break
            
        if len(inliers) < min_inliers:
            break
            
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        inlier_cloud = remaining.select_by_index(inliers)
        centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
        planes.append({'normal': normal, 'd': d, 'centroid': centroid, 'cloud': inlier_cloud})
        remaining = remaining.select_by_index(inliers, invert=True)
        
    return planes

def project_to_plane(pcd, plane_model):
    points = np.asarray(pcd.points)
    n = plane_model['normal']
    d = plane_model['d']
    dist = np.dot(points, n) + d
    projected = points - np.outer(dist, n)
    proj_pcd = o3d.geometry.PointCloud()
    proj_pcd.points = o3d.utility.Vector3dVector(projected)
    if pcd.has_colors():
        proj_pcd.colors = pcd.colors
    return proj_pcd

def compute_fpfh(pcd_down, voxel_size):
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))
    return fpfh

def try_relocalize(pcd_i_down, fpfh_i, keyframes, voxel_size, min_frame_gap=15, current_idx=0):
    best = None
    for kf in keyframes:
        if current_idx - kf['index'] < min_frame_gap:
            continue
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            pcd_i_down, kf['pcd_down'], fpfh_i, kf['fpfh'],
            mutual_filter=True,
            max_correspondence_distance=voxel_size * 1.5,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

        if result_ransac.fitness < 0.2:
            continue

        icp_refine = o3d.pipelines.registration.registration_icp(
            pcd_i_down, kf['pcd_down'], voxel_size * 1.5, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))

        if icp_refine.fitness < 0.35:
            continue

        T_world_i = icp_refine.transformation
        if best is None or icp_refine.fitness > best[2]:
            best = (True, T_world_i, icp_refine.fitness, kf['index'])

    if best is None:
        return False, None, 0.0, -1
    return best

def compute_alpha_shape_mesh(pcd, alpha=0.1):
    try:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        return mesh
    except Exception:
        return None

def sample_points_from_mesh(mesh, number_of_points=10000):
    if mesh is None or len(mesh.triangles) == 0:
        return o3d.geometry.PointCloud()
    return mesh.sample_points_uniformly(number_of_points=number_of_points)

def fill_planar_holes(pcd, voxel_size=0.002, n_planes=1, alpha=0.1):
    print("--- Đang thực hiện Planar Hole-Filling (Alpha Shape) ---")
    planes = extract_planes(pcd, n_planes=n_planes, dist_thresh=voxel_size*3, min_inliers=1000)
    if not planes:
        return pcd
        
    filled_pcd = o3d.geometry.PointCloud()
    remaining_points = np.asarray(pcd.points)
    remaining_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    all_inlier_indices = set()
    
    for plane in planes:
        inlier_pcd = plane['cloud']
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        for pt in np.asarray(inlier_pcd.points):
            _, idx, _ = kdtree.search_knn_vector_3d(pt, 1)
            all_inlier_indices.add(idx[0])
            
        proj_pcd = project_to_plane(inlier_pcd, plane)
        mesh = compute_alpha_shape_mesh(proj_pcd, alpha=alpha)
        
        if mesh is not None and len(mesh.triangles) > 0:
            area = mesh.get_surface_area()
            num_points = int(area / ((voxel_size/2.0)**2))
            if num_points > 0:
                sampled_pcd = sample_points_from_mesh(mesh, number_of_points=num_points)
                filled_pcd += sampled_pcd
        else:
            filled_pcd += proj_pcd
            
    all_indices = set(range(len(remaining_points)))
    outlier_indices = list(all_indices - all_inlier_indices)
    
    if len(outlier_indices) > 0:
        outlier_pcd = o3d.geometry.PointCloud()
        outlier_pcd.points = o3d.utility.Vector3dVector(remaining_points[outlier_indices])
        if remaining_colors is not None:
            outlier_pcd.colors = o3d.utility.Vector3dVector(remaining_colors[outlier_indices])
        filled_pcd += outlier_pcd
        
    filled_pcd = filled_pcd.voxel_down_sample(voxel_size=voxel_size)
    return filled_pcd

def main():
    parser = argparse.ArgumentParser(description="TMVisionary multi-view Causal SLAM")
    parser.add_argument("dataset_dir")
    parser.add_argument("--voxel-size", type=float, default=4.0, help="TSDF/registration voxel size in mm")
    parser.add_argument("--low-memory", action="store_true")
    parser.add_argument("--skip-slam", action="store_true")
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--fitness-min", type=float, default=0.25)
    parser.add_argument("--track-loss-limit", type=int, default=2)
    parser.add_argument("--keyframe-every", type=int, default=8)
    parser.add_argument("--global-optimize", action=argparse.BooleanOptionalAction, default=True,
                         help="Xây Pose Graph và chạy global optimization để khép vòng 360° trước khi tích hợp TSDF.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    voxel_size = args.voxel_size / 1000.0

    intr = load_intrinsics(dataset_dir)
    rgbd_images, init_poses, timestamps = load_dataset(dataset_dir)
    n_frames = len(rgbd_images)
    if n_frames < 2:
        print("Cần ít nhất 2 frame để chạy SLAM!")
        sys.exit(1)

    gyro_data = load_imu_data(dataset_dir)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        int(intr["width"]), int(intr["height"]),
        float(intr["fx"]), float(intr["fy"]),
        float(intr["cx"]), float(intr["cy"]))

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=voxel_size * 5.0,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    optimized_poses = []
    
    if args.skip_slam:
        for i in range(n_frames):
            volume.integrate(rgbd_images[i], intrinsic, np.linalg.inv(init_poses[i]))
    else:
        optimized_poses = [init_poses[0]]
        diagnostics = []

        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(init_poses[0]))
        rgbd_keep = [None] * n_frames
        if args.global_optimize:
            rgbd_keep[0] = rgbd_images[0]

        T_rel_prev = np.eye(4)
        track_loss_streak = 0
        keyframes = []
        RELOC_VOXEL = 0.03

        def maybe_add_keyframe(pcd_full, T_world, frame_idx):
            if frame_idx % max(1, args.keyframe_every) != 0:
                return
            kf_down = pcd_full.voxel_down_sample(RELOC_VOXEL)
            if len(kf_down.points) < 50:
                return
            kf_fpfh = compute_fpfh(kf_down, RELOC_VOXEL)
            kf_world = copy.deepcopy(kf_down)
            kf_world.transform(T_world)
            keyframes.append({'pcd_down': kf_world, 'fpfh': kf_fpfh,
                               'T_world': T_world, 'index': frame_idx})
        
        pcd_history = []
        pcd_0_full = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_images[0], intrinsic)
        pcd_0 = pcd_0_full.voxel_down_sample(0.02)
        pcd_0.transform(init_poses[0])
        pcd_history.append(pcd_0)
        maybe_add_keyframe(pcd_0_full, init_poses[0], 0)

        for i in range(1, n_frames):
            rgbd_raw_i    = rgbd_images[i]
            rgbd_raw_prev = rgbd_images[i - 1]
            print(f"\n[Frame {i}/{n_frames-1}]", end=" ", flush=True)

            R_imu = integrate_imu(timestamps[i - 1], timestamps[i], gyro_data)
            T_init = np.eye(4)
            T_init[:3, :3] = R_imu
            if track_loss_streak == 0:
                T_init[:3, 3] = T_rel_prev[:3, 3]

            success, T_rel, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                rgbd_raw_i, rgbd_raw_prev, intrinsic, T_init,
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                o3d.pipelines.odometry.OdometryOption(
                    depth_diff_max=0.1, depth_min=0.1, depth_max=2.5))
    
            local_map_full = o3d.geometry.PointCloud()
            for p in pcd_history[-5:]:
                local_map_full += p
    
            pcd_i_full = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_raw_i, intrinsic)
            T_world_guess = (optimized_poses[-1] @ T_rel) if success else (optimized_poses[-1] @ T_init)
    
            current_transformation = T_world_guess
            vs_c, dist_c, iters_c = 0.04, 0.5, 50
            pcd_i_down_c = pcd_i_full.voxel_down_sample(vs_c)
            pcd_i_down_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs_c*2.5, max_nn=30))
            local_map_down_c = local_map_full.voxel_down_sample(vs_c)
            local_map_down_c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs_c*2.5, max_nn=30))
            icp_c = o3d.pipelines.registration.registration_icp(
                pcd_i_down_c, local_map_down_c, dist_c, current_transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters_c))
            current_transformation = icp_c.transformation
            final_icp_fitness = icp_c.fitness
            final_icp_rmse = icp_c.inlier_rmse

            vs_f, dist_f, iters_f = 0.02, 0.08, 50
            pcd_i_down_f = pcd_i_full.voxel_down_sample(vs_f)
            pcd_i_down_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs_f*2.5, max_nn=30))
            local_map_down_f = local_map_full.voxel_down_sample(vs_f)
            local_map_down_f.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vs_f*2.5, max_nn=30))
            try:
                icp_f = o3d.pipelines.registration.registration_colored_icp(
                    pcd_i_down_f, local_map_down_f, dist_f, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters_f))
                current_transformation = icp_f.transformation
                final_icp_fitness = icp_f.fitness
                final_icp_rmse = icp_f.inlier_rmse
            except RuntimeError:
                pass

            pcd_i_down_fine = pcd_i_down_f
            pcd_i = pcd_i_full.voxel_down_sample(0.02)

            T_world_eval = np.copy(current_transformation)
            T_rel_eval = np.linalg.inv(optimized_poses[-1]) @ T_world_eval
            trans_jump = np.linalg.norm(T_rel_eval[:3, 3])
            try:
                rot_jump = np.linalg.norm(R.from_matrix(T_rel_eval[:3, :3]).as_rotvec(degrees=True))
            except ValueError:
                rot_jump = 0.0

            state = "GOOD"
            if final_icp_fitness < args.fitness_min:
                state = "LOST"
            elif final_icp_fitness < 0.4 or final_icp_rmse > 0.02 or trans_jump > 0.1 or rot_jump > 10.0:
                state = "WARNING"
                
            diagnostics.append({
                "frame": i,
                "timestamp": timestamps[i],
                "fitness": round(float(final_icp_fitness), 4),
                "rmse": round(float(final_icp_rmse), 4),
                "trans_jump": round(float(trans_jump), 4),
                "rot_jump": round(float(rot_jump), 4),
                "state": state
            })

            relocalized = False
            if final_icp_fitness > args.fitness_min:
                T_world_final = np.copy(current_transformation)
                T_rel_final = np.linalg.inv(optimized_poses[-1]) @ T_world_final
                optimized_poses.append(T_world_final)
                T_rel_prev = T_rel_final
                track_loss_streak = 0
                print(f"Multi-Scale+Colored ICP OK fitness={final_icp_fitness:.3f}")

                if args.global_optimize:
                    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world_final))
                    edge_transform = np.linalg.inv(T_rel_final)
                    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                        pcd_i_down_f, local_map_down_f, dist_f, edge_transform)
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                        i - 1, i, edge_transform, information, uncertain=False))

                    # Định kỳ dò Loop Closure với các keyframe cũ (đặc biệt là keyframe đầu vòng quay)
                    if i % max(1, args.keyframe_every) == 0 and len(keyframes) >= 3:
                        fpfh_i_loop = compute_fpfh(pcd_i_down_fine, RELOC_VOXEL)
                        loop_ok, T_world_loop, loop_fit, kf_idx = try_relocalize(
                            pcd_i_down_fine, fpfh_i_loop, keyframes, vs_f, min_frame_gap=16, current_idx=i)
                        if loop_ok:
                            T_world_kf = optimized_poses[kf_idx]
                            T_i_to_kf_local = np.linalg.inv(T_world_kf) @ T_world_loop
                            info_loop = np.eye(6) * (loop_fit * 50.0)
                            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                i, kf_idx, T_i_to_kf_local, info_loop, uncertain=True))
                            print(f"   🔗 [Loop Edge] Frame {i} <-> Keyframe #{kf_idx} (fitness={loop_fit:.3f})")

                    if (i % 15 == 0) and len(keyframes) > 0:
                        curr_pos = T_world_final[:3, 3]
                        for kf in keyframes:
                            kf_idx = kf['index']
                            if (i - kf_idx) < 40:
                                continue
                            if np.linalg.norm(curr_pos - kf['T_world'][:3, 3]) < 0.15:
                                icp_loop = o3d.pipelines.registration.registration_icp(
                                    pcd_i_down_f, kf['pcd_down'], dist_f * 2.0, T_world_final,
                                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                                if icp_loop.fitness >= 0.40 and icp_loop.inlier_rmse <= dist_f * 1.5:
                                    T_world_loop = icp_loop.transformation
                                    T_i_to_kf_local = np.linalg.inv(optimized_poses[kf_idx]) @ T_world_loop
                                    info_loop = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                                        pcd_i_down_f, kf['pcd_down'], dist_f * 2.0, T_world_loop)
                                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                        i, kf_idx, T_i_to_kf_local, info_loop, uncertain=True))
                                    print(f"   🔗 [Spatial Loop Edge] Frame {i} <-> Keyframe #{kf_idx} "
                                          f"(fitness={icp_loop.fitness:.3f}, rmse={icp_loop.inlier_rmse*1000:.1f}mm)")
                                    break
            else:
                track_loss_streak += 1
                reloc_ok = False
                if track_loss_streak >= args.track_loss_limit and keyframes:
                    fpfh_i_fine = compute_fpfh(pcd_i_down_fine, RELOC_VOXEL)
                    reloc_ok, T_world_reloc, reloc_fitness, kf_idx = try_relocalize(
                        pcd_i_down_fine, fpfh_i_fine, keyframes, vs_f, min_frame_gap=1, current_idx=i)
                    if reloc_ok:
                        T_world_final = T_world_reloc
                        T_rel_prev = np.linalg.inv(optimized_poses[-1]) @ T_world_final
                        optimized_poses.append(T_world_final)
                        track_loss_streak = 0
                        relocalized = True
                        if args.global_optimize:
                            pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world_final))
                            T_world_kf = optimized_poses[kf_idx]
                            T_i_to_kf_local = np.linalg.inv(T_world_kf) @ T_world_final
                            info_loop = np.eye(6) * (reloc_fitness * 50.0)
                            pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                                i, kf_idx, T_i_to_kf_local, info_loop, uncertain=True))

                if not reloc_ok:
                    T_world_final = T_world_guess
                    optimized_poses.append(T_world_final)
                    T_rel_prev = np.linalg.inv(optimized_poses[-2]) @ T_world_final
                    if args.global_optimize:
                        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(T_world_final))
                        edge_transform = np.linalg.inv(T_rel_prev)
                        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(
                            i - 1, i, edge_transform, np.eye(6) * 0.1, uncertain=True))
    
            pcd_i_world = copy.deepcopy(pcd_i)
            pcd_i_world.transform(T_world_final)
            pcd_history.append(pcd_i_world)
            if len(pcd_history) > 10:
                pcd_history.pop(0)

            if final_icp_fitness > args.fitness_min or relocalized:
                maybe_add_keyframe(pcd_i_full, T_world_final, i)
    
            if args.global_optimize:
                rgbd_keep[i - 1] = rgbd_raw_prev
                rgbd_images[i - 1] = None
            else:
                volume.integrate(rgbd_images[i - 1], intrinsic, np.linalg.inv(optimized_poses[i - 1]))
                rgbd_images[i - 1] = None

            del pcd_i, local_map_full
            gc.collect()

        if args.global_optimize:
            rgbd_keep[n_frames - 1] = rgbd_images[n_frames - 1]
            rgbd_images[n_frames - 1] = None

            print(f"\n--- Pose Graph Global Optimization ({len(pose_graph.nodes)} nodes, "
                  f"{len(pose_graph.edges)} edges) ---")
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                o3d.pipelines.registration.GlobalOptimizationOption(
                    max_correspondence_distance=voxel_size * 1.5,
                    edge_prune_threshold=0.25,
                    preference_loop_closure=2.0,
                    reference_node=0))
            optimized_poses = [node.pose for node in pose_graph.nodes]

            for k in range(n_frames):
                if rgbd_keep[k] is not None:
                    volume.integrate(rgbd_keep[k], intrinsic, np.linalg.inv(optimized_poses[k]))
                    rgbd_keep[k] = None
            gc.collect()
        else:
            if not args.skip_slam and n_frames > 0 and rgbd_images[-1] is not None:
                volume.integrate(rgbd_images[-1], intrinsic, np.linalg.inv(optimized_poses[-1]))
                rgbd_images[-1] = None

    final_pcd = volume.extract_point_cloud()
    del volume
    gc.collect()

    if len(final_pcd.points) == 0:
        print("Lỗi: TSDF trả về mây điểm rỗng!")
        sys.exit(1)
        
    final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    try:
        labels = np.array(final_pcd.cluster_dbscan(eps=voxel_size * 8.0, min_points=30))
        if len(labels) > 0:
            counts = np.bincount(labels[labels >= 0])
            valid_labels = np.where(counts > 200)[0]
            mask = np.isin(labels, valid_labels)
            final_pcd = final_pcd.select_by_index(np.where(mask)[0])
    except Exception:
        pass
    
    if not args.low_memory:
        final_pcd = fill_planar_holes(final_pcd, voxel_size=voxel_size, n_planes=1, alpha=0.1)
            
    final_pcd.scale(1000.0, center=(0, 0, 0))
    
    if not args.skip_slam:
        diag_path = os.path.join(dataset_dir, "tracking_health.csv")
        try:
            with open(diag_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["frame", "timestamp", "fitness", "rmse", "trans_jump", "rot_jump", "state"])
                writer.writeheader()
                for row in diagnostics:
                    writer.writerow(row)
        except Exception:
            pass
    
    output_path = os.path.join(dataset_dir, "final_optimized_model.ply")
    o3d.io.write_point_cloud(output_path, final_pcd)
    print(f"✅ Đã lưu mô hình tối ưu tại: {output_path}")

    xyz = np.asarray(final_pcd.points, dtype=np.float32)
    rgb = (np.asarray(final_pcd.colors) * 255.0).astype(np.uint8)
    bin_path = os.path.join(dataset_dir, "final_optimized_model.bin")
    with open(bin_path, "wb") as f:
        f.write(np.array([len(xyz)], dtype=np.int32).tobytes())
        f.write(xyz.tobytes())
        f.write(rgb.tobytes())
        
    if not args.no_vis:
        o3d.visualization.draw_geometries([final_pcd], window_name="TMVisionary SLAM Result")

if __name__ == "__main__":
    main()