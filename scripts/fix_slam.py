with open("slam_object_builder.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.startswith("        submaps.append(active)") and i > 600:
        start_idx = i
    if line.startswith("    if len(trusted_indices) < 3:") and i > 600:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    new_block = """        submaps.append(active)
        print(f"      Submaps: {[len(submap) for submap in submaps]}; rejected={rejected}")

        for sm_id, sm in enumerate(submaps):
            export_debug_stage(args.dataset_dir, f"part{sm_id + 1}_raw", sm,
                               optimized_poses, color_np, depth_np, masks,
                               intrinsic_tensor, voxel=args.merge_voxel)

        # ── STEP 5: Xóa nối submap tuần tự, sinh Candidate pairs ─────────────
        print("   -> (STEP 5) Sinh Candidate pairs thay thế cho Submap tuần tự...")
        from candidate_selection import generate_candidate_pairs
        
        valid_frames = set()
        for sm in submaps:
            valid_frames.update(sm)
            
        candidate_pairs = generate_candidate_pairs(
            valid_frames=valid_frames,
            optimized_poses=optimized_poses,
            color_np=color_np,
            max_temporal=3,
            spatial_dist_thr=0.15,
            spatial_angle_thr=30.0,
            num_visual_loops=3
        )
        print(f"      Sinh được {len(candidate_pairs)} candidate pairs.")
        
        # ── STEP 6: Hoàn thiện View Graph & PGO ─────────────
        print("   -> (STEP 6) Xây dựng View Graph và chạy Pose Graph Optimization...")
        from pose_graph_optimizer import build_and_optimize_pose_graph
        optimized_poses, pgo_diagnostics = build_and_optimize_pose_graph(
            valid_frames=valid_frames,
            candidate_pairs=candidate_pairs,
            optimized_poses=optimized_poses,
            color_np=color_np,
            depth_np=depth_np,
            masks=masks,
            intrinsic_tensor=intrinsic_tensor,
            voxel_size=args.merge_voxel,
            min_icp_fitness=args.merge_min_fitness,
            max_icp_rmse=args.merge_max_rmse,
            max_cycle_translation=args.max_cycle_translation,
            max_cycle_rotation=args.max_cycle_rotation,
            use_colored_icp=False
        )
        
        trusted_indices = sorted(list(valid_frames))
        world_verified_indices = set(trusted_indices)
        
        try:
            with open(report_path, "r", encoding="utf-8") as rf:
                existing_report = json.load(rf)
        except Exception:
            existing_report = {}
            
        existing_report.update({
            "submaps": submaps,
            "trusted_indices": trusted_indices,
            "candidate_pairs": candidate_pairs,
            "pgo_diagnostics": pgo_diagnostics
        })
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(existing_report, report_file, indent=2)

    export_debug_stage(args.dataset_dir, "final_strict_pool", trusted_indices,
                       optimized_poses, color_np, depth_np, masks,
                       intrinsic_tensor, voxel=args.merge_voxel)

"""
    lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    with open("slam_object_builder.py", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Success")
else:
    print(f"Markers not found. start={start_idx}, end={end_idx}")
