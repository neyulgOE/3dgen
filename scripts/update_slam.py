import re

with open("slam_object_builder.py", "r", encoding="utf-8") as f:
    content = f.read()

start_marker = "        submaps.append(active)"
end_marker = "    if len(trusted_indices) < 3:"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

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
            "candidate_pairs": candidate_pairs
        })
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(existing_report, report_file, indent=2)

    export_debug_stage(args.dataset_dir, "final_strict_pool", trusted_indices,
                       optimized_poses, color_np, depth_np, masks,
                       intrinsic_tensor, voxel=args.merge_voxel)

"""
    content = content[:start_idx] + new_block + content[end_idx:]
    with open("slam_object_builder.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Success")
else:
    print("Markers not found.")
