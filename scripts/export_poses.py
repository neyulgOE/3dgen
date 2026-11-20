import re

with open("slam_object_builder.py", "r", encoding="utf-8") as f:
    content = f.read()

target = """    out = (args.output if args.output
           else os.path.join(args.dataset_dir, "scanned_object_slam_mesh.ply"))
    o3d.io.write_triangle_mesh(out, mesh, write_ascii=False)
    print(f"\\n✅ Saved: {out}")
    print(f"   Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")"""

replacement = """    out = (args.output if args.output
           else os.path.join(args.dataset_dir, "scanned_object_slam_mesh.ply"))
    o3d.io.write_triangle_mesh(out, mesh, write_ascii=False)
    print(f"\\n✅ Saved Mesh: {out}")
    print(f"   Vertices: {len(mesh.vertices):,}  Triangles: {len(mesh.triangles):,}")

    # ── 10. Export Poses for MVS Texturing ────────────────────────
    # Save the optimized global poses to a json so that MVS texturing 
    # can bake the texture into the PLY/OBJ.
    poses_out = os.path.join(args.dataset_dir, "global_poses_optimized.json")
    exported_poses = {}
    for idx, p in enumerate(optimized_poses):
        exported_poses[idx] = p.tolist() if isinstance(p, np.ndarray) else p
    
    with open(poses_out, "w", encoding="utf-8") as f:
        json.dump(exported_poses, f, indent=2)
    print(f"✅ Saved Optimized Poses: {poses_out}")
"""

new_content = content.replace(target, replacement)
if new_content != content:
    with open("slam_object_builder.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Success")
else:
    print("Could not find target string to replace.")
