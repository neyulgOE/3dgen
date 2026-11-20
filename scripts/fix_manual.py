with open("slam_object_builder.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if line.strip() == "submaps.append(active)" and i > 510 and i < 520:
        start_idx = i
    if line.startswith("    elif args.assume_turntable:") and i > 560:
        end_idx = i

if start_idx != -1 and end_idx != -1:
    new_block = """                          submaps.append(active)
                          break

        trusted_indices = [anchor]
        for sm in submaps:
            for idx in sm:
                if idx not in trusted_indices:
                    trusted_indices.append(idx)
        
        if len(trusted_indices) < 3:
            raise RuntimeError("Manual turntable tracking found fewer than 3 verified views.")
"""
    lines = lines[:start_idx] + [new_block] + lines[end_idx:]
    with open("slam_object_builder.py", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Success")
else:
    print(f"Markers not found. start={start_idx}, end={end_idx}")
