import os
import sys
import time
import json
import math
import shutil
import subprocess
import numpy as np
import cv2
import open3d as o3d
import torch

def _resolve_compute_device():
    """Pick CUDA if this Open3D build actually supports it, else CPU.

    FIX: see geometry_utils._resolve_compute_device -- the hardcoded
    Device("cuda:0") crashed on any Open3D build without
    BUILD_CUDA_MODULE=ON. Same fix applied here since this module
    independently redeclares O3D_DEVICE_GPU rather than reusing
    geometry_utils's.
    """
    try:
        if o3d.core.cuda.device_count() > 0:
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return o3d.core.Device("CPU:0")


O3D_DEVICE_GPU = _resolve_compute_device()
O3D_DEVICE_CPU = o3d.core.Device("CPU:0")

def _find_prompt_frame(color_np, yolo_model, min_confidence=0.65):
    """Tìm frame đầu tiên YOLO detect được object.
    Trả về (frame_idx, points [[x,y],...], labels [1,...], bbox hoặc None).
    Point prompt là tâm YOLO bbox. Fallback về tâm ảnh frame 0 nếu YOLO fail.
    """
    for idx, img in enumerate(color_np):
        if yolo_model is None:
            h, w = img.shape[:2]
            return idx, [[w // 2, h // 2]], [1], None
        res = yolo_model(img, verbose=False)
        boxes = res[0].boxes
        if boxes is None:
            continue
        best_area, best_box = -1, None
        for i in range(len(boxes)):
            conf = float(boxes.conf[i].item())
            if int(boxes.cls[i].item()) == 0 and conf >= min_confidence:
                b = boxes.xyxy[i].cpu().numpy().flatten()
                area = (b[2] - b[0]) * (b[3] - b[1])
                if area > best_area:
                    best_area = area
                    best_box = b
        if best_box is not None:
            cx = int((best_box[0] + best_box[2]) / 2)
            cy = int((best_box[1] + best_box[3]) / 2)
            return idx, [[cx, cy]], [1], [int(x) for x in best_box]
    # Fallback
    h, w = color_np[0].shape[:2]
    return 0, [[w // 2, h // 2]], [1], None

def segment_all_frames_sam2(color_np, sam2_predictor, yolo_model,
                             min_confidence=0.65, min_px=300):
    """Segment toàn bộ frames bằng SAM2 VideoPredictor."""
    import torch
    import tempfile
    import shutil

    n = len(color_np)
    H, W = color_np[0].shape[:2]

    tmp_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        print(f"   SAM2: writing {n} frames to temp dir...")
        for i, img in enumerate(color_np):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(tmp_dir, f"{i:06d}.jpg"), rgb)

        prompt_idx, pts, labels, yolo_box = _find_prompt_frame(
            color_np, yolo_model, min_confidence)
        print(f"   SAM2: prompt frame={prompt_idx}, points={pts}, yolo_box={yolo_box}")

        raw_masks = {}
        chunk_size = 150
        
        last_chunk_mask = None
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(n, chunk_start + chunk_size)
            chunk_n = chunk_end - chunk_start
            
            # Find prompt for this chunk
            chunk_color_np = color_np[chunk_start:chunk_end]
            
            c_prompt_idx, pts, labels, yolo_box = _find_prompt_frame(
                chunk_color_np, yolo_model, min_confidence)
                
            # If this is not the first chunk and YOLO failed (box is None), try to use the last mask
            if chunk_start > 0 and yolo_box is None and last_chunk_mask is not None:
                ys, xs = np.where(last_chunk_mask > 0)
                if len(ys) > 0:
                    pts = [[int(np.mean(xs)), int(np.mean(ys))]]
                    c_prompt_idx = 0  # first frame of this chunk
                    print(f"   SAM2 [Chunk {chunk_start}]: YOLO failed, using last mask centroid {pts}")
            
            print(f"   SAM2 [Chunk {chunk_start}-{chunk_end}]: prompt idx={c_prompt_idx}, pts={pts}, box={yolo_box}")
            
            # The files are named 000000.jpg to 000350.jpg, but SAM2 expects them to be in a directory.
            # We can point sam2_predictor to a sub-directory containing just this chunk's frames,
            # but sam2_predictor.init_state loads all images in the folder.
            # So we create a chunk-specific temp dir!
            chunk_tmp_dir = os.path.join(tmp_dir, f"chunk_{chunk_start}")
            os.makedirs(chunk_tmp_dir, exist_ok=True)
            for i in range(chunk_start, chunk_end):
                # Copy from the main tmp_dir or just rename? Let's just create symlinks or copy.
                src = os.path.join(tmp_dir, f"{i:06d}.jpg")
                dst = os.path.join(chunk_tmp_dir, f"{(i - chunk_start):06d}.jpg")
                shutil.copy2(src, dst)
            
            inference_state = sam2_predictor.init_state(video_path=chunk_tmp_dir)
            sam2_predictor.reset_state(inference_state)
            
            # BUG CŨ: yolo_box được tính (_find_prompt_frame trả về cả box),
            # nhưng KHÔNG BAO GIỜ thực sự truyền cho SAM2 — chỉ point (tâm bbox)
            # được dùng làm prompt. Một điểm mồi duy nhất khiến SAM2 dễ chỉ khoanh
            # phần thân cốc (khối lớn, đặc) mà bỏ sót quai cốc (chi tiết mảnh, xa
            # tâm điểm mồi) — đúng như các mask bị thiếu quai trong ảnh debug.
            # Truyền thêm box giúp SAM2 biết rõ toàn bộ phạm vi vật thể ngay từ
            # đầu, bao gồm cả quai.
            box_arr = (np.array(yolo_box, dtype=np.float32) if yolo_box is not None else None)
            sam2_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=c_prompt_idx,
                obj_id=1,
                points=np.array(pts, dtype=np.float32),
                labels=np.array(labels, dtype=np.int32),
                box=box_arr,
            )
            for frame_idx, obj_ids, mask_logits in sam2_predictor.propagate_in_video(
                    inference_state, start_frame_idx=c_prompt_idx, reverse=False):
                obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)
                if 1 in obj_ids_list:
                    idx = obj_ids_list.index(1)
                    m = (mask_logits[idx, 0].cpu().numpy() > 0.0).astype(np.uint8)
                    raw_masks[chunk_start + frame_idx] = m
                    
            # Backward propagate
            if c_prompt_idx > 0:
                sam2_predictor.reset_state(inference_state)
                sam2_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=c_prompt_idx,
                    obj_id=1,
                    points=np.array(pts, dtype=np.float32),
                    labels=np.array(labels, dtype=np.int32),
                    box=box_arr,
                )
                for frame_idx, obj_ids, mask_logits in sam2_predictor.propagate_in_video(
                        inference_state, start_frame_idx=c_prompt_idx, reverse=True):
                    obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, 'tolist') else list(obj_ids)
                    if 1 in obj_ids_list and (chunk_start + frame_idx) not in raw_masks:
                        idx = obj_ids_list.index(1)
                        m = (mask_logits[idx, 0].cpu().numpy() > 0.0).astype(np.uint8)
                        raw_masks[chunk_start + frame_idx] = m
                        
            # Save the last mask of this chunk for the next chunk
            last_chunk_mask = raw_masks.get(chunk_end - 1, None)
            
            # Free chunk resources
            del inference_state
            torch.cuda.empty_cache()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    masks = []
    for i in range(n):
        raw = raw_masks.get(i)
        if raw is None:
            masks.append(None)
            continue
        m = cv2.resize(raw.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        m = (m > 0.5).astype(np.uint8)
        masks.append(m if np.count_nonzero(m) >= min_px else None)

    # ── Temporal quality gate ────────────────────────────────────────────────
    # SAM2 video propagation đôi khi "nhảy đối tượng" (identity switch) giữa
    # chừng — bám theo một vệt sáng/bóng đổ/vật khác thay vì object gốc — sinh
    # ra mask có hình dạng vô lý (lưỡi liềm, vòng khuyết) hoàn toàn khác vật thể
    # thật, dù diện tích không nhỏ nên vẫn qua được ngưỡng min_px phía trên.
    # Loại mask này KHÔNG bị lọc bởi min_px, và nếu lọt xuống dưới sẽ sinh ra
    # point cloud rác (chính là nguyên nhân part2_raw chỉ có ~39 điểm/frame,
    # thưa thớt và không giống vật thể) làm hỏng luôn ICP/relocalize phía sau
    # (toàn bộ cảnh báo "0 correspondence" trong log).
    #
    # Heuristic: so diện tích & tâm mask với TRUNG VỊ của một cửa sổ các frame
    # lân cận (median chịu được vài frame hỏng xen kẽ, không bị kéo lệch như
    # mean). Một mask lệch quá xa (diện tích nhảy > 2.5x hoặc < 0.35x, HOẶC tâm
    # dịch quá nửa cạnh ảnh) so với trung vị lân cận bị coi là khả nghi và loại
    # bỏ (set về None) — pipeline phía sau xử lý None y hệt như "SAM2 fail".
    def _mask_stats(m):
        if m is None:
            return None
        area = float(np.count_nonzero(m))
        if area == 0:
            return None
        ys, xs = np.where(m > 0)
        return area, float(xs.mean()), float(ys.mean())

    stats = [_mask_stats(m) for m in masks]
    window = 5  # số frame lân cận mỗi bên dùng để tính trung vị tham chiếu
    diag = math.hypot(W, H)
    rejected = 0
    filtered_masks = list(masks)
    for i in range(n):
        if stats[i] is None:
            continue
        neighbor_stats = [stats[j] for j in range(max(0, i - window), min(n, i + window + 1))
                          if j != i and stats[j] is not None]
        if len(neighbor_stats) < 3:
            continue  # không đủ láng giềng tốt để so sánh, giữ nguyên (thà sót còn hơn xoá nhầm)
        areas = np.array([s[0] for s in neighbor_stats])
        cxs = np.array([s[1] for s in neighbor_stats])
        cys = np.array([s[2] for s in neighbor_stats])
        med_area = float(np.median(areas))
        med_cx, med_cy = float(np.median(cxs)), float(np.median(cys))
        area, cx, cy = stats[i]
        area_ratio = area / max(med_area, 1.0)
        centroid_jump = math.hypot(cx - med_cx, cy - med_cy) / diag
        if area_ratio > 2.5 or area_ratio < 0.35 or centroid_jump > 0.35:
            filtered_masks[i] = None
            rejected += 1
            print(f"      [Mask QC] frame {i:03d}: loại bỏ mask khả nghi "
                  f"(area_ratio={area_ratio:.2f}, centroid_jump={centroid_jump:.2f}) "
                  f"— nghi SAM2 bị nhảy đối tượng.")
    masks = filtered_masks
    if rejected:
        print(f"   SAM2 [Mask QC]: loại {rejected}/{n} mask khả nghi (identity switch/nhiễu môi trường).")

    valid = sum(m is not None for m in masks)
    print(f"   SAM2: {valid}/{n} frames have valid masks")
    return masks