import sys
import os
import cv2
import glob
from ultralytics import YOLO, FastSAM, YOLOWorld
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 fastsam_batch.py <directory> <timestamp> [prompt]")
        sys.exit(1)
        
    img_dir = sys.argv[1]
    timestamp = sys.argv[2]
    prompt = sys.argv[3].lower() if len(sys.argv) > 3 else "cup"
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FastSAM-s.pt')
    fsam_model = FastSAM(model_path)
    
    # Load YOLO-World for Open-Vocabulary detection
    try:
        yolo_model = YOLOWorld('yolov8s-world.pt')
        yolo_model.set_classes([prompt])
        use_yolo_world = True
        print(f"Using YOLO-World to detect '{prompt}'...")
    except Exception as e:
        print(f"Failed to load YOLO-World: {e}")
        use_yolo_world = False
    
    img_paths = sorted(glob.glob(os.path.join(img_dir, f"keyframe_{timestamp}_*.png")))
    print(f"Found {len(img_paths)} keyframes to process.")
    
    prev_box = None
    
    for path in img_paths:
        img = cv2.imread(path)
        if img is None: continue
        
        masked_img = np.zeros_like(img)
        best_mask = None
        
        if use_yolo_world:
            # 1. Detect Box using YOLO-World
            res_yolo = yolo_model(img, conf=0.1, verbose=False)
            boxes = res_yolo[0].boxes
            
            best_box = None
            min_dist = float('inf')
            max_conf = -1.0
            
            if boxes is not None:
                for i in range(len(boxes)):
                    # YOLO-World only has one class now (index 0) because we set_classes([prompt])
                    cls_id = int(boxes.cls[i].item())
                    if cls_id == 0:
                        b = boxes.xyxy[i].cpu().numpy().flatten()
                        
                        if prev_box is not None:
                            # Lock-on mode: choose the box whose center is closest to prev_box
                            c_prev_x = (prev_box[0] + prev_box[2]) / 2.0
                            c_prev_y = (prev_box[1] + prev_box[3]) / 2.0
                            c_curr_x = (b[0] + b[2]) / 2.0
                            c_curr_y = (b[1] + b[3]) / 2.0
                            dist = (c_prev_x - c_curr_x)**2 + (c_prev_y - c_curr_y)**2
                            
                            # Max allowed jump distance squared (e.g. 200 pixels)
                            if dist < min_dist and dist < 40000:
                                min_dist = dist
                                best_box = [int(x) for x in b]
                        else:
                            # First frame or lost: choose the box with highest confidence
                            c_curr_x = (b[0] + b[2]) / 2.0
                            c_curr_y = (b[1] + b[3]) / 2.0
                            
                            conf = boxes.conf[i].item()
                            if conf > max_conf:
                                max_conf = conf
                                best_box = [int(x) for x in b]
            
            # Temporal fallback: If YOLOWorld misses but we saw it in previous frame, reuse the box
            if best_box is None and prev_box is not None:
                # Do not expand the box significantly, otherwise it will eventually capture the wall
                best_box = prev_box
            
            # 2. Refine Mask using FastSAM
            if best_box is not None:
                res_fsam = fsam_model(img, bboxes=[best_box], verbose=False)
                if len(res_fsam) > 0 and res_fsam[0].masks is not None:
                    best_mask = res_fsam[0].masks.data.cpu().numpy()[0]
                    prev_box = best_box # Save for next frame
        else:
            # Fallback to Center-Point Prompt (picking the smallest object covering the center)
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            
            res_fsam = fsam_model(img, conf=0.25, iou=0.45, verbose=False)
            if len(res_fsam) > 0 and res_fsam[0].masks is not None:
                masks = res_fsam[0].masks.data.cpu().numpy()
                boxes = res_fsam[0].boxes.xyxy.cpu().numpy()
                
                best_idx = -1
                min_area = float('inf')
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # Check if the center of the image is inside this bounding box
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        area = (x2 - x1) * (y2 - y1)
                        # Pick the smallest object covering the center to avoid picking the background/table
                        if area < min_area:
                            min_area = area
                            best_idx = i
                
                if best_idx != -1:
                    best_mask = masks[best_idx]
        
        # Apply mask
        if best_mask is not None:
            best_mask = cv2.resize(best_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            kernel = np.ones((5, 5), np.uint8)
            best_mask = cv2.erode(best_mask, kernel, iterations=1)
            
            masked_img[best_mask > 0] = img[best_mask > 0]
            
        cv2.imwrite(path, masked_img)
    
    print("Batch processing complete.")

if __name__ == '__main__':
    main()
