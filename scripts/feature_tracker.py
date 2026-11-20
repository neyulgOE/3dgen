import cv2
import numpy as np
import json
import os
import argparse

from pose_recovery import evaluate_registration

class FeatureTracker:
    def __init__(self, fx, fy, cx, cy, n_features=2000, min_inliers=15, inlier_thr_m=0.015):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.min_inliers = min_inliers
        self.inlier_thr_m = inlier_thr_m
        
        # Initialize ORB
        self.orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8, edgeThreshold=31)
        # BFMatcher with NORM_HAMMING
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def extract_features(self, img_bgr, mask=None):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if mask is not None:
            mask = (mask > 0).astype(np.uint8) * 255
        kp, des = self.orb.detectAndCompute(gray, mask)
        return kp, des

    def lift_to_3d(self, pts_2d, depth_img, min_depth_mm=50):
        # pts_2d: (N, 2)
        N = len(pts_2d)
        pts_3d = np.full((N, 3), np.nan, dtype=np.float32)
        valid = np.zeros(N, dtype=bool)
        
        for i in range(N):
            x, y = pts_2d[i]
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < depth_img.shape[0] and 0 <= xi < depth_img.shape[1]:
                z_mm = depth_img[yi, xi]
                if z_mm > min_depth_mm:
                    z = z_mm / 1000.0
                    X = (x - self.cx) * z / self.fx
                    Y = (y - self.cy) * z / self.fy
                    pts_3d[i] = [X, Y, z]
                    valid[i] = True
        return pts_3d, valid

    def match_frames(self, imgA_bgr, imgB_bgr, depthA, depthB, maskA=None, maskB=None):
        # 1. Extract features
        kpA, desA = self.extract_features(imgA_bgr, maskA)
        kpB, desB = self.extract_features(imgB_bgr, maskB)
        
        if desA is None or desB is None or len(kpA) < self.min_inliers or len(kpB) < self.min_inliers:
            return {"status": "REJECTED", "reason": "Not enough features extracted"}
            
        # 2. Match features
        matches = self.matcher.knnMatch(desA, desB, k=2)
        
        # 3. Ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) != 2: continue
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        num_matches = len(good_matches)
        if num_matches < self.min_inliers:
            return {"status": "REJECTED", "reason": f"Only {num_matches} good matches (ratio test)"}
            
        ptsA = np.array([kpA[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        ptsB = np.array([kpB[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        
        # 4. Lift to 3D
        ptsA_3d, validA = self.lift_to_3d(ptsA, depthA)
        ptsB_3d, validB = self.lift_to_3d(ptsB, depthB)
        
        valid_both = validA & validB
        if valid_both.sum() < self.min_inliers:
            return {"status": "REJECTED", "reason": f"Only {valid_both.sum()} valid 3D matches"}
            
        p_anc_3d = ptsA_3d[valid_both]
        p_tgt_3d = ptsB_3d[valid_both]
        p_tgt_2d = ptsB[valid_both]
        
        # 5. RANSAC + Verification using evaluate_registration
        res = evaluate_registration(
            p_anc_3d, p_tgt_3d, p_tgt_2d,
            n_valid=len(ptsA), # use total queries (2D matches) as base for inlier ratio
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            min_inliers=self.min_inliers,
            inlier_thr_m=self.inlier_thr_m,
            ransac_iters=256
        )
        
        res["num_matches"] = num_matches
        res["valid_3d_matches"] = int(valid_both.sum())
        
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("--frameA", type=int, default=0)
    parser.add_argument("--frameB", type=int, default=5)
    args = parser.parse_args()

    ds = args.dataset_dir
    # Load intrinsics
    intr_path = os.path.join(ds, "intrinsics.json")
    if os.path.exists(intr_path):
        with open(intr_path) as f:
            d = json.load(f)
        fx, fy = d.get("fx", 615.0), d.get("fy", 615.0)
        cx, cy = d.get("cx", 320.0), d.get("cy", 240.0)
    else:
        fx = fy = 615.0; cx = 320.0; cy = 240.0

    tracker = FeatureTracker(fx, fy, cx, cy)

    # Resolve actual filenames
    fnA = f"{args.frameA:06d}.png"
    fnB = f"{args.frameB:06d}.png"
    fnames_path = os.path.join(ds, "frame_filenames.json")
    if os.path.exists(fnames_path):
        with open(fnames_path) as f:
            names = json.load(f).get("filenames", [])
            if args.frameA < len(names): fnA = names[args.frameA]
            if args.frameB < len(names): fnB = names[args.frameB]

    imgA = cv2.imread(os.path.join(ds, "rgb", fnA))
    imgB = cv2.imread(os.path.join(ds, "rgb", fnB))
    depthA = cv2.imread(os.path.join(ds, "depth_raw", fnA), cv2.IMREAD_UNCHANGED)
    depthB = cv2.imread(os.path.join(ds, "depth_raw", fnB), cv2.IMREAD_UNCHANGED)

    maskA_path = os.path.join(ds, "reconstruction", f"{args.frameA:06d}_mask.png")
    maskB_path = os.path.join(ds, "reconstruction", f"{args.frameB:06d}_mask.png")
    maskA = cv2.imread(maskA_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(maskA_path) else None
    maskB = cv2.imread(maskB_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(maskB_path) else None

    if imgA is None or imgB is None:
        print("Images not found!")
        return

    print(f"Matching frame {args.frameA} and {args.frameB}...")
    res = tracker.match_frames(imgA, imgB, depthA, depthB, maskA, maskB)
    
    print(f"Status: {res['status']}")
    if res['status'] != "REJECTED":
        print(f"Matches: {res['num_matches']}  Valid 3D: {res['valid_3d_matches']}")
        print(f"Inliers: {res['inliers']} ({res.get('inlier_ratio',0)*100:.1f}%)")
        print(f"Reproj Error (Median): {res.get('reprojection_error_median_px', -1)} px")
        print(f"Cycle Transl: {res.get('cycle_translation_mm', -1):.1f} mm")
        print(f"Translation: {res.get('translation_mm', -1):.1f} mm")
    else:
        print(f"Reason: {res.get('reason')}")

if __name__ == "__main__":
    main()
