import cv2
import numpy as np


class DepthPreFilter:
    """
    Bộ lọc tiền xử lý Depth Map cho Intel RealSense D435f:
    1. Depth Range Gate (Lower / Upper Limit)
    2. Edge Artifact & False Point Removal (Khử flying-pixels ở mép vật thể)
    3. Noise Removal (Khử nhiễu hạt/gai bằng Median trên miền hợp lệ)
    4. Gap Filling (Lấp lỗ thủng depth=0 bên trong mask vật thể do bề mặt đen/bóng)
    5. Surface Smoothing (Lọc Bilateral bảo toàn góc cạnh)
    6. Normal Outlier Removal (Loại bỏ điểm nhìn sượt góc lớn - Grazing angle)
    7. Outlier Island Removal (Loại bỏ cụm đốm cô lập)
    """

    def __init__(
        self,
        surface_smoothing="strong",      # 'off' | 'weak' | 'normal' | 'strong'
        gap_filling="normal",            # 'off' | 'weak' | 'normal' | 'strong'
        noise_removal="strong",          # 'off' | 'normal' | 'strong'
        outlier_removal_level=2,         # 0 | 1 | 2 | 3
        normal_outlier_removal="normal", # 'off' | 'weak' | 'normal' | 'strong'
        edge_artifact_removal=True,
        depth_min_mm=80.0,
        depth_max_mm=1200.0,
    ):
        self.surface_smoothing = surface_smoothing.lower() if isinstance(surface_smoothing, str) else "normal"
        self.gap_filling = gap_filling.lower() if isinstance(gap_filling, str) else "normal"
        self.noise_removal = noise_removal.lower() if isinstance(noise_removal, str) else "normal"
        self.outlier_removal_level = int(outlier_removal_level)
        self.normal_outlier_removal = normal_outlier_removal.lower() if isinstance(normal_outlier_removal, str) else "normal"
        self.edge_artifact_removal = bool(edge_artifact_removal)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)

    def process_frame(self, depth_raw, mask=None, fx=386.65, fy=386.65, cx=326.09, cy=240.78):
        """
        Xử lý 1 frame depth (đơn vị mm, uint16 hoặc float32).
        Nếu truyền `mask` (từ SAM2), tính năng Gap Filling sẽ lấp kín các lỗ thủng
        cục bộ (depth=0 do vùng màu đen hấp thụ IR) bên trong vật thể.
        """
        orig_dtype = depth_raw.dtype
        d = depth_raw.astype(np.float32).copy()

        # ── 1. Depth Range Gate (Depth Lower / Upper Limit) ─────────────────
        valid = (d >= self.depth_min_mm) & (d <= self.depth_max_mm)
        d[~valid] = 0.0

        # ── 2. Edge Artifact & False Point Removal (Khử flying pixels) ──────
        if self.edge_artifact_removal:
            k3 = np.ones((3, 3), np.uint8)
            d_for_min = np.where(d > 0, d, 1e6).astype(np.float32)
            local_min = cv2.erode(d_for_min, k3)
            local_max = cv2.dilate(d, k3)
            local_span = local_max - local_min
            # Ngưỡng nhảy bậc tỉ lệ theo khoảng cách Z (3.5% độ sâu hoặc tối thiểu 12mm)
            step_thr = np.maximum(12.0, 0.035 * d)
            flying_edge = (d > 0) & (local_span > step_thr)
            d[flying_edge] = 0.0

        # ── 3. Noise Removal (Khử nhiễu hạt/gai trước khi nội suy) ──────────
        if self.noise_removal in ("normal", "strong"):
            ksize = 5 if self.noise_removal == "strong" else 3
            med = cv2.medianBlur(d, ksize)
            # Chỉ thay thế bằng median ở các điểm hợp lệ và không làm nhảy quá 15mm
            use_med = (d > 0) & (med > 0) & (np.abs(d - med) < 15.0)
            d[use_med] = med[use_med]

        # ── 4. Gap Filling (Lấp lỗ thủng vùng đen bên trong mặt hộp) ─────────
        if self.gap_filling != "off":
            radius_map = {"weak": 3, "normal": 7, "strong": 13}
            fill_rad = radius_map.get(self.gap_filling, 7)
            d = self._fill_internal_gaps(d, mask, max_radius=fill_rad)

        # ── 5. Surface Smoothing (Bilateral Filter bảo toàn góc cạnh) ───────
        if self.surface_smoothing != "off":
            d = self._bilateral_smooth(d, level=self.surface_smoothing)

        # ── 6. Normal Outlier Removal (Loại bỏ điểm nhìn sượt góc lớn) ──────
        if self.normal_outlier_removal != "off":
            # Góc tối đa giữa pháp tuyến bề mặt và tia nhìn camera
            angle_map = {"weak": 80.0, "normal": 73.0, "strong": 65.0}
            max_grazing_deg = angle_map.get(self.normal_outlier_removal, 73.0)
            d = self._remove_normal_outliers(d, fx, fy, cx, cy, max_grazing_deg)

        # ── 7. Outlier Removal (Loại bỏ các cụm đốm depth cô lập) ───────────
        if self.outlier_removal_level > 0:
            min_island_px = {1: 25, 2: 60, 3: 120}.get(self.outlier_removal_level, 60)
            d = self._remove_small_depth_islands(d, min_island_px)

        if orig_dtype == np.uint16:
            return np.clip(np.round(d), 0, 65535).astype(np.uint16)
        return d

    def _fill_internal_gaps(self, d, mask, max_radius=7):
        """
        Lấp các lỗ khuyết (d == 0) nằm lọt thỏm bên trong bề mặt hợp lệ hoặc trong SAM2 mask,
        nhưng chỉ lấp khi độ sâu biên quanh lỗ đồng nhất (cùng 1 mặt phẳng, không nối qua mép).
        """
        valid = (d > 0).astype(np.uint8)
        k_size = 2 * max_radius + 1
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, k_close)
        holes = (closed > 0) & (valid == 0)
        if mask is not None:
            mask_u8 = (mask > 0).astype(np.uint8)
            mask_eroded = cv2.erode(mask_u8, np.ones((5, 5), np.uint8))
            holes = holes | ((mask_eroded > 0) & (valid == 0))

        if not np.any(holes):
            return d

        # Kiểm tra độ chênh lệch độ sâu quanh lỗ: chỉ lấp nếu biên lỗ phẳng (< 25mm)
        d_for_min = np.where(d > 0, d, 1e6).astype(np.float32)
        b_min = cv2.erode(d_for_min, k_close)
        b_max = cv2.dilate(d, k_close)
        planar_holes = holes & ((b_max - b_min) < 25.0)

        # Dùng phép giãn nở có trọng số Gaussian đa tầng (Multi-pass Normalized Convolution)
        # để nội suy mượt từ viền lỗ vào tâm lỗ lớn mà không bị lem bậc thang
        filled = d.copy()
        for _ in range(3):
            valid_now = (filled > 0).astype(np.float32)
            num = filled * valid_now
            num_blur = cv2.GaussianBlur(num, (k_size, k_size), sigmaX=max_radius * 0.7)
            den_blur = cv2.GaussianBlur(valid_now, (k_size, k_size), sigmaX=max_radius * 0.7)
            update_mask = (filled == 0) & planar_holes & (den_blur > 0.05)
            if not np.any(update_mask):
                break
            filled[update_mask] = num_blur[update_mask] / np.maximum(den_blur[update_mask], 1e-5)

        return filled

    def _bilateral_smooth(self, d, level="strong"):
        """Làm mượt bề mặt bảo toàn góc cạnh bằng Bilateral Filter."""
        params = {
            "weak":   (5, 3.5, 5.0),
            "normal": (7, 5.5, 7.0),
            "strong": (9, 8.0, 9.0),
        }
        diam, sigma_z_mm, sigma_space = params.get(level, (9, 8.0, 9.0))
        valid = d > 0
        if np.count_nonzero(valid) < 100:
            return d

        smoothed = cv2.bilateralFilter(d, d=diam, sigmaColor=sigma_z_mm, sigmaSpace=sigma_space)
        out = d.copy()
        keep = valid & (np.abs(smoothed - d) < sigma_z_mm * 1.5)
        out[keep] = smoothed[keep]
        return out

    def _remove_normal_outliers(self, d, fx, fy, cx, cy, max_grazing_deg=73.0):
        """
        Loại bỏ các pixel có pháp tuyến bề mặt lệch góc quá lớn so với tia nhìn camera
        (Grazing-angle / Normal Outlier Removal).
        """
        valid = d > 0
        if np.count_nonzero(valid) < 100:
            return d

        H, W = d.shape
        u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        Z = d
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        # Đạo hàm theo u, v để tính vector tiếp tuyến
        dX_du = cv2.Sobel(X, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dY_du = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        dZ_du = cv2.Sobel(Z, cv2.CV_32F, 1, 0, ksize=3) / 8.0

        dX_dv = cv2.Sobel(X, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        dY_dv = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        dZ_dv = cv2.Sobel(Z, cv2.CV_32F, 0, 1, ksize=3) / 8.0

        # Tích có hướng n = dP/dv x dP/du
        nx = dY_dv * dZ_du - dZ_dv * dY_du
        ny = dZ_dv * dX_du - dX_dv * dZ_du
        nz = dX_dv * dY_du - dY_dv * dX_du
        norm_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
        nx /= norm_len
        ny /= norm_len
        nz /= norm_len

        # Vector tia nhìn từ vật thể về camera
        p_len = np.sqrt(X * X + Y * Y + Z * Z) + 1e-8
        cos_theta = np.abs((nx * (-X) + ny * (-Y) + nz * (-Z)) / p_len)
        cos_limit = float(np.cos(np.radians(max_grazing_deg)))

        eroded_valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        bad_normal = eroded_valid & (cos_theta < cos_limit)

        out = d.copy()
        out[bad_normal] = 0.0
        return out

    def _remove_small_depth_islands(self, d, min_area_px=60):
        valid_u8 = (d > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(valid_u8, connectivity=8)
        if num_labels <= 1:
            return d
        out = d.copy()
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area_px:
                out[labels == lbl] = 0.0
        return out
