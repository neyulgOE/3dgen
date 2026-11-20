# 🗺️ TMVisionary 3D Pipeline — Master Roadmap & Architecture

Tài liệu này mô tả kiến trúc hiện tại của hệ thống (sau khi hoàn tất Giai đoạn 1 và 2) và lộ trình phát triển (Plan) chi tiết cho các Giai đoạn tiếp theo.
Dự án áp dụng tư tưởng thiết kế "Blackbox SDK" (được truyền cảm hứng từ LIPScan-3D-SDK và kiến trúc Intel RealSense), bóc tách triệt để luồng thu nhận dữ liệu, xử lý hình học và thuật toán ghép nối.

---

## 🏗️ 1. Kiến trúc Hệ thống Hiện tại (Chuẩn SDK)

Hệ thống được chia làm 4 tầng (Layers) độc lập, giao tiếp với nhau qua các interface chuẩn:

```text
[1. CAMERA LAYER - Thu thập dữ liệu]
  ├── CameraWorker (TMVisionary)
  │    └── Nhận dữ liệu TCP từ Robot 
  │    └── Đóng gói thành RGBDFrame (Depth thô + Color)
  │    └── emit rgbdFrameReady(RGBDFrame)
  │
  ├── (Tương lai) RealSenseWorker
  │    └── Lấy frame từ cổng USB (Librealsense2)
  │    └── emit rgbdFrameReady(RGBDFrame)

[2. PROJECTION LAYER - Xử lý không gian] (AppController điều phối)
  ├── Nhận RGBDFrame
  ├── DepthProjector (CloudBuilder cũ)
  │    └── Tính toán công thức: (u, v, Z) + Intrinsics ➔ (X, Y, Z)
  │    └── Trả ra PointCloud (Tọa độ Camera)
  ├── CoordinateTransformer
  │    └── Tính toán Pose Inverse (T_inv) từ Waypoint
  └── Đóng gói thành GeometryFrame(PointCloud, Pose)
  
[3. FUSION LAYER - "Hộp đen" Thuật toán] (Nằm trong ScanEngine)
  ├── Nhận PointCloud + T_inv
  ├── IFusionAlgorithm (Interface)
  │    ├── CloudAccumulator (VoxelMap) - Thuật toán hiện tại (Đã triển khai)
  │    ├── Open3D_TSDF_Pipeline - Tương lai (Giai đoạn 4)
  │    └── FPFH_ICP_Pipeline - Tương lai
  ├── Xây dựng Object3D (Chứa mây điểm, lưới mesh, v.v)
  └── emit objectUpdated(Object3D)

[4. PRESENTATION LAYER - Hiển thị & Xuất File] (MainWindow)
  ├── PointCloudWidget (OpenGL Renderer)
  ├── MeshExporter (PLY, OBJ)
  └── NerfDatasetExporter (Xuất ảnh RGB + transforms.json)
```

---

## 🚀 2. Lộ trình Phát triển (The Master Plan)

Chúng ta đã hoàn thành **Giai đoạn 1** (Thiết lập `IFusionAlgorithm` và `ScanEngine` Blackbox) và **Giai đoạn 2** (Tái cấu trúc Camera Layer với `RGBDFrame` và `DepthProjector`). Dưới đây là kế hoạch chi tiết cho phần còn lại của dự án.

### 🟡 Giai đoạn 3: Tích hợp Intel RealSense D400 Series (Trọng tâm tiếp theo)
**Mục tiêu:** Mở rộng hệ thống để hỗ trợ camera RealSense cắm trực tiếp qua USB, tận dụng khả năng quét realtime 60FPS và chất lượng chiều sâu vượt trội.
- **Phát triển `RealSenseWorker`:** Tạo một class kế thừa từ `ICameraSource`. Bọc thư viện `librealsense2` (sử dụng `rs2::pipeline`, `rs2::config`).
- **Xử lý luồng Quét (Chế độ Cầm tay - Handheld):** Do camera RealSense sẽ được cầm tay thay vì gắn trên robot, hệ thống không còn nhận được tọa độ tuyệt đối (Pose) từ bộ điều khiển robot nữa. Do đó, tầng Projection/Fusion sẽ cần tích hợp thêm thuật toán **Visual Odometry / SLAM** (ví dụ: RGB-D Odometry của Open3D) để tự động ước lượng vị trí và góc xoay của camera theo thời gian thực.
- **Tiền xử lý (Pre-processing):** Tận dụng sức mạnh CPU/GPU của PC để bật các bộ lọc của RealSense (Spatial Filter, Temporal Filter) trước khi đẩy `RGBDFrame` ra ngoài.

### 🟡 Giai đoạn 4: Nâng cấp Thuật toán Ghép nối (Advanced Fusion / ICP)
**Mục tiêu:** Thay thế thuật toán cộng dồn (`VoxelMap`) thô sơ bằng phương pháp nối điểm chính xác (ICP hoặc TSDF Volume) bằng C++ Open3D.
- **Tích hợp Open3D:** Biên dịch và link thư viện `Open3D C++` vào dự án Qt.
- **Tạo Plugin mới:** Viết class `Open3DFusion` kế thừa từ `IFusionAlgorithm`.
- **Luồng xử lý ICP tiêu chuẩn:**
  1. Downsample đám mây điểm.
  2. Tính toán Normals (Pháp tuyến) và đặc trưng FPFH.
  3. Chạy Global Registration (RANSAC).
  4. Chạy Local Registration (Point-to-Plane ICP).
- Nhờ kiến trúc Giai đoạn 1, việc thay đổi thuật toán chỉ tốn đúng 1 dòng code trong `ScanEngine`: đổi `_algorithm = new CloudAccumulator()` thành `_algorithm = new Open3DFusion()`.

### 🟡 Giai đoạn 5: Tái tạo Bề mặt (Surface Reconstruction) & Xuất Mesh
**Mục tiêu:** Biến đám mây điểm (Point Cloud) thành mô hình 3D khối đặc (Mesh - OBJ/STL) để phục vụ In 3D hoặc Inspection.
- Nâng cấp cấu trúc dữ liệu `Object3D` để chứa mảng các tam giác (Faces/Triangles).
- Chạy thuật toán **Poisson Surface Reconstruction** hoặc **BPA (Ball-Pivoting Algorithm)**.
- Cập nhật module `Exporter` để hỗ trợ ghi file định dạng `.obj` hoặc `.stl`.

### 🟡 Giai đoạn 6: Tái tạo 3D Nâng cao (Global Registration & Deep Learning)
**Mục tiêu:** Giải quyết dứt điểm bài toán Handheld SLAM khi không có mỏ neo tọa độ, đồng thời mở ra hướng đi tạo 3D quang học siêu thực tế (Photorealistic).
- **Trích xuất đặc trưng (Feature Matching):** Dùng Open3D FPFH (Fast Point Feature Histograms) kết hợp RANSAC để tự động nhận diện góc cạnh và tìm ma trận ghép nối (Global Registration) thay cho ICP truyền thống. Đây là giải pháp tiêu chuẩn cho thiết bị cầm tay.
- **Tùy chọn nâng cao - Deep Learning SLAM:** Thay thế FPFH bằng các mạng Neural Networks (GeoTransformer, DCP) để trích xuất đặc trưng thông minh hơn, chịu được vệt nhòe hoặc vật thể chuyển động.
- **Tùy chọn nâng cao - 3D Gaussian Splatting / NeRF:** 
  - Xây dựng `NerfDatasetExporter`: Thu thập mảng ảnh RGB và ma trận Pose (`T_cam_i_inv`) xuất ra định dạng `transforms.json`.
  - Hỗ trợ người dùng kéo thả thư mục kết quả vào Instant-NGP hoặc Luma AI để render ra môi trường 3D quang học sống động (Digital Twin).

---

## 🛠️ 3. Danh sách Công việc (Sprint Backlog) cho Giai đoạn 3

Để bắt tay vào làm Giai đoạn 3, chúng ta sẽ cần làm các task sau:
1. [ ] **Thiết lập librealsense2:** Tải SDK, cấu hình file `TMVisionaryViewer.pro` để link các file `.lib` và include `.h` của RealSense vào dự án.
2. [ ] **Giao diện chuyển đổi Camera:** Thêm Combobox vào `SettingsDialog` (tab Camera) cho phép User chọn nguồn dữ liệu: `TMVisionary (TCP)` hoặc `Intel RealSense (USB)`.
3. [ ] **Khởi tạo RealSenseWorker:** Viết bộ khung (Skeleton) cho `RealSenseWorker`, implement các hàm `init()`, `connectHost()` (có thể bỏ qua IP/Port), và `fetchCount()` để lấy frame từ USB.
