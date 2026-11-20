# Kế hoạch Phát triển Hệ thống 3D Reconstruction với Sparse RGB-D

Dựa trên phản hồi, dưới đây là bản thiết kế kiến trúc đã được cập nhật: **Tích hợp trực tiếp vào C++**, **sử dụng VIO thuần túy (quét tự do)**, và **tích hợp AI Segmentation bằng ONNX**.

## Giải quyết bài toán: "Tích hợp thẳng vào App (C++) có bị nặng không?"
**Trả lời:** Tích hợp trực tiếp vào C++ hoàn toàn khả thi và thực tế sẽ chạy **nhanh hơn** Python, nhưng mã nguồn sẽ cồng kềnh hơn do phải liên kết nhiều thư viện. Để App (UI) không bị đơ/lag (nặng máy) khi chạy, chúng ta bắt buộc áp dụng kiến trúc **Đa luồng (Multi-threading)**:
- **Luồng giao diện (Main UI Thread):** Chỉ hiển thị luồng stream từ camera, thanh tiến trình (progress bar), và render Point Cloud cuối cùng.
- **Luồng Background Worker (QThread / std::thread):** Gánh toàn bộ các tác vụ nặng: SLAM/VIO định vị, AI Segmentation, và tính toán Voxel.

## Đánh giá SAM vs YOLOv8-Seg (Cho bài toán Xóa Nền)
Việc chọn AI phụ thuộc vào bài toán đặc thù của dự án. Vì hệ thống tích hợp bằng C++, việc chạy AI sẽ thông qua bộ máy **ONNXRuntime** hoặc **TensorRT**.

| Tiêu chí | SAM (Segment Anything - Meta) | YOLOv8-Seg (Ultralytics) |
| :--- | :--- | :--- |
| **Bản chất** | Zero-shot Segmentation (Cắt được mọi thứ trên đời chỉ bằng cách "chấm" 1 điểm vào vật) | Đã được train sẵn trên 80-1000 class (COCO/ImageNet). Chỉ cắt được đồ vật nó đã "học". |
| **Độ chính xác viền** | Cực kỳ cao, sát từng pixel. | Khá tốt, nhưng viền không sắc nét bằng SAM. |
| **Tốc độ (Inference)** | Rất nặng và chậm (vài giây/ảnh trên CPU). Khó đưa bản gốc vào C++. | Tốc độ bàn thờ (Real-time 60fps+). Đưa vào C++ qua ONNX cực kỳ dễ và mượt. |
| **Kết luận cho dự án** | Phù hợp nếu dự án quét các vật thể máy móc/cơ khí kỳ dị chưa từng có trong từ điển. | Phù hợp nếu quét người, đồ đạc phổ thông, hoặc bạn có team để tự train lại YOLO cho vật thể riêng. |

> [!TIP]
> **Khuyến nghị (Phương án dung hòa): Dùng FastSAM hoặc MobileSAM.**
> Đây là các biến thể xây dựng trên kiến trúc của YOLOv8 nhưng được train bằng dữ liệu của SAM. Chúng có ưu điểm: **Cắt được mọi vật (zero-shot) giống SAM + Tốc độ cực nhanh và dễ nạp vào C++ (ONNX) giống YOLOv8.** Chúng ta sẽ chốt sử dụng **FastSAM (ONNX)** cho Phase 3.

---

## Các Giai đoạn Triển khai (Phasing)

### Phase 1: Thu thập Dữ liệu & Quét Tự do (VIO Tracking)
**Mục tiêu:** Quét cầm tay tự do, không cần điểm neo (ArUco).
- **Hardware:** Intel RealSense D435if.
- **Tracking:** Tích hợp thư viện **RTAB-Map C++ API** hoặc thuật toán VIO (từ `librealsense2` kết hợp bộ lọc Kalman) để lấy tọa độ liên tục từ Depth + IMU.
- **Luồng chạy:**
  1. Người dùng bấm "Start Scan" trên UI.
  2. Background Thread thu thập mảng liên tục các cấu trúc: `struct FrameData { cv::Mat rgb; cv::Mat depth; Eigen::Matrix4f pose; }`.

### Phase 2: Rút trích 24 Keyframes (Sparse Extraction)
**Mục tiêu:** Giảm tải tính toán từ hàng nghìn frame xuống 24 frame.
- Thuật toán K-Means Clustering hoặc Grid Sampling trong không gian tọa độ cầu 3D.
- Từ quỹ đạo VIO, lấy ra đúng 24 vị trí (poses) dàn đều xung quanh điểm trung tâm của vật thể (có cả các góc ngước từ dưới lên và chúc từ trên xuống).
- Truy xuất 24 bộ `FrameData` có độ sắc nét tốt nhất (dựa trên thuật toán tính phương sai Laplacian của ảnh) tại các vị trí đó.

### Phase 3: AI Tách nền trong C++ (FastSAM / ONNXRuntime)
**Mục tiêu:** Loại bỏ hoàn toàn nhiễu nền (tay, bàn, xưởng) ra khỏi 24 ảnh.
- Nạp model `FastSAM.onnx` vào môi trường C++ thông qua **ONNXRuntime**.
- **Luồng xử lý tự động hóa:**
  1. Lấy vị trí trung tâm bức ảnh (W/2, H/2) làm điểm mồi (prompt point).
  2. Gửi ảnh RGB và điểm mồi vào model FastSAM thông qua C++ Tensor.
  3. Model trả về một Mask đen trắng cắt chuẩn xác vật thể ở giữa.
- **Output:** 24 `cv::Mat` mask.

### Phase 4: Voxel Reconstruction Core (Open3D C++)
**Mục tiêu:** Dựng hình học từ 24 view đã được tách nền. (Áp dụng thư viện Open3D C++)
1. **Khởi tạo:** Tạo `open3d::geometry::TSDFVolume` (Voxel Grid).
2. **Voxel Clipping (Khử nhiễu):** Trước khi đưa vào TSDF, các tia chiếu (Raycast) từ camera tới điểm Depth mà không nằm trong vùng Mask trắng (ở Phase 3) sẽ bị gán Depth = 0 (bỏ qua). Điều này đảm bảo nền nhà không bao giờ bị dựng lên.
3. **TSDF Integration:** Gộp 24 ảnh Depth và Pose vào TSDF Volume.
4. **Voxel Filling (Trám lỗ rỗng):** Viết thêm một hàm C++ duyệt qua các block Voxel. Nếu Voxel nằm trong Silhouette (bóng mask 2D) nhưng TSDF value rỗng (do D435if mù màu đen), dùng nội suy khoảng cách từ các Voxel xung quanh để trám lại.

### Phase 5: Xuất Mesh & Tối ưu Vân bề mặt (Texture Mapping)
- Sử dụng hàm `ExtractTriangleMesh()` của Open3D C++ (bản chất là Marching Cubes) để rút trích mô hình 3D thô từ TSDF Volume.
- Chạy Poisson Surface Reconstruction hoặc Laplacian Smoothing để làm mượt mesh.
- Tính toán và dán Texture từ 24 ảnh RGB ban đầu lên mặt Mesh dựa trên góc chiếu vuông góc nhất để tạo ra file `.ply` hoặc `.obj` hoàn thiện, đẩy ngược lại UI Thread để hiển thị.

## Kế hoạch Triển khai (Action Plan)
1. Bắt đầu với Phase 1: Tạo nhánh mới, thiết lập RTAB-Map hoặc bộ lọc VIO trong C++ với thư viện `librealsense2` để ghi nhận quỹ đạo (Trajectory).
2. Triển khai Phase 2: Viết script C++ chia không gian quỹ đạo để rút 24 khung hình.
3. Setup ONNXRuntime trong dự án C++ và chạy thử FastSAM (Phase 3).
