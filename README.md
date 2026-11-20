# TMVisionary 3D Studio (`3dgen`)

Ứng dụng quét và tái tạo mô hình lưới 3D (3D Reconstruction & Mesh Studio) sử dụng camera chiều sâu Intel RealSense (D435/D435f), kết hợp giao diện C++/Qt thời gian thực với pipeline AI & SLAM ngoại tuyến (Python / Open3D Tensor GPU / PyTorch).

## Kiến trúc hệ thống
- **Frontend (C++ / Qt / OpenGL — `src/`):**
  - Thu thập dữ liệu RGB-D + IMU, theo dõi VIO thời gian thực (`vio_tracker.cpp`).
  - Trình xem Point Cloud & Mesh Studio (`meshviewerwidget.cpp`) hỗ trợ Auto Clean, tẩy thủ công, Undo và xuất `.ply` / `.obj`.
- **Backend AI & SLAM (Python — `scripts/`):**
  - `slam_object_builder.py`: Tách nền vật thể bằng YOLOWorld + SAM2, lọc tiền xử lý Depth, tối ưu đồ thị tư thế đa vòng (`pose_graph_optimizer.py`), khép vòng 360°, phục hồi mất dấu bằng CoTracker3 (`sparse_tracker.py`), tích hợp TSDF GPU và hậu xử lý lưới tự thích nghi hình học (`geometry_utils.py`).
  - `slam_engine.py`: Pipeline tối ưu mây điểm toàn cảnh có tích hợp IMU và Global PGO.

## Biên dịch & Chạy
1. Mở `TMVisionaryViewer.pro` bằng Qt Creator (hoặc chạy `qmake && make -j$(nproc)`).
2. Cài đặt phụ thuộc Python: `open3d`, `torch`, `opencv-python`, `ultralytics`, `sam2`.
3. Chạy ứng dụng, ghi dataset trong tab **Scan & 3D Point Cloud**, sau đó dựng lưới trong tab **Mesh Studio & Cleanup**.
