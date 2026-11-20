#ifndef SCANENGINE_H
#define SCANENGINE_H

#include <QObject>
#include <QMatrix4x4>
#include "object3d.h"
#include "ifusionalgorithm.h"

// ─────────────────────────────────────────────────────────────────────────────
// ScanEngine — BlackBox xử lý thuật toán SLAM / 3D Reconstruction.
//
// Nhiệm vụ:
// - Nhận các Frame (PointCloud + Pose) từ Camera/Controller.
// - Chạy thuật toán hợp nhất (VoxelMap, ICP, Global Registration...).
// - Trả ra một Object3D duy nhất đã được xử lý hoàn chỉnh.
// ─────────────────────────────────────────────────────────────────────────────

class ScanEngine : public QObject
{
    Q_OBJECT
public:
    explicit ScanEngine(QObject *parent = nullptr);

    // ── Interface chính ──
    void addFrame(const PointCloud &cloud, const QMatrix4x4 &pose = QMatrix4x4());
    void clear();

    Object3D getObject3D() const;

    // ── Cấu hình thuật toán ──
    void setEnabled(bool on);
    void setVoxelSize(float mm);
    void setScanMask(bool on, float cx, float cy, float cz, float r);
    void setIcpEnabled(bool on);
    
    // ── Thao tác dữ liệu ──
    void loadRawCloud(const std::vector<float> &xyz, const std::vector<uint8_t> &rgb, int count);

signals:
    // Bắn ra khi có object mới được cập nhật (Ví dụ: sau khi chạy ICP hoặc Full Rebuild)
    void objectUpdated(const Object3D &object);
    
    // Bắn ra khi có một lượng điểm mới (Delta) được thêm vào nhanh (Fast rendering)
    void partialUpdateReady(const PointCloud &delta);
    
    // Thông báo thống kê
    void statsUpdated(int voxelCount, int frameCount);

private slots:
    void onAccumulatorDeltaReady(const PointCloud &delta);
    void onAccumulatorFullRebuild(const PointCloud &full);
    void onAccumulatorStatsUpdated(int voxelCount, int pointCount);

private:
    // Tạm thời encapsulate CloudAccumulator bên trong thông qua IFusionAlgorithm.
    // Tương lai có thể thay thế bằng C++ Open3D Pipeline hoặc FPFH.
    IFusionAlgorithm* _algorithm;
    Object3D _currentObject;
};

#endif // SCANENGINE_H
