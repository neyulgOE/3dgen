#ifndef IFUSIONALGORITHM_H
#define IFUSIONALGORITHM_H

#include <QObject>
#include <QMatrix4x4>
#include <vector>
#include "types.h"

// ─────────────────────────────────────────────────────────────────────────────
// IFusionAlgorithm — Interface chuẩn cho các thuật toán xử lý và ghép 3D.
//
// Bất kỳ thuật toán nào (VoxelMap, TSDF, ICP...) muốn cắm vào ScanEngine
// đều phải kế thừa interface này.
// ─────────────────────────────────────────────────────────────────────────────

class IFusionAlgorithm : public QObject
{
    Q_OBJECT
public:
    explicit IFusionAlgorithm(QObject* parent = nullptr) : QObject(parent) {}
    virtual ~IFusionAlgorithm() = default;

    // ── Interface bắt buộc ──
    virtual void setEnabled(bool on) = 0;
    virtual void addCloud(const PointCloud &cloud, const QMatrix4x4 &pose) = 0;
    virtual void clear() = 0;
    virtual void loadRawCloud(const std::vector<float> &xyz, const std::vector<uint8_t> &rgb, int count) = 0;

    // ── Cấu hình (Có thể throw/ignore nếu thuật toán không hỗ trợ) ──
    virtual void setVoxelSize(float mm) = 0;
    virtual void setScanMask(bool on, float cx, float cy, float cz, float r) = 0;
    virtual void setIcpEnabled(bool on) = 0;

signals:
    void statsUpdated(int voxelCount, int pointCount);
    void deltaReady(PointCloud delta);
    void fullRebuildReady(PointCloud full);
};

#endif // IFUSIONALGORITHM_H
