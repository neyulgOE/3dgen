#ifndef CLOUDACCUMULATOR_H
#define CLOUDACCUMULATOR_H

#include "types.h"
#include "voxelmap.h"
#include "ifusionalgorithm.h"
#include <QMatrix4x4>
#include <QObject>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// CloudAccumulator — tích lũy PointCloud qua VoxelMap EMA.
//
// Pipeline mới: mỗi cloud được transform về frame của waypoint 0 bằng
// T_cam_i_inv trước khi nạp vào VoxelMap.
// Với non-orbital hoặc waypoint 0: truyền QMatrix4x4() (Identity).
//
// Main thread only — không cần mutex vì Qt signal/slot đảm bảo sequencing.
// ─────────────────────────────────────────────────────────────────────────────

class CloudAccumulator : public IFusionAlgorithm
{
    Q_OBJECT
public:
    explicit CloudAccumulator(QObject *parent = nullptr);

    void setEnabled(bool on) override;
    bool isEnabled() const { return _enabled; }

    /// Thêm cloud mới, transform về frame 0 bằng T_cam_i_inv trước khi tích lũy.
    /// Với waypoint 0 hoặc non-orbital: truyền QMatrix4x4() (Identity).
    void addCloud(const PointCloud &cloud, const QMatrix4x4 &T_cam_i_inv = QMatrix4x4()) override;

    void clear() override;
    void loadRawCloud(const std::vector<float> &xyz, const std::vector<uint8_t> &rgb, int count) override;

    int count()      const { return _accumCount; }
    int voxelCount() const { return _voxelMap.size(); }

    // Voxel size — áp dụng sau lần clear() tiếp theo
    void  setVoxelSize(float mm) override { _voxelMap.setVoxelSize(mm); }
    float voxelSize()  const     { return _voxelMap.voxelSize(); }

    // Scan mask — chỉ giữ điểm trong sphere (hệ frame 0).
    // Center XYZ được nhập trực tiếp trong frame 0 (sau transform) nên
    // không phụ thuộc vào R_cam_tool hay hướng trục camera.
    void setScanMask(bool on,
                     float centerX = 0.f, float centerY = 0.f, float centerZ = 500.f,
                     float radius  = 300.f) override;

    void setIcpEnabled(bool on) override { _useIcp = on; }
    bool isIcpEnabled() const { return _useIcp; }

    // Export access — main thread only
    const std::vector<float>   &xyz() const { return _accumXYZ; }
    const std::vector<uint8_t> &rgb() const { return _accumRGB; }

private:
    void _appendToFlatBuffers(const PointCloud &pc, const QMatrix4x4 &T);
    void _rebuildFlatBuffers (const PointCloud &pc, const QMatrix4x4 &T);

    bool _enabled = false;
    bool _useIcp  = false;

    VoxelMap _voxelMap;

    std::vector<float>   _accumXYZ;
    std::vector<uint8_t> _accumRGB;
    int                  _accumCount = 0;

    bool  _useScanMask = false;
    float _scanCenterX = 0.f;
    float _scanCenterY = 0.f;
    float _scanCenterZ = 500.f;
    float _scanRadius  = 300.f;
};

#endif // CLOUDACCUMULATOR_H