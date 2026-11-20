#include "cloudaccumulator.h"
#include <QMatrix4x4>
#include <cmath>

// ─── Constructor ──────────────────────────────────────────────────────────────

CloudAccumulator::CloudAccumulator(QObject *parent)
    : IFusionAlgorithm(parent)
{
}

// ─── Control ──────────────────────────────────────────────────────────────────

void CloudAccumulator::setEnabled(bool on)
{
    _enabled = on;
}

void CloudAccumulator::setScanMask(bool on,
                                   float centerX, float centerY, float centerZ,
                                   float radius)
{
    _useScanMask = on;
    _scanCenterX = centerX;
    _scanCenterY = centerY;
    _scanCenterZ = centerZ;
    _scanRadius  = radius;
}

// ─── addCloud ─────────────────────────────────────────────────────────────────

void CloudAccumulator::addCloud(const PointCloud &cloud, const QMatrix4x4 &T_cam_i_inv)
{
    if (!_enabled || cloud.count <= 0) return;

    // Transform cloud về frame 0 trước khi nạp vào VoxelMap.
    // Nếu T là Identity (waypoint 0 / non-orbital) thì chỉ copy coords.
    PointCloud transformed = cloud; // copy (vectors được copy)

    QMatrix4x4 refinedTransform = T_cam_i_inv;
    if (_useIcp && _voxelMap.size() > 0) {
        _voxelMap.computeICP(cloud, refinedTransform, 5);
    }

    const bool isIdentity = refinedTransform.isIdentity();
    if (!isIdentity) {
        const float r00 = refinedTransform(0,0), r01 = refinedTransform(0,1), r02 = refinedTransform(0,2), tx = refinedTransform(0,3);
        const float r10 = refinedTransform(1,0), r11 = refinedTransform(1,1), r12 = refinedTransform(1,2), ty = refinedTransform(1,3);
        const float r20 = refinedTransform(2,0), r21 = refinedTransform(2,1), r22 = refinedTransform(2,2), tz = refinedTransform(2,3);

        float *xyz = transformed.xyz.data();
        const int n = transformed.count;
        for (int i = 0; i < n; ++i) {
            const float x = xyz[i*3+0];
            const float y = xyz[i*3+1];
            const float z = xyz[i*3+2];
            xyz[i*3+0] = r00*x + r01*y + r02*z + tx;
            xyz[i*3+1] = r10*x + r11*y + r12*z + ty;
            xyz[i*3+2] = r20*x + r21*y + r22*z + tz;
        }
    }

    // Áp dụng scan mask (sphere filter trong frame 0)
    if (_useScanMask && transformed.count > 0) {
        const float r2 = _scanRadius * _scanRadius;
        PointCloud masked;
        masked.xyz.reserve(transformed.count * 3);
        masked.rgb.reserve(transformed.count * 3);

        const float *xyz = transformed.xyz.data();
        const uint8_t *rgb = transformed.rgb.data();
        const int n = transformed.count;

        for (int i = 0; i < n; ++i) {
            const float dx = xyz[i*3+0] - _scanCenterX;
            const float dy = xyz[i*3+1] - _scanCenterY;
            const float dz = xyz[i*3+2] - _scanCenterZ;
            if (dx*dx + dy*dy + dz*dz <= r2) {
                masked.xyz.push_back(xyz[i*3+0]);
                masked.xyz.push_back(xyz[i*3+1]);
                masked.xyz.push_back(xyz[i*3+2]);
                if (!transformed.rgb.empty()) {
                    masked.rgb.push_back(rgb[i*3+0]);
                    masked.rgb.push_back(rgb[i*3+1]);
                    masked.rgb.push_back(rgb[i*3+2]);
                }
                ++masked.count;
            }
        }
        transformed = std::move(masked);
    }

    if (transformed.count <= 0) return;

    ++_accumCount;

    // Feed into voxel map - returns only the newly-promoted voxels (delta)
    PointCloud delta = _voxelMap.addCloud(transformed);

    // If using ICP, existing points in the VoxelMap get their EMA averages updated.
    // We must trigger a full rebuild so the PointCloudWidget renders the new refined positions!
    if (_useIcp || _voxelMap.needsFullRebuild()) {
        _voxelMap.clearRebuildFlag();
        PointCloud full = _voxelMap.toPointCloud();
        _rebuildFlatBuffers(full, QMatrix4x4());
        emit fullRebuildReady(full);
    } else if (delta.count > 0) {
        _appendToFlatBuffers(delta, QMatrix4x4());
        emit deltaReady(delta);
    }

    emit statsUpdated(_voxelMap.size(), _accumCount);
}

// ─── clear ────────────────────────────────────────────────────────────────────

void CloudAccumulator::clear()
{
    _voxelMap.clear();
    _accumXYZ.clear();
    _accumRGB.clear();
    _accumCount = 0;

    emit fullRebuildReady(PointCloud{});
    emit statsUpdated(0, 0);
}

void CloudAccumulator::loadRawCloud(const std::vector<float> &xyz, const std::vector<uint8_t> &rgb, int count)
{
    clear();
    
    PointCloud full;
    full.count = count;
    full.xyz = xyz;
    full.rgb = rgb;
    
    _accumCount = count;
    
    emit fullRebuildReady(full);
}

// ─── Internal flat buffer helpers ─────────────────────────────────────────────

void CloudAccumulator::_appendToFlatBuffers(const PointCloud &pc, const QMatrix4x4 &T)
{
    const bool identity = T.isIdentity();
    const int n = pc.count;
    _accumXYZ.reserve(_accumXYZ.size() + n * 3);
    _accumRGB.reserve(_accumRGB.size() + n * 3);

    for (int i = 0; i < n; ++i) {
        float x = pc.xyz[i*3+0];
        float y = pc.xyz[i*3+1];
        float z = pc.xyz[i*3+2];

        if (!identity) {
            float nx = T(0,0)*x + T(0,1)*y + T(0,2)*z + T(0,3);
            float ny = T(1,0)*x + T(1,1)*y + T(1,2)*z + T(1,3);
            float nz = T(2,0)*x + T(2,1)*y + T(2,2)*z + T(2,3);
            x = nx; y = ny; z = nz;
        }

        _accumXYZ.push_back(x);
        _accumXYZ.push_back(y);
        _accumXYZ.push_back(z);

        if ((int)pc.rgb.size() >= (i+1)*3) {
            _accumRGB.push_back(pc.rgb[i*3+0]);
            _accumRGB.push_back(pc.rgb[i*3+1]);
            _accumRGB.push_back(pc.rgb[i*3+2]);
        } else {
            _accumRGB.push_back(180);
            _accumRGB.push_back(180);
            _accumRGB.push_back(180);
        }
    }
}

void CloudAccumulator::_rebuildFlatBuffers(const PointCloud &pc, const QMatrix4x4 &T)
{
    _accumXYZ.clear();
    _accumRGB.clear();
    _appendToFlatBuffers(pc, T);
}
