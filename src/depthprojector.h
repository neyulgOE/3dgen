#ifndef DEPTHPROJECTOR_H
#define DEPTHPROJECTOR_H

#include "types.h"

// ─────────────────────────────────────────────────────────────────────────────
// DepthProjector — backproject RGBD frame thành PointCloud (Camera frame).
//
// Tách ra khỏi CameraWorker để có thể reuse và test độc lập.
// Output chưa được transform sang Base frame — AppController/PoseTracker làm việc đó.
// ─────────────────────────────────────────────────────────────────────────────

class DepthProjector {
public:
    void setIntrinsics(const CameraIntrinsics &intr);

    // Build point cloud từ RGBD frame (Camera frame, chưa transform sang Base)
    PointCloud build(const RGBFrame &rgb, const DepthFrame &depth, const QVector<float> &z_mm_array) const;

private:
    // Backproject pixel (u,v) → 3D point. Trả về false nếu depth invalid.
    bool backproject(const float *depthData, int W, int H,
                     const std::vector<float> &invFxU,
                     const std::vector<float> &invFyV,
                     int u, int v,
                     float &px, float &py, float &pz) const;

    CameraIntrinsics _intr;
};

#endif // DEPTHPROJECTOR_H