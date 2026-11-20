#include "depthprojector.h"
#include <cmath>
#include <QtGlobal>

static constexpr float MAX_CLOUD_Z_MM = 2000.f;

void DepthProjector::setIntrinsics(const CameraIntrinsics &intr)
{
    _intr = intr;
}

bool DepthProjector::backproject(const float *depthData, int W, int H,
                               const std::vector<float> &invFxU,
                               const std::vector<float> &invFyV,
                               int u, int v,
                               float &px, float &py, float &pz) const
{
    if (u < 0 || u >= W || v < 0 || v >= H) return false;
    const float z = depthData[v * W + u];
    if (z < 50.f || z > MAX_CLOUD_Z_MM || std::isnan(z)) return false;
    px = invFxU[u] * z;
    py = invFyV[v] * z;
    pz = z;
    return true;
}

PointCloud DepthProjector::build(const RGBFrame &rgb, const DepthFrame &depth, const QVector<float> &z_mm_array) const
{
    const int W = depth.width;
    const int H = depth.height;
    const int pixels = W * H;

    PointCloud cloud;
    cloud.xyz.resize(pixels * 3);
    cloud.rgb.resize(pixels * 3);
    cloud.nxyz.resize(pixels * 3, 0.f);

    const float fx = _intr.fx, fy = _intr.fy;
    const float cx = _intr.cx, cy = _intr.cy;
    const bool  hasRgb  = rgb.valid;
    const float scaleX  = hasRgb ? float(rgb.width)  / float(W) : 1.f;
    const float scaleY  = hasRgb ? float(rgb.height) / float(H) : 1.f;
    const int   rgbW    = rgb.width;

    std::vector<float> invFxU(W), invFyV(H);
    for (int u = 0; u < W; ++u) invFxU[u] = (u - cx) / fx;
    for (int v = 0; v < H; ++v) invFyV[v] = (v - cy) / fy;

    std::vector<int> rgbRowBase(H);
    std::vector<int> rgbColIdx(W);
    if (hasRgb) {
        const int rgbH = rgb.height;
        for (int v = 0; v < H; ++v) {
            int rv = qBound(0, int(v * scaleY), rgbH - 1);
            rgbRowBase[v] = rv * rgbW * 3;
        }
        for (int u = 0; u < W; ++u) {
            int ru = qBound(0, int(u * scaleX), rgbW - 1);
            rgbColIdx[u]  = ru * 3;
        }
    }

    const float  *depthPtr = z_mm_array.constData();
    const uchar  *rgbPtr   = hasRgb
                              ? reinterpret_cast<const uchar*>(rgb.data.constData()) : nullptr;
    float   *outXYZ  = cloud.xyz.data();
    uint8_t *outRGB  = cloud.rgb.data();
    float   *outNXYZ = cloud.nxyz.data();
    int count = 0;

    // ── Lambda để tính normal ─────────────────────────────────────────────────
    auto calcNormal = [&](int u, int v, float px, float py, float pz, float *nml) {
        float pux, puy, puz, pvx, pvy, pvz;
        const bool hasPu = backproject(depthPtr, W, H, invFxU, invFyV, u+1, v,   pux, puy, puz);
        const bool hasPv = backproject(depthPtr, W, H, invFxU, invFyV, u,   v+1, pvx, pvy, pvz);

        if (hasPu && hasPv) {
            float ex = pux - px, ey = puy - py, ez = puz - pz;
            float fx2= pvx - px, fy2= pvy - py, fz2= pvz - pz;
            float nx = ey * fz2 - ez * fy2;
            float ny = ez * fx2 - ex * fz2;
            float nz2= ex * fy2 - ey * fx2;
            float len = std::sqrt(nx*nx + ny*ny + nz2*nz2);
            if (len > 1e-8f) {
                nml[0] = nx / len;
                nml[1] = ny / len;
                nml[2] = nz2 / len;
                return;
            }
        }
        nml[0] = nml[1] = nml[2] = 0.f;
    };

    if (hasRgb) {
        for (int v = 0; v < H; ++v) {
            const float *rowDepth   = depthPtr + v * W;
            const int    rowRgbBase = rgbRowBase[v];
            const float  invFy_v    = invFyV[v];
            for (int u = 0; u < W; ++u) {
                const float z = rowDepth[u];
                if (z < 50.f || z > MAX_CLOUD_Z_MM || std::isnan(z)) continue;

                float *xyz = outXYZ + count * 3;
                xyz[0] = invFxU[u] * z;
                xyz[1] = invFy_v   * z;
                xyz[2] = z;

                const uchar *src = rgbPtr + rowRgbBase + rgbColIdx[u];
                uint8_t *col = outRGB + count * 3;
                col[0] = src[0]; col[1] = src[1]; col[2] = src[2];

                calcNormal(u, v, xyz[0], xyz[1], xyz[2], outNXYZ + count * 3);
                ++count;
            }
        }
    } else {
        constexpr float zBase  = 200.f;
        constexpr float zScale = 1.f / 3000.f;
        for (int v = 0; v < H; ++v) {
            const float *rowDepth = depthPtr + v * W;
            const float  invFy_v  = invFyV[v];
            for (int u = 0; u < W; ++u) {
                const float z = rowDepth[u];
                if (z < 50.f || z > MAX_CLOUD_Z_MM || std::isnan(z)) continue;

                float *xyz = outXYZ + count * 3;
                xyz[0] = invFxU[u] * z;
                xyz[1] = invFy_v   * z;
                xyz[2] = z;

                const float t = qBound(0.f, (z - zBase) * zScale, 1.f);
                uint8_t *col = outRGB + count * 3;
                col[0] = uint8_t(t * 255.f);
                col[1] = uint8_t((1.f - std::abs(t - 0.5f) * 2.f) * 255.f);
                col[2] = uint8_t((1.f - t) * 255.f);

                calcNormal(u, v, xyz[0], xyz[1], xyz[2], outNXYZ + count * 3);
                ++count;
            }
        }
    }

    cloud.xyz.resize(count * 3);
    cloud.rgb.resize(count * 3);
    cloud.nxyz.resize(count * 3);
    cloud.count = count;
    return cloud;
}