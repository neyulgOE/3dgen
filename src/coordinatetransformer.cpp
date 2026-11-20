#include "coordinatetransformer.h"
#include <cmath>

// ── Cấu hình R_cam_tool
// ───────────────────────────────────────────────────────

void CoordinateTransformer::setCamToToolAxes(const QVector3D &xAxis,
                                             const QVector3D &yAxis,
                                             const QVector3D &zAxis) {
    _R_cam_tool(0, 0) = xAxis.x();
    _R_cam_tool(0, 1) = xAxis.y();
    _R_cam_tool(0, 2) = xAxis.z();
    _R_cam_tool(1, 0) = yAxis.x();
    _R_cam_tool(1, 1) = yAxis.y();
    _R_cam_tool(1, 2) = yAxis.z();
    _R_cam_tool(2, 0) = zAxis.x();
    _R_cam_tool(2, 1) = zAxis.y();
    _R_cam_tool(2, 2) = zAxis.z();
}

void CoordinateTransformer::setCamToToolMatrix(const QMatrix3x3 &R) {
    _R_cam_tool = R;
}

void CoordinateTransformer::setCamToToolOffset(const QVector3D &offset) {
    _t_cam_in_tool = offset;
}

void CoordinateTransformer::setHomeTcpPose(double x, double y, double z, double rx, double ry, double rz) {
    _home_tcp_x = x;
    _home_tcp_y = y;
    _home_tcp_z = z;
    _home_tcp_rx = rx;
    _home_tcp_ry = ry;
    _home_tcp_rz = rz;
    _has_home_pose = true;
}

QMatrix4x4 CoordinateTransformer::poseToMatrix(double x, double y, double z, double rx, double ry, double rz) {
    constexpr double DEG2RAD = M_PI / 180.0;
    const double rad_x = rx * DEG2RAD;
    const double rad_y = ry * DEG2RAD;
    const double rad_z = rz * DEG2RAD;

    const double cx = std::cos(rad_x), sx = std::sin(rad_x);
    const double cy = std::cos(rad_y), sy = std::sin(rad_y);
    const double cz = std::cos(rad_z), sz = std::sin(rad_z);

    QMatrix4x4 T;
    // ZYX Euler order
    T(0, 0) = float(cy * cz);
    T(0, 1) = float(cz * sx * sy - cx * sz);
    T(0, 2) = float(cx * cz * sy + sx * sz);
    T(1, 0) = float(cy * sz);
    T(1, 1) = float(cx * cz + sx * sy * sz);
    T(1, 2) = float(-cz * sx + cx * sy * sz);
    T(2, 0) = float(-sy);
    T(2, 1) = float(cy * sx);
    T(2, 2) = float(cx * cy);
    
    T(0, 3) = float(x);
    T(1, 3) = float(y);
    T(2, 3) = float(z);
    
    T(3, 0) = 0.0f; T(3, 1) = 0.0f; T(3, 2) = 0.0f; T(3, 3) = 1.0f;
    return T;
}

// ── buildTcpCam — helper nội bộ ────────────────────────────────────────────
// T_tcp_cam: biến đổi điểm TỪ Camera frame SANG TCP frame.
//   p_tcp = R_cam_tool^T * p_cam + t_cam_in_tool
//
// _R_cam_tool được đặt bởi setCamToToolAxes(xAxis,yAxis,zAxis) trong đó
//   hàng i = trục Camera-i biểu diễn trong tọa độ Tool.
//   → _R_cam_tool * v_tool = v_cam  (Tool→Cam)
//   → T_{tcp←cam} rotation = _R_cam_tool^T  (Cam→TCP)
//
QMatrix4x4 CoordinateTransformer::buildTcpCam() const
{
    QMatrix4x4 T;
    T.setToIdentity();
    // Rotation = R_cam_tool^T  (đổi chuyển điểm từ Cam → TCP)
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            T(r, c) = _R_cam_tool(c, r);   // transpose
    // Translation = vị trí gốc tọa độ Camera trong TCP frame
    T(0, 3) = _t_cam_in_tool.x();
    T(1, 3) = _t_cam_in_tool.y();
    T(2, 3) = _t_cam_in_tool.z();
    T(3, 3) = 1.f;
    return T;
}

QMatrix4x4 CoordinateTransformer::buildTcpAtWaypoint(
    const QMatrix4x4 &T_base_tcp0,
    double dx, double dy, double dz,
    double droll, double dpitch, double dyaw) const
{
    // Robot Techman (ReferenceToolLegacy) áp dụng offset trong hệ Tool (TCS):
    //   - T_base_tcpi = T_base_tcp0 * T_delta
    // Bằng chứng: dz>0 đi xuống, dx>0 lùi về base, khớp hoàn toàn với trục Tool mặc định.
    // Keep this model in sync with AppController::robotDoOffset(): translate
    // first, then rotate in place. A single combined pose would rotate the
    // translation vector too and can make accumulated frames lean/shear.
    QMatrix4x4 T_translate = poseToMatrix(dx, dy, dz, 0.0, 0.0, 0.0);
    QMatrix4x4 T_rotate    = poseToMatrix(0.0, 0.0, 0.0, droll, dpitch, dyaw);
    return T_base_tcp0 * T_translate * T_rotate;
}

QMatrix4x4 CoordinateTransformer::computeInvTransform(double dx, double dy,
                                                      double dz, double droll,
                                                      double dpitch,
                                                      double dyaw) const {
    // Waypoint 0 (Home) — không cần biến đổi
    if (dx == 0.0 && dy == 0.0 && dz == 0.0 &&
        droll == 0.0 && dpitch == 0.0 && dyaw == 0.0) {
        return QMatrix4x4();
    }
    if (!_has_home_pose) {
        return QMatrix4x4();
    }

    // ── Bước 1: Pose TCP tại Home ─────────────────────────────────────────────
    QMatrix4x4 T_base_tcp0 = poseToMatrix(_home_tcp_x, _home_tcp_y, _home_tcp_z,
                                          _home_tcp_rx, _home_tcp_ry, _home_tcp_rz);

    // ── Bước 2: Pose TCP tại waypoint i (Base-frame offset) ───────────────────
    QMatrix4x4 T_base_tcpi = buildTcpAtWaypoint(
        T_base_tcp0, dx, dy, dz, droll, dpitch, dyaw);

    // ── Bước 3: T_tcp_cam — biến đổi Camera → TCP ────────────────────────────
    QMatrix4x4 T_tcp_cam = buildTcpCam();

    // ── Bước 4: Pose Camera tuyệt đối trong Base ──────────────────────────────
    // T_base_cam = T_base_tcp * T_tcp_cam
    QMatrix4x4 T_base_cam0 = T_base_tcp0 * T_tcp_cam;
    QMatrix4x4 T_base_cami = T_base_tcpi * T_tcp_cam;

    // ── Bước 5: Transform tương đối cam_i → cam_0 ────────────────────────────
    // p_cam0 = T_base_cam0^-1 * T_base_cami * p_cami
    bool ok = false;
    QMatrix4x4 T_base_cam0_inv = T_base_cam0.inverted(&ok);
    if (!ok) return QMatrix4x4();

    return T_base_cam0_inv * T_base_cami;
}

// ── computeBaseCamTransform — trả về T_base_cam tuyệt đối để visualize ────
QMatrix4x4 CoordinateTransformer::computeBaseCamTransform(
    double dx, double dy, double dz,
    double droll, double dpitch, double dyaw) const
{
    if (!_has_home_pose) return QMatrix4x4();

    QMatrix4x4 T_base_tcp0 = poseToMatrix(_home_tcp_x, _home_tcp_y, _home_tcp_z,
                                          _home_tcp_rx, _home_tcp_ry, _home_tcp_rz);
    QMatrix4x4 T_base_tcp  = (dx==0&&dy==0&&dz==0&&droll==0&&dpitch==0&&dyaw==0)
                              ? T_base_tcp0
                              : buildTcpAtWaypoint(T_base_tcp0, dx, dy, dz, droll, dpitch, dyaw);
    return T_base_tcp * buildTcpCam();
}

// ── transformCloud
// ────────────────────────────────────────────────────────────

void CoordinateTransformer::transformCloud(PointCloud &cloud,
                                           const QMatrix4x4 &T) {
    const float r00 = T(0, 0), r01 = T(0, 1), r02 = T(0, 2), tx = T(0, 3);
    const float r10 = T(1, 0), r11 = T(1, 1), r12 = T(1, 2), ty = T(1, 3);
    const float r20 = T(2, 0), r21 = T(2, 1), r22 = T(2, 2), tz = T(2, 3);

    float *xyz = cloud.xyz.data();
    const int n = cloud.count;
    for (int i = 0; i < n; ++i) {
        const float x = xyz[i * 3 + 0];
        const float y = xyz[i * 3 + 1];
        const float z = xyz[i * 3 + 2];
        xyz[i * 3 + 0] = r00 * x + r01 * y + r02 * z + tx;
        xyz[i * 3 + 1] = r10 * x + r11 * y + r12 * z + ty;
        xyz[i * 3 + 2] = r20 * x + r21 * y + r22 * z + tz;
    }
}

// ── transformNormals
// ──────────────────────────────────────────────────────────

void CoordinateTransformer::transformNormals(PointCloud &cloud,
                                             const QMatrix4x4 &T) {
    if (cloud.nxyz.empty())
        return;

    const float r00 = T(0, 0), r01 = T(0, 1), r02 = T(0, 2);
    const float r10 = T(1, 0), r11 = T(1, 1), r12 = T(1, 2);
    const float r20 = T(2, 0), r21 = T(2, 1), r22 = T(2, 2);

    float *nxyz = cloud.nxyz.data();
    const int n = cloud.count;
    for (int i = 0; i < n; ++i) {
        const float nx = nxyz[i * 3 + 0];
        const float ny = nxyz[i * 3 + 1];
        const float nz = nxyz[i * 3 + 2];
        if (nx == 0.f && ny == 0.f && nz == 0.f)
            continue;
        nxyz[i * 3 + 0] = r00 * nx + r01 * ny + r02 * nz;
        nxyz[i * 3 + 1] = r10 * nx + r11 * ny + r12 * nz;
        nxyz[i * 3 + 2] = r20 * nx + r21 * ny + r22 * nz;
    }
}
