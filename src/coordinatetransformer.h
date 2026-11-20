#ifndef COORDINATETRANSFORMER_H
#define COORDINATETRANSFORMER_H

#include "types.h"
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QVector3D>
#include <cmath>

/**
 * CoordinateTransformer
 *
 * Sau khi refactor (bỏ pose robot tuyệt đối):
 *   - Chỉ quản lý R_cam_tool (hiệu chuẩn cố định).
 *   - Không còn R_tool_base, recompose(), interpolatePose(), v.v.
 *
 * Pipeline reconstruction (camera-frame-0 làm gốc):
 *   Tại waypoint i, robot đã dịch chuyển delta (dx,dy,dz,droll,dpitch,dyaw)
 *   so với waypoint 0. Để đưa PointCloud tại waypoint i về frame của waypoint 0:
 *
 *     1. R_delta_tool = Rz(dyaw)*Ry(dpitch)*Rx(droll)  (ZYX Euler, deg→rad)
 *     2. Chuyển sang camera frame:
 *            R_fwd   = R_cam_tool * R_delta_tool * R_cam_tool^T
 *            t_total = t_delta_tool + (R_delta_tool - I) * t_cam_in_tool
 *            t_fwd   = R_cam_tool * t_total
 *     3. Forward model (cam_0 → cam_i):
 *            p_i = R_fwd^T * p_0 - R_fwd^T * t_fwd
 *            T_i_W = [R_fwd^T | -R_fwd^T * t_fwd]
 *     4. Inverse T_W_i (cam_i → cam_0):
 *            R_W_i = R_fwd   (NOT R_fwd^T)
 *            t_W_i = t_fwd   (NOT -R_fwd^T * t_fwd)
 *     5. Trả về QMatrix4x4 T = [R_fwd | t_fwd]
 *
 *   Verify: p_0 = R_fwd * p_i + t_fwd
 *         = R_fwd*(R_fwd^T*p_0 - R_fwd^T*t_fwd) + t_fwd = p_0 ✓
 *
 * Waypoint 0: truyền OrbitalWaypoint zero-filled → T = Identity.
 * Non-orbital mode: caller truyền QMatrix4x4() (Identity).
 */
class CoordinateTransformer
{
public:
    // ── Cấu hình R_cam_tool ──────────────────────────────────────────────────

    /// Đặt 3 hàng (row-vectors) của R_cam_tool:
    ///   hàng 0 = chiều Camera-X biểu diễn bằng trục Tool
    ///   hàng 1 = chiều Camera-Y
    ///   hàng 2 = chiều Camera-Z
    void setCamToToolAxes(const QVector3D &xAxis,
                          const QVector3D &yAxis,
                          const QVector3D &zAxis);

    /// Đặt trực tiếp toàn bộ ma trận R_cam_tool (row-major).
    void setCamToToolMatrix(const QMatrix3x3 &R);

    /// Đặt vector t_cam_in_tool (vị trí Camera so với Tool Center, mm).
    void setCamToToolOffset(const QVector3D &offset);

    // ── Tính transform nghịch đảo cho waypoint i ─────────────────────────────

    /// Thiết lập Absolute Home TCP Pose (Base Coordinate System)
    void setHomeTcpPose(double x, double y, double z, double rx, double ry, double rz);

    /// Tính QMatrix4x4 T_cam_i_inv để đưa cloud tại waypoint i về frame 0.
    /// Offset [dx,dy,dz,droll,dpitch,dyaw] trong hệ Base (confirm từ robot TM).
    QMatrix4x4 computeInvTransform(double dx, double dy, double dz,
                                   double droll, double dpitch, double dyaw) const;

    /// Tính T_base_cam tuyệt đối trong Base frame để hiển thị camera trajectory.
    /// Waypoint 0 (Home): truyền tất cả = 0.
    QMatrix4x4 computeBaseCamTransform(double dx, double dy, double dz,
                                       double droll, double dpitch, double dyaw) const;
    
    // Helper để tạo ma trận 4x4 từ TCP pose
    static QMatrix4x4 poseToMatrix(double x, double y, double z, double rx, double ry, double rz);

    // ── Tiện ích transform (vẫn giữ để PointCloudWidget / debug dùng) ────────

    /// Transform toàn bộ PointCloud bằng một QMatrix4x4 (in-place).
    static void transformCloud(PointCloud &cloud, const QMatrix4x4 &T);

    /// Transform normals bằng phần rotation của QMatrix4x4 (in-place).
    /// Normal (0,0,0) là sentinel → bỏ qua.
    static void transformNormals(PointCloud &cloud, const QMatrix4x4 &T);

    // ── Truy vấn ─────────────────────────────────────────────────────────────
    const QMatrix3x3 &camToToolMatrix() const { return _R_cam_tool; }

private:
    QMatrix3x3 _R_cam_tool{};  ///< hàng i = trục Camera-i trong Tool frame
                               ///< → _R_cam_tool * v_tool = v_cam (Tool→Cam)
    QVector3D  _t_cam_in_tool{}; ///< Vị trí gốc Camera trong TCP frame (mm)

    // Lưu trữ Home TCP Pose
    double _home_tcp_x = 0;
    double _home_tcp_y = 0;
    double _home_tcp_z = 0;
    double _home_tcp_rx = 0;
    double _home_tcp_ry = 0;
    double _home_tcp_rz = 0;
    bool _has_home_pose = false;

    // ── Private helpers ─────────────────────────────────────────────
    /// T_tcp_cam: biến đổi điểm TỪ Cam SANG TCP.
    ///   Rotation = R_cam_tool^T, Translation = t_cam_in_tool
    QMatrix4x4 buildTcpCam() const;

    /// Xây dựng T_base_tcp tại waypoint với offset trong Base frame.
    ///   t = t_home + [dx,dy,dz], R = R_delta * R_home
    QMatrix4x4 buildTcpAtWaypoint(const QMatrix4x4 &T_base_tcp0,
                                   double dx, double dy, double dz,
                                   double droll, double dpitch, double dyaw) const;
};

#endif // COORDINATETRANSFORMER_H