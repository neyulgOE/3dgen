#ifndef TYPES_H
#define TYPES_H

#include <QString>
#include <QVector>
#include <QByteArray>
#include <QDateTime>
#include <vector>
#include <cstdint>
#include <cmath>
#include <QMatrix4x4>   // thêm include nếu chưa có


// ─── Point cloud ─────────────────────────────────────────────────────────────

struct PointCloud {
    std::vector<float>   xyz;
    std::vector<uint8_t> rgb;
    std::vector<float>   nxyz;  // unit normals; (0,0,0) = sentinel
    int count = 0;
};

// ─── Camera intrinsics ────────────────────────────────────────────────────────

struct CameraIntrinsics {
    QString cameraModel = "d435f";
    float fx = 615.f, fy = 615.f;
    float cx = 320.f, cy = 240.f;
    int depthW = 640, depthH = 480;
    int rgbW   = 640, rgbH   = 480;
};

// ─── Camera frame types ───────────────────────────────────────────────────────

enum class CameraStreamType { RGB, Depth, RGBD };

struct IMUData {
    quint64 timestamp_ns = 0;
    QString type; // "accel" or "gyro"
    float x = 0.f, y = 0.f, z = 0.f;
};

struct RGBFrame {
    int width = 0, height = 0;
    QByteArray data;
    quint64 timestamp_ns = 0;
    bool valid = false;
};

struct DepthFrame {
    int width = 0, height = 0;
    QVector<float> z_mm_data;
    float z_min = 0.f, z_max = 0.f;
    quint64 timestamp_ns = 0;
    bool valid = false;
};

struct RGBDFrame {
    RGBFrame   rgb;
    DepthFrame depth;
    bool valid = false;
};


// ─── Robot pose ───────────────────────────────────────────────────────────────

struct RobotPose {
    double x=0, y=0, z=0;    // mm
    double rx=0, ry=0, rz=0; // deg

    // Quaternion cho slerp interpolation
    double qw=1, qx=0, qy=0, qz=0;

    quint64 timestamp_ns = 0;
    bool valid = false;
};

// ─── Geometry frame — PointCloud đã ở Base frame, kèm pose metadata ──────────
// Thay thế PointCloud trong signal chain AppController → MainWindow.
// pose metadata cần thiết cho surface reconstruction (biết viewing direction).

struct GeometryFrame {
    PointCloud  cloud_raw;
    RobotPose   pose;           // giữ để UI debug hiển thị
    QMatrix4x4  camInvTransform; // transform đưa cloud về frame waypoint 0
    // Identity = waypoint 0 hoặc non-orbital mode
    quint64     capturedAt = 0;
};
// ─── Pose stamped ─────────────────────────────────────────────────────────────

struct PoseStamped {
    RobotPose pose;
    quint64   timestamp_ns = 0;
};

// ─── Scan frame ───────────────────────────────────────────────────────────────

struct ScanFrame {
    RGBDFrame  rgbd;
    RobotPose  pose;
    quint64    capturedAt;
    bool valid = false;
};

// ─── Tracking frame (for SLAM/VIO) ─────────────────────────────────────────────

struct TrackingFrame {
    RGBDFrame rgbd;
    QMatrix4x4 pose; // Ma trận 4x4 từ VIO
    quint64 timestamp_ns = 0;
    bool valid = false;
};

// ─── Stop-and-scan waypoint ───────────────────────────────────────────────────

struct ScanWaypoint {
    double dx = 0, dy = 0, dz = 0;
    double droll = 0, dpitch = 0, dyaw = 0;
    int    nFrames  = 5;
    int    settleMs = 400;

    // Orbital mode: index vào OrbitalWaypoint list để lấy đúng transform.
    // -1 = không phải orbital (manual waypoint hoặc Home intermediate).
    // >= 0 = capture waypoint, tương ứng _orbitalWps[orbitalWpIndex].
    int    orbitalWpIndex = -1;
};

// ─────────────────────────────────────────────────────────────
// Marker dictionary type reported by the camera
// ─────────────────────────────────────────────────────────────
enum class MarkerType
{
    Any      = 0,   // no filter (wildcard)
    Dict5x5  = 1,   // DICT_5X5_100
    Dict6x6  = 2,   // DICT_6X6_100
};

inline QString markerTypeName(MarkerType t)
{
    switch (t) {
    case MarkerType::Dict5x5: return "5x5_100";
    case MarkerType::Dict6x6: return "6x6_100";
    default:                  return "Any";
    }
}

struct MarkerDetection
{
    int id = -1;
    MarkerType markerType = MarkerType::Any;  // dictionary type from camera
    bool has_xyz = false;
    double x = 0.0, y = 0.0, z = 0.0;
    int center_u = 0, center_v = 0;
    int c0_u = 0, c0_v = 0;
    int c1_u = 0, c1_v = 0;
    int c2_u = 0, c2_v = 0;
    int c3_u = 0, c3_v = 0;

    // ── Per-marker pose / orientation (from ArucoMarker proto) ──────────
    // Euler angles from solvePnP rvec (radians). R = Rz(yaw)*Ry(pitch)*Rx(roll).
    bool   has_rpy = false;
    double roll = 0.0, pitch = 0.0, yaw = 0.0;

    // Rodrigues rotation vector (marker frame -> camera frame).
    double rx = 0.0, ry = 0.0, rz = 0.0;

    // solvePnP translation vector (debug/backup; main x,y,z normally come from depth).
    bool   has_pnp_t = false;
    double tx = 0.0, ty = 0.0, tz = 0.0;

    // Marker normal vector in camera frame (marker local Z axis).
    bool   has_normal = false;
    double nx = 0.0, ny = 0.0, nz = 0.0;

    // 2D approach angle projected onto AMR-like plane (x_2d=z_cam, y_2d=-x_cam).
    bool   has_approach_theta = false;
    double approach_theta = 0.0;
};

struct MarkerFrame1202
{
    QString type;
    quint64 pub_nsec  = 0;
    quint64 data_nsec = 0;
    int width = 0, height = 0;
    bool found = false;
    int marker_count = 0;
    QVector<MarkerDetection> markers;

    // ── Orientation cấp frame (tính từ centre-line 2 marker, web tính sẵn) ──
    // has_yaw: valid khi >= 2 marker có xyz hợp lệ.
    bool   has_yaw = false;
    double yaw_deg = 0.0;
    // has_roll_pitch: valid khi >= 1 marker có plane normal hợp lệ.
    bool   has_roll_pitch = false;
    double roll_deg  = 0.0;
    double pitch_deg = 0.0;
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

inline QString nowStringForFile() {
    return QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
}

#endif // TYPES_H