#ifndef POINTCLOUDWIDGET_H
#define POINTCLOUDWIDGET_H

#include "types.h"
#include "cloudaccumulator.h"
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QVector3D>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QPoint>

// ─── PointCloudWidget ─────────────────────────────────────────────────────────
// Trách nhiệm duy nhất: render OpenGL + mouse/keyboard navigation.
//
// Accumulation logic đã được tách sang CloudAccumulator.
// Widget nhận delta/full rebuild từ CloudAccumulator qua signal/slot,
// rồi upload lên GPU và vẽ.
// ─────────────────────────────────────────────────────────────────────────────

class PointCloudWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
public:
    explicit PointCloudWidget(QWidget *parent = nullptr);
    ~PointCloudWidget() override;

    void setIntrinsics(const CameraIntrinsics &intr);
    void resetView();
    void autoCenter();   // tính centroid từ _accumXYZ, set _target
    void setViewPreset(int preset); // 0=Reset, 1=Top, 2=Front, 3=Side, 4=Iso
    void setPointSize(float sz);
    float pointSize() const { return _pointSize; }
    void setShowGrid(bool show);
    bool showGrid() const { return _showGrid; }
    bool captureScreenshot(const QString &savePath = QString());
    int  accumulatedCount() const { return _accumCount; }

    // Accessor cho CloudAccumulator (owned by MainWindow)
    void setAccumulator(CloudAccumulator *accum);
    void setScanMask(bool enabled, float cx, float cy, float cz, float radius);

public slots:
    // Nhận point cloud live (non-accum mode)
    void updatePointCloud(const PointCloud &cloud);

    // Nhận từ CloudAccumulator signals
    void onDeltaReady(const PointCloud &delta);
    void onFullRebuildReady(const PointCloud &full);

    void setAccumulate(bool on);
    void clearAccumulated();   // delegates to _accum
    void setShowFov(bool show);

    /// Nhận danh sách T_base_cam từ AppController — vẽ camera trajectory.
    void onCameraTrajectoryUpdated(const QVector<QMatrix4x4> &poses);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;

    void mousePressEvent  (QMouseEvent  *e) override;
    void mouseMoveEvent   (QMouseEvent  *e) override;
    void mouseReleaseEvent(QMouseEvent  *e) override;
    void wheelEvent       (QWheelEvent  *e) override;
    void keyPressEvent    (QKeyEvent    *e) override;

private:
    void uploadCurrentVBO();
    void appendAccumVBO();
    void fullUploadAccumVBO();
    void drawGrid();
    void drawScanMaskSphere();
    void drawCameraTrajectory();  ///< Vẽ frustum/axes tại mỗi camera pose
    QMatrix4x4 mvpMatrix() const;

    // ── Intrinsics ────────────────────────────────────────────────────────────
    CameraIntrinsics _intr;

    // ── Accumulator (not owned — owned by MainWindow) ─────────────────────────
    CloudAccumulator *_accum = nullptr;
    bool              _accumulate = false;
    bool              _showFov = true;
    bool              _showGrid = true;
    float             _pointSize = 2.0f;

    // ── Pending GPU uploads from accumulator ──────────────────────────────────
    PointCloud _pendingFull;
    PointCloud _pendingDelta;
    bool       _accumDirtyFull   = false;
    bool       _accumDirtyAppend = false;
    int        _accumCount       = 0;   // mirrors accum->count() for HUD display

    // ── Point cloud data (live mode, main thread only) ────────────────────────
    PointCloud _curCloud;
    bool       _cloudDirty = false;

    // ── OpenGL objects ────────────────────────────────────────────────────────
    GLuint _vboXYZ = 0;
    GLuint _vboRGB = 0;
    GLuint _accumVboXYZ = 0;
    GLuint _accumVboRGB = 0;
    int    _uploadedCount = 0;
    int    _accumUploadedCount = 0;
    int    _accumVboCapacityXYZ = 0;
    int    _accumVboCapacityRGB = 0;

    // ── Buffers cho append (CPU side, để re-upload khi grow) ──────────────────
    std::vector<float>   _accumXYZ;
    std::vector<uint8_t> _accumRGB;
    std::vector<float>   _deltaXYZ;
    std::vector<uint8_t> _deltaRGB;
    int                  _deltaCount = 0;

    // ── Camera orbit state ────────────────────────────────────────────────────
    float  _yaw      =   0.f;
    float  _pitch    = -35.f;
    float  _distance = 1200.f;
    QVector3D _target {0.f, 0.f, 750.f};

    // ── Mouse drag ────────────────────────────────────────────────────────────
    bool   _dragging = false;
    bool   _panning  = false;
    QPoint _lastMouse;

    // ── Scan mask sphere ──────────────────────────────────────────────────────
    bool  _maskEnabled = false;
    float _maskCx      = 0.f;
    float _maskCy      = 0.f;
    float _maskCz      = 0.f;
    float _maskRadius  = 300.f;

    // ── Camera trajectory (Base frame, T_base_cam) ────────────────────────
    QVector<QMatrix4x4> _camPoses;  ///< mỗi phần tử = T_base_cam tại waypoint i
};

#endif // POINTCLOUDWIDGET_H