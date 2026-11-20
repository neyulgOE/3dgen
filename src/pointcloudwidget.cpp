#include "pointcloudwidget.h"
#include <QOpenGLFunctions>
#include <QPainter>
#include <QKeyEvent>
#include <QDateTime>
#include <QDir>
#include <cmath>
#include <QtMath>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

PointCloudWidget::PointCloudWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);

    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setSamples(2);
    fmt.setSwapInterval(1);
    setFormat(fmt);
}

PointCloudWidget::~PointCloudWidget()
{
    makeCurrent();
    if (_vboXYZ)      glDeleteBuffers(1, &_vboXYZ);
    if (_vboRGB)      glDeleteBuffers(1, &_vboRGB);
    if (_accumVboXYZ) glDeleteBuffers(1, &_accumVboXYZ);
    if (_accumVboRGB) glDeleteBuffers(1, &_accumVboRGB);
    doneCurrent();
}

// ─── Public API ───────────────────────────────────────────────────────────────

void PointCloudWidget::setIntrinsics(const CameraIntrinsics &intr)
{
    _intr = intr;
    _target = QVector3D(0.f, 0.f, 750.f);
}

void PointCloudWidget::setScanMask(bool enabled, float cx, float cy, float cz, float radius)
{
    _maskEnabled = enabled;
    _maskCx = cx; _maskCy = cy; _maskCz = cz;
    _maskRadius = radius;
    update();
}

void PointCloudWidget::resetView()
{
    _yaw      =   0.f;
    _pitch    = -35.f;
    _distance = 1200.f;
    _target   = QVector3D(0.f, 0.f, 750.f);
    update();
}

void PointCloudWidget::autoCenter()
{
    if (_accumXYZ.empty()) return;
    double cx = 0, cy = 0, cz = 0;
    const int n = int(_accumXYZ.size() / 3);
    for (int i = 0; i < n; ++i) {
        cx += _accumXYZ[i*3+0];
        cy += _accumXYZ[i*3+1];
        cz += _accumXYZ[i*3+2];
    }
    _target = QVector3D(float(cx/n), float(cy/n), float(cz/n));
    update();
}

void PointCloudWidget::setPointSize(float sz)
{
    _pointSize = std::max(1.0f, std::min(10.0f, sz));
    update();
}

void PointCloudWidget::setShowGrid(bool show)
{
    if (_showGrid == show) return;
    _showGrid = show;
    update();
}

void PointCloudWidget::setViewPreset(int preset)
{
    switch (preset) {
    case 0: // Reset
        resetView();
        break;
    case 1: // Top
        _yaw   =   0.f;
        _pitch =  89.9f;
        break;
    case 2: // Front
        _yaw   =   0.f;
        _pitch =   0.f;
        break;
    case 3: // Side
        _yaw   =  90.f;
        _pitch =   0.f;
        break;
    case 4: // Iso
        _yaw   =  45.f;
        _pitch = -35.f;
        break;
    default:
        break;
    }
    update();
}

bool PointCloudWidget::captureScreenshot(const QString &savePath)
{
    QImage img = grabFramebuffer();
    QString path = savePath;
    if (path.isEmpty()) {
        QString dir = QDir::currentPath() + "/screenshots";
        QDir().mkpath(dir);
        path = dir + "/cloud_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
    }
    return img.save(path);
}

void PointCloudWidget::setAccumulator(CloudAccumulator *accum)
{
    _accum = accum;
}

// ─── Slots: nhận từ CloudAccumulator ─────────────────────────────────────────

void PointCloudWidget::onDeltaReady(const PointCloud &delta)
{
    _accumCount += delta.count;

    // Stage delta for GPU upload. CPU merge into _accumXYZ/_accumRGB happens
    // inside appendAccumVBO() AFTER the GPU write, so those vectors always
    // reflect exactly what has been uploaded (_accumUploadedCount points).
    _deltaXYZ.insert(_deltaXYZ.end(), delta.xyz.begin(), delta.xyz.end());
    _deltaRGB.insert(_deltaRGB.end(), delta.rgb.begin(), delta.rgb.end());
    _deltaCount += delta.count;

    _accumDirtyAppend = true;
    update();
}

void PointCloudWidget::onFullRebuildReady(const PointCloud &full)
{
    _pendingFull      = full;
    _accumXYZ         = full.xyz;
    _accumRGB         = full.rgb;
    _accumCount       = full.count;
    _accumDirtyFull   = true;
    _accumDirtyAppend = false;

    // Clear pending delta
    _deltaXYZ.clear();
    _deltaRGB.clear();
    _deltaCount = 0;

    // Re-center camera on the rebuilt cloud (frame 0 coords, not camera-space 750mm)
    if (_accumulate && _accumCount > 0)
        autoCenter();

    update();
}

// ─── Slot: nhận point cloud live (non-accum) ──────────────────────────────────

void PointCloudWidget::updatePointCloud(const PointCloud &cloud)
{
    if (!_accumulate) {
        _curCloud   = cloud;
        _cloudDirty = true;
        update();
    }
    // Khi _accumulate == true, CloudAccumulator xử lý và emit delta/full
    // Widget không cần làm gì ở đây
}

void PointCloudWidget::setAccumulate(bool on)
{
    _accumulate = on;
}

void PointCloudWidget::setShowFov(bool show)
{
    _showFov = show;
    update();
}

void PointCloudWidget::clearAccumulated()
{
    // CloudAccumulator::clear() sẽ emit fullRebuildReady(empty)
    // → onFullRebuildReady() được gọi → GPU clear
    if (_accum) _accum->clear();
}

// ─── OpenGL lifecycle ─────────────────────────────────────────────────────────

void PointCloudWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glClearColor(0.08f, 0.08f, 0.10f, 1.f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glGenBuffers(1, &_vboXYZ);
    glGenBuffers(1, &_vboRGB);
    glGenBuffers(1, &_accumVboXYZ);
    glGenBuffers(1, &_accumVboRGB);
}

void PointCloudWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void PointCloudWidget::paintGL()
{
    // Đảm bảo bật lại Depth Test vì QPainter (chạy ở cuối hàm) có thể đã tắt nó ở frame trước
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // ── Upload / append VBOs ─────────────────────────────────────────────────
    if (_accumulate) {
        if (_accumDirtyFull) {
            fullUploadAccumVBO();
            _accumDirtyFull   = false;
        }
        // Always check append even if full rebuild happened, to apply pending deltas
        if (_accumDirtyAppend) {
            appendAccumVBO();
            _accumDirtyAppend = false;
        }
    } else {
        if (_cloudDirty) {
            uploadCurrentVBO();
            _cloudDirty = false;
        }
    }

    const int drawCount = _accumulate ? _accumUploadedCount : _uploadedCount;
    if (drawCount == 0) {
        QPainter p(this);
        p.setPen(QColor(120, 120, 130));
        p.setFont(QFont("Arial", 11));
        p.drawText(rect(), Qt::AlignCenter,
                   "Chưa có point cloud\nChọn mode RGB-D và bấm Fetch Once hoặc Start Stream");
        return;
    }

    // ── Setup matrices ────────────────────────────────────────────────────────
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    QMatrix4x4 proj;
    float aspect = width() > 0 ? float(width()) / float(height()) : 1.f;
    proj.perspective(45.f, aspect, 10.f, 20000.f);
    glLoadMatrixf(proj.constData());

    glMatrixMode(GL_MODELVIEW);
    glLoadMatrixf(mvpMatrix().constData());

    drawGrid();
    drawScanMaskSphere();
    if (_showFov) {
        drawCameraTrajectory();
    }

    glPointSize(_pointSize);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    if (_accumulate) {
        glBindBuffer(GL_ARRAY_BUFFER, _accumVboXYZ);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, _accumVboRGB);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);
        glDrawArrays(GL_POINTS, 0, _accumUploadedCount);
    } else {
        glBindBuffer(GL_ARRAY_BUFFER, _vboXYZ);
        glVertexPointer(3, GL_FLOAT, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, _vboRGB);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);
        glDrawArrays(GL_POINTS, 0, _uploadedCount);
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // ── HUD overlay ───────────────────────────────────────────────────────────
    QPainter p(this);
    p.setPen(QColor(160, 160, 170));
    p.setFont(QFont("Consolas", 9));
    const int voxels = (_accum && _accumulate) ? _accum->voxelCount() : 0;
    QString accumStr = _accumulate
                           ? QString("  |  🔴 ACCUMULATE ON (voxels: %1)").arg(voxels)
                           : "  |  ⬜ accumulate off";
    p.drawText(8, 18, QString("Points: %1%2  |  Drag: orbit  |  RMB/Shift: pan  |  Wheel: zoom  |  R: reset  |  A: auto-center  |  C: clear")
                          .arg(drawCount).arg(accumStr));
}

// ─── VBO upload helpers ───────────────────────────────────────────────────────

void PointCloudWidget::uploadCurrentVBO()
{
    _uploadedCount = _curCloud.count;
    if (_uploadedCount == 0) return;

    glBindBuffer(GL_ARRAY_BUFFER, _vboXYZ);
    glBufferData(GL_ARRAY_BUFFER,
                 _uploadedCount * 3 * sizeof(float),
                 _curCloud.xyz.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _vboRGB);
    glBufferData(GL_ARRAY_BUFFER,
                 _uploadedCount * 3 * sizeof(uint8_t),
                 _curCloud.rgb.data(),
                 GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void PointCloudWidget::appendAccumVBO()
{
    if (_deltaCount == 0) return;

    const int         newTotal  = _accumUploadedCount + _deltaCount;
    const GLsizeiptr  oldXYZBytes = (GLsizeiptr)_accumUploadedCount * 3 * sizeof(float);
    const GLsizeiptr  oldRGBBytes = (GLsizeiptr)_accumUploadedCount * 3 * sizeof(uint8_t);
    const GLsizeiptr  dXYZBytes   = (GLsizeiptr)_deltaCount          * 3 * sizeof(float);
    const GLsizeiptr  dRGBBytes   = (GLsizeiptr)_deltaCount          * 3 * sizeof(uint8_t);

    // ── Grow helper ───────────────────────────────────────────────────────────
    //
    // On reallocation we need to re-upload the data that is already on the GPU.
    // We use _accumXYZ / _accumRGB as the CPU-side source of truth for that data.
    //
    // IMPORTANT invariant: at the point growVBO() is called, _accumXYZ and
    // _accumRGB contain exactly _accumUploadedCount points — i.e. the delta from
    // onDeltaReady() has NOT yet been merged into those vectors.  The CPU merge
    // happens at the bottom of this function, AFTER the GPU upload, so the
    // invariant is always satisfied.  (Previous code broke this by merging in
    // onDeltaReady before paintGL ran, creating a fragile implicit dependency on
    // delta being at the tail of the vector.)
    auto growVBO = [&](GLuint vbo, int &cap,
                       const void *oldData, GLsizeiptr oldBytes,
                       GLsizeiptr newUnitSize)
    {
        if (newTotal <= cap) return;

        const int newCap = qMax(newTotal, cap * 2 + 256);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)newCap * newUnitSize,
                     nullptr, GL_DYNAMIC_DRAW);
        if (_accumUploadedCount > 0)
            glBufferSubData(GL_ARRAY_BUFFER, 0, oldBytes, oldData);
        cap = newCap;
    };

    // Grow if needed, then append delta.
    growVBO(_accumVboXYZ, _accumVboCapacityXYZ,
            _accumXYZ.data(), oldXYZBytes, 3 * sizeof(float));
    glBindBuffer(GL_ARRAY_BUFFER, _accumVboXYZ);
    glBufferSubData(GL_ARRAY_BUFFER, oldXYZBytes, dXYZBytes, _deltaXYZ.data());

    growVBO(_accumVboRGB, _accumVboCapacityRGB,
            _accumRGB.data(), oldRGBBytes, 3 * sizeof(uint8_t));
    glBindBuffer(GL_ARRAY_BUFFER, _accumVboRGB);
    glBufferSubData(GL_ARRAY_BUFFER, oldRGBBytes, dRGBBytes, _deltaRGB.data());

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Merge delta into CPU shadow buffers AFTER GPU upload so that the
    // invariant above holds: _accumXYZ/_accumRGB always reflect only what
    // has already been uploaded (_accumUploadedCount points).
    _accumXYZ.insert(_accumXYZ.end(), _deltaXYZ.begin(), _deltaXYZ.end());
    _accumRGB.insert(_accumRGB.end(), _deltaRGB.begin(), _deltaRGB.end());

    _accumUploadedCount = newTotal;

    _deltaXYZ.clear();
    _deltaRGB.clear();
    _deltaCount = 0;
}

void PointCloudWidget::fullUploadAccumVBO()
{
    const int actualFullCount = _accumXYZ.size() / 3;
    _accumUploadedCount  = actualFullCount;
    _accumVboCapacityXYZ = actualFullCount;
    _accumVboCapacityRGB = actualFullCount;

    glBindBuffer(GL_ARRAY_BUFFER, _accumVboXYZ);
    if (actualFullCount > 0) {
        glBufferData(GL_ARRAY_BUFFER,
                     actualFullCount * 3 * sizeof(float),
                     _accumXYZ.data(),
                     GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, _accumVboRGB);
    if (actualFullCount > 0 && _accumRGB.size() >= static_cast<size_t>(actualFullCount * 3)) {
        glBufferData(GL_ARRAY_BUFFER,
                     actualFullCount * 3 * sizeof(uint8_t),
                     _accumRGB.data(),
                     GL_DYNAMIC_DRAW);
    } else {
        glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    }

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    _deltaXYZ.clear();
    _deltaRGB.clear();
    _deltaCount = 0;
}

// ─── Camera matrix ────────────────────────────────────────────────────────────

QMatrix4x4 PointCloudWidget::mvpMatrix() const
{
    QMatrix4x4 view;

    float yawRad   = qDegreesToRadians(_yaw);
    float pitchRad = qDegreesToRadians(_pitch);

    float cx = std::cos(pitchRad) * std::sin(yawRad);
    float cy = std::sin(pitchRad);
    float cz = std::cos(pitchRad) * std::cos(yawRad);

    QVector3D eye = _target + QVector3D(cx, cy, cz) * _distance;
    
    float np = std::fmod(_pitch, 360.0f);
    if (np < 0.0f) np += 360.0f;
    QVector3D up(0.f, -1.f, 0.f);
    if (np > 90.0f && np < 270.0f) {
        up = QVector3D(0.f, 1.f, 0.f);
    }

    view.lookAt(eye, _target, up);
    return view;
}

// ─── Floor grid ───────────────────────────────────────────────────────────────

void PointCloudWidget::drawGrid()
{
    if (!_showGrid) return;

    glLineWidth(1.f);
    glBegin(GL_LINES);
    glColor3f(0.25f, 0.25f, 0.28f);

    const float step  = 100.f;
    const float range = 1500.f;
    const float y     = 300.f;
    const int   steps = static_cast<int>(2.f * range / step) + 1;  // = 31

    for (int i = 0; i < steps; ++i) {
        const float x = -range + i * step;
        glVertex3f(x, y, -range);
        glVertex3f(x, y,  range);
    }
    for (int i = 0; i < steps; ++i) {
        const float z = -range + i * step;
        glVertex3f(-range, y, z);
        glVertex3f( range, y, z);
    }
    glEnd();
}

void PointCloudWidget::drawScanMaskSphere()
{
    if (!_maskEnabled) return;

    glLineWidth(1.2f);
    glColor4f(0.2f, 0.9f, 0.4f, 0.55f);   // xanh lá, bán trong suốt

    const int   stacks = 12, slices = 16;
    const float r = _maskRadius;

    // Vẽ các vòng latitude
    for (int i = 1; i < stacks; ++i) {
        float phi = float(M_PI) * i / stacks;
        float y   = _maskCy + r * std::cos(phi);
        float rr  = r * std::sin(phi);

        glBegin(GL_LINE_LOOP);
        for (int j = 0; j <= slices; ++j) {
            float theta = 2.f * float(M_PI) * j / slices;
            glVertex3f(_maskCx + rr * std::cos(theta),
                       y,
                       _maskCz + rr * std::sin(theta));
        }
        glEnd();
    }

    // Vẽ các đường kinh tuyến
    for (int j = 0; j < slices; ++j) {
        float theta = 2.f * float(M_PI) * j / slices;
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= stacks; ++i) {
            float phi = float(M_PI) * i / stacks;
            glVertex3f(_maskCx + r * std::sin(phi) * std::cos(theta),
                       _maskCy + r * std::cos(phi),
                       _maskCz + r * std::sin(phi) * std::sin(theta));
        }
        glEnd();
    }
}

// ─── Camera trajectory slot & draw ─────────────────────────────────────────────────

void PointCloudWidget::onCameraTrajectoryUpdated(const QVector<QMatrix4x4> &poses)
{
    _camPoses = poses;
    update();
}

void PointCloudWidget::drawCameraTrajectory()
{
    if (_camPoses.isEmpty()) return;

    glLineWidth(2.0f);
    glDisable(GL_DEPTH_TEST);  // luôn hiển thị trên đỉnh point cloud

    for (const QMatrix4x4 &T : _camPoses) {
        if (T.isIdentity()) continue;  // chưa có pose

        // Tính 4 góc của Frustum (Pyramid) trong tọa độ Camera
        // Tính 4 góc của Frustum (Pyramid) trong tọa độ Camera
        // Giả sử khoảng cách z = 80mm
        const float fz = 80.f;
        const float fx = 50.f; // Nửa chiều rộng (w/2)
        const float fy = 35.f; // Nửa chiều cao (h/2)
        
        // 4 đỉnh của đáy (trong hệ Camera: X=Right, Y=Down, Z=Forward)
        // Lưu ý: OpenCV Camera Z = Forward
        QVector3D p0(0, 0, 0); // Đỉnh tháp (optical center)
        QVector3D p1(-fx, -fy, fz); // Top-Left
        QVector3D p2( fx, -fy, fz); // Top-Right
        QVector3D p3( fx,  fy, fz); // Bottom-Right
        QVector3D p4(-fx,  fy, fz); // Bottom-Left

        // Transform 5 điểm này sang Base frame
        QVector3D w0 = T.map(p0);
        QVector3D w1 = T.map(p1);
        QVector3D w2 = T.map(p2);
        QVector3D w3 = T.map(p3);
        QVector3D w4 = T.map(p4);

        // Vẽ Frustum giống Rerun (màu xanh dương nhạt / cyan)
        glColor4f(0.2f, 0.7f, 1.0f, 0.8f);
        glLineWidth(2.0f);
        
        // Vẽ viền ngoài
        glBegin(GL_LINES);
            // Nối từ đỉnh ra 4 góc
            glVertex3f(w0.x(), w0.y(), w0.z()); glVertex3f(w1.x(), w1.y(), w1.z());
            glVertex3f(w0.x(), w0.y(), w0.z()); glVertex3f(w2.x(), w2.y(), w2.z());
            glVertex3f(w0.x(), w0.y(), w0.z()); glVertex3f(w3.x(), w3.y(), w3.z());
            glVertex3f(w0.x(), w0.y(), w0.z()); glVertex3f(w4.x(), w4.y(), w4.z());
            
            // Viền hình chữ nhật đáy
            glVertex3f(w1.x(), w1.y(), w1.z()); glVertex3f(w2.x(), w2.y(), w2.z());
            glVertex3f(w2.x(), w2.y(), w2.z()); glVertex3f(w3.x(), w3.y(), w3.z());
            glVertex3f(w3.x(), w3.y(), w3.z()); glVertex3f(w4.x(), w4.y(), w4.z());
            glVertex3f(w4.x(), w4.y(), w4.z()); glVertex3f(w1.x(), w1.y(), w1.z());
        glEnd();
        
        // Vẽ nhãn thứ tự Camera (dấu chấm đỏ tại gốc)
        glColor4f(1.0f, 0.2f, 0.2f, 0.8f);
        glPointSize(8.0f);
        glBegin(GL_POINTS);
            glVertex3f(w0.x(), w0.y(), w0.z());
        glEnd();
    }

    // Vẽ đường nối giữa các pose (trajectory path)
    if (_camPoses.size() >= 2) {
        glLineWidth(1.0f);
        glColor4f(1.f, 0.85f, 0.2f, 0.7f);  // vàng, bán trong
        glBegin(GL_LINE_STRIP);
        for (const QMatrix4x4 &T : _camPoses) {
            if (!T.isIdentity())
                glVertex3f(T(0,3), T(1,3), T(2,3));
        }
        glEnd();
    }

    glEnable(GL_DEPTH_TEST);  // bật lại depth test
    glLineWidth(1.0f);
    glPointSize(2.0f);
}

// ─── Mouse orbit / pan / zoom ─────────────────────────────────────────────────

void PointCloudWidget::mousePressEvent(QMouseEvent *e)
{
    _lastMouse = e->pos();
    if (e->button() == Qt::LeftButton) {
        _dragging = true;
        _panning  = (e->modifiers() & Qt::ShiftModifier);
    } else if (e->button() == Qt::RightButton) {
        _panning  = true;
        _dragging = false;
    }
}

void PointCloudWidget::mouseMoveEvent(QMouseEvent *e)
{
    if (!_dragging && !_panning) return;

    QPoint delta = e->pos() - _lastMouse;
    _lastMouse   = e->pos();

    if (_panning || (e->buttons() & Qt::RightButton)) {
        float yawRad   = qDegreesToRadians(_yaw);
        float pitchRad = qDegreesToRadians(_pitch);
        QVector3D fwd(std::cos(pitchRad)*std::sin(yawRad),
                      std::sin(pitchRad),
                      std::cos(pitchRad)*std::cos(yawRad));
        
        float np = std::fmod(_pitch, 360.0f);
        if (np < 0.0f) np += 360.0f;
        QVector3D worldUp(0.f, -1.f, 0.f);
        if (np > 90.0f && np < 270.0f) worldUp = QVector3D(0.f, 1.f, 0.f);

        QVector3D right = QVector3D::crossProduct(fwd, worldUp).normalized();
        QVector3D up    = QVector3D::crossProduct(right, fwd).normalized();
        
        float panSpeed = _distance * 0.001f;
        _target -= right * (delta.x() * panSpeed);
        _target += up    * (delta.y() * panSpeed);
    } else {
        _yaw   += delta.x() * 0.5f;
        _pitch += delta.y() * 0.4f;
        
        if (_yaw > 360.f) _yaw -= 360.f;
        else if (_yaw < 0.f) _yaw += 360.f;
        
        if (_pitch > 360.f) _pitch -= 360.f;
        else if (_pitch < 0.f) _pitch += 360.f;
    }
    update();
}

void PointCloudWidget::mouseReleaseEvent(QMouseEvent *)
{
    _dragging = false;
    _panning  = false;
}

void PointCloudWidget::wheelEvent(QWheelEvent *e)
{
    float factor = (e->angleDelta().y() > 0) ? 0.88f : 1.14f;
    _distance = qBound(100.f, _distance * factor, 15000.f);
    update();
}

void PointCloudWidget::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_R)
        resetView();
    else if (e->key() == Qt::Key_C)
        clearAccumulated();
    else if (e->key() == Qt::Key_A)
        autoCenter();
    else
        QOpenGLWidget::keyPressEvent(e);
}