#include "meshviewerwidget.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QPainter>
#include <QFile>
#include <QDataStream>
#include <QTextStream>
#include <QFileInfo>
#include <QDateTime>
#include <QDir>
#include <QtMath>
#include <QRegularExpression>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <queue>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

MeshViewerWidget::MeshViewerWidget(QWidget *parent)
    : QOpenGLWidget(parent)
{
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);

    QSurfaceFormat fmt;
    fmt.setDepthBufferSize(24);
    fmt.setSamples(4);
    fmt.setSwapInterval(1);
    setFormat(fmt);

    _rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
}

MeshViewerWidget::~MeshViewerWidget()
{
    makeCurrent();
    if (_vboXYZ) glDeleteBuffers(1, &_vboXYZ);
    if (_vboNorm) glDeleteBuffers(1, &_vboNorm);
    if (_vboRGB) glDeleteBuffers(1, &_vboRGB);
    if (_ebo)    glDeleteBuffers(1, &_ebo);
    doneCurrent();
}

// ─── OpenGL lifecycle ─────────────────────────────────────────────────────────

void MeshViewerWidget::initializeGL()
{
    initializeOpenGLFunctions();
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    glEnable(GL_NORMALIZE);

    glGenBuffers(1, &_vboXYZ);
    glGenBuffers(1, &_vboNorm);
    glGenBuffers(1, &_vboRGB);
    glGenBuffers(1, &_ebo);
}

void MeshViewerWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);
}

void MeshViewerWidget::paintGL()
{
    // Bắt buộc bật lại Depth Test & Depth Mask ở mỗi frame vì QPainter ở paintEvent() đã tắt nó ở frame trước
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    if (_whiteBg) {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    } else {
        glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (_vx.empty()) {
        QPainter p(this);
        p.setPen(QColor(140, 140, 150));
        p.setFont(QFont("Arial", 13));
        p.drawText(rect(), Qt::AlignCenter,
                   "Chưa có Mesh\nBấm '📦 Xuất Mesh' để tạo, rồi '👁 Xem Mesh'");
        return;
    }

    if (_gpuDirty) {
        uploadToGPU();
        _gpuDirty = false;
    }

    // ── MVP matrices & Camera Headlight ───────────────────────────────────
    QMatrix4x4 proj = projMatrix();
    QMatrix4x4 view = viewMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(proj.constData());

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // Đặt đèn ở Eye Space (trước khi nhân view matrix) để đèn luôn chiếu từ hướng camera
    GLfloat lightPos[] = {0.0f, 0.0f, 1.0f, 0.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

    GLfloat ambient[] = {0.60f, 0.60f, 0.60f, 1.0f};
    GLfloat diffuse[] = {0.55f, 0.55f, 0.55f, 1.0f};
    glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);

    // Sau khi đặt đèn xong mới nạp view matrix của vật thể
    glLoadMatrixf(view.constData());

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, _vboXYZ);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, _vboNorm);
    glNormalPointer(GL_FLOAT, 0, nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, _vboRGB);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, nullptr);

    if (_wireframeMode) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    if (!_faces.empty()) {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
        glDrawElements(GL_TRIANGLES, (GLsizei)_faces.size(), GL_UNSIGNED_INT, nullptr);
    } else {
        glDisable(GL_LIGHTING);
        glPointSize(3.0f);
        glDrawArrays(GL_POINTS, 0, (GLsizei)_vx.size());
        glEnable(GL_LIGHTING);
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void MeshViewerWidget::paintEvent(QPaintEvent *e)
{
    QOpenGLWidget::paintEvent(e);

    QPainter p(this);
    p.setPen(_whiteBg ? QColor(40, 40, 50) : Qt::white);
    p.setFont(QFont("Arial", 11, QFont::Bold));
    if (!_vx.empty()) {
        p.drawText(10, 20, "Mesh Viewer");
    }

    if (_eraserMode) {
        p.setPen(_whiteBg ? QColor(160, 40, 100) : QColor(255, 100, 200));
        p.setFont(QFont("Arial", 10, QFont::Bold));
        p.drawText(QRect(8, 8, 190, 26), Qt::AlignCenter,
                   "✂  Tẩy Thủ Công — Kéo chuột để khoanh");
    }
    p.setPen(_whiteBg ? QColor(100, 100, 100) : QColor(160, 160, 170));
    p.setFont(QFont("Consolas", 9));
    p.drawText(8, height() - 6,
               QString("Tam giác: %1  |  LMB: xoay  |  RMB: pan  |  Scroll: zoom  |  C: center cam")
               .arg(triangleCount()));
}

// ─── GPU upload ───────────────────────────────────────────────────────────────

void MeshViewerWidget::uploadToGPU()
{
    int n = (int)_vx.size();
    if (n == 0) return;

    // Interleave XYZ
    std::vector<float> xyz(n * 3);
    std::vector<float> norm(n * 3);
    for (int i = 0; i < n; ++i) {
        xyz[i*3+0] = _vx[i];
        xyz[i*3+1] = _vy[i];
        xyz[i*3+2] = _vz[i];
        norm[i*3+0] = _vnx.empty() ? 0.0f : _vnx[i];
        norm[i*3+1] = _vnx.empty() ? 0.0f : _vny[i];
        norm[i*3+2] = _vnx.empty() ? 1.0f : _vnz[i];
    }
    glBindBuffer(GL_ARRAY_BUFFER, _vboXYZ);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(xyz.size() * sizeof(float)),
                 xyz.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, _vboNorm);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(norm.size() * sizeof(float)),
                 norm.data(), GL_STATIC_DRAW);

    // RGB
    std::vector<uint8_t> rgb(n * 3);
    for (int i = 0; i < n; ++i) {
        rgb[i*3+0] = _vr[i];
        rgb[i*3+1] = _vg[i];
        rgb[i*3+2] = _vb[i];
    }
    glBindBuffer(GL_ARRAY_BUFFER, _vboRGB);
    glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(rgb.size() * sizeof(uint8_t)),
                 rgb.data(), GL_STATIC_DRAW);

    // Index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(_faces.size() * sizeof(uint32_t)),
                 _faces.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// ─── Camera matrices ──────────────────────────────────────────────────────────

QMatrix4x4 MeshViewerWidget::viewMatrix() const
{
    float yawRad   = qDegreesToRadians(_yaw);
    float pitchRad = qDegreesToRadians(_pitch);

    float cx = std::cos(pitchRad) * std::sin(yawRad);
    float cy = std::sin(pitchRad);
    float cz = std::cos(pitchRad) * std::cos(yawRad);

    QVector3D eye = _target + QVector3D(cx, cy, cz) * _distance;
    
    float np = std::fmod(_pitch, 360.0f);
    if (np < 0.0f) np += 360.0f;
    QVector3D up(0.f, 1.f, 0.f);
    if (np > 90.0f && np < 270.0f) {
        up = QVector3D(0.f, -1.f, 0.f);
    }

    QMatrix4x4 v;
    v.lookAt(eye, _target, up);
    return v;
}

QMatrix4x4 MeshViewerWidget::projMatrix() const
{
    QMatrix4x4 p;
    float aspect = (height() > 0) ? float(width()) / float(height()) : 1.f;
    p.perspective(45.f, aspect, std::max(0.005f, _distance * 0.05f), _distance * 10.f);
    return p;
}

// ─── Mouse interaction ────────────────────────────────────────────────────────

void MeshViewerWidget::mousePressEvent(QMouseEvent *e)
{
    _lastMouse = e->pos();

    if (_eraserMode && e->button() == Qt::LeftButton) {
        _selecting = true;
        _selStart  = e->pos();
        _rubberBand->setGeometry(QRect(_selStart, QSize()));
        _rubberBand->show();
        return;
    }

    if (e->button() == Qt::LeftButton)
        _rotating = true;
    if (e->button() == Qt::RightButton)
        _panning = true;
}

void MeshViewerWidget::mouseMoveEvent(QMouseEvent *e)
{
    if (_eraserMode && _selecting) {
        _rubberBand->setGeometry(QRect(_selStart, e->pos()).normalized());
        return;
    }

    QPoint delta = e->pos() - _lastMouse;
    _lastMouse   = e->pos();

    if (_rotating) {
        _yaw   += delta.x() * 0.5f;
        _pitch += delta.y() * 0.5f;
        
        if (_yaw > 360.f) _yaw -= 360.f;
        else if (_yaw < 0.f) _yaw += 360.f;
        
        if (_pitch > 360.f) _pitch -= 360.f;
        else if (_pitch < 0.f) _pitch += 360.f;
        
        update();
    }
    if (_panning) {
        float yawRad   = qDegreesToRadians(_yaw);
        float pitchRad = qDegreesToRadians(_pitch);
        QVector3D fwd(std::cos(pitchRad)*std::sin(yawRad),
                      std::sin(pitchRad),
                      std::cos(pitchRad)*std::cos(yawRad));
        
        float np = std::fmod(_pitch, 360.0f);
        if (np < 0.0f) np += 360.0f;
        QVector3D worldUp(0.f, 1.f, 0.f);
        if (np > 90.0f && np < 270.0f) worldUp = QVector3D(0.f, -1.f, 0.f);

        QVector3D right = QVector3D::crossProduct(fwd, worldUp).normalized();
        QVector3D up    = QVector3D::crossProduct(right, fwd).normalized();
        float panScale  = _distance * 0.001f;
        _target -= right * delta.x() * panScale;
        _target += up    * delta.y() * panScale;
        update();
    }
}

void MeshViewerWidget::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_C) {
        computeCenter();
        update();
    }
}

void MeshViewerWidget::mouseReleaseEvent(QMouseEvent *e)
{
    if (_eraserMode && _selecting && e->button() == Qt::LeftButton) {
        _selecting = false;
        _rubberBand->hide();
        QRect sel = QRect(_selStart, e->pos()).normalized();
        if (sel.width() > 3 && sel.height() > 3) {
            eraseInRect(sel);
        }
        return;
    }

    if (e->button() == Qt::LeftButton)  _rotating = false;
    if (e->button() == Qt::RightButton) _panning  = false;
}

void MeshViewerWidget::wheelEvent(QWheelEvent *e)
{
    float factor = (e->angleDelta().y() > 0) ? 0.85f : 1.18f;
    _distance = qBound(0.001f, _distance * factor, 1000.f);
    update();
}

// ─── Eraser ───────────────────────────────────────────────────────────────────

void MeshViewerWidget::eraseInRect(const QRect &screenRect)
{
    if (_faces.empty()) return;

    // Build MVP
    QMatrix4x4 mvp = projMatrix() * viewMatrix();
    int W = width(), H = height();

    int n = (int)_vx.size();
    // Project each vertex to screen space
    std::vector<bool> inBox(n, false);
    for (int i = 0; i < n; ++i) {
        QVector4D clip = mvp * QVector4D(_vx[i], _vy[i], _vz[i], 1.0f);
        if (clip.w() <= 0.0f) continue;
        float ndcX = clip.x() / clip.w();
        float ndcY = clip.y() / clip.w();
        int sx = (int)((ndcX + 1.0f) * 0.5f * W);
        int sy = (int)((1.0f - ndcY) * 0.5f * H); // flip Y
        inBox[i] = screenRect.contains(sx, sy);
    }

    // Collect faces where ANY vertex is in the box
    std::vector<uint32_t> newFaces;
    newFaces.reserve(_faces.size());
    int removed = 0;
    for (size_t t = 0; t < _faces.size(); t += 3) {
        uint32_t a = _faces[t], b = _faces[t+1], c = _faces[t+2];
        if (inBox[a] || inBox[b] || inBox[c]) {
            ++removed;
        } else {
            newFaces.push_back(a);
            newFaces.push_back(b);
            newFaces.push_back(c);
        }
    }

    if (removed == 0) {
        emit statusMessage("Không có tam giác nào trong vùng chọn!");
        return;
    }

    pushHistory();
    _faces    = std::move(newFaces);
    _gpuDirty = true;
    update();

    emit meshChanged(triangleCount());
    emit statusMessage(QString("Đã xóa %1 tam giác. Còn %2 tam giác.")
                       .arg(removed).arg(triangleCount()));
}

// ─── History ──────────────────────────────────────────────────────────────────

void MeshViewerWidget::pushHistory()
{
    if ((int)_history.size() >= MAX_HISTORY)
        _history.erase(_history.begin());
    _history.push_back(_faces);
}

// ─── Public API ───────────────────────────────────────────────────────────────

bool MeshViewerWidget::undo()
{
    if (_history.empty()) {
        emit statusMessage("Không có lịch sử để Undo!");
        return false;
    }
    _faces    = _history.back();
    _history.pop_back();
    _gpuDirty = true;
    update();
    emit meshChanged(triangleCount());
    emit statusMessage(QString("Đã Undo. Còn %1 tam giác.").arg(triangleCount()));
    return true;
}

bool MeshViewerWidget::autoClean()
{
    if (_faces.empty()) {
        emit statusMessage("Chưa có mesh để làm sạch!");
        return false;
    }

    int nVerts = (int)_vx.size();

    // Build adjacency list
    std::vector<std::vector<int>> adj(nVerts);
    for (size_t t = 0; t < _faces.size(); t += 3) {
        int a = (int)_faces[t], b = (int)_faces[t+1], c = (int)_faces[t+2];
        adj[a].push_back(b); adj[a].push_back(c);
        adj[b].push_back(a); adj[b].push_back(c);
        adj[c].push_back(a); adj[c].push_back(b);
    }

    // BFS/DFS to find connected components
    std::vector<int> comp(nVerts, -1);
    int numComp = 0;
    std::vector<int> compSize;

    for (int start = 0; start < nVerts; ++start) {
        if (comp[start] != -1) continue;
        // check if used
        // BFS
        std::queue<int> q;
        q.push(start);
        comp[start] = numComp;
        int sz = 0;
        while (!q.empty()) {
            int v = q.front(); q.pop(); ++sz;
            for (int nb : adj[v]) {
                if (comp[nb] == -1) {
                    comp[nb] = numComp;
                    q.push(nb);
                }
            }
        }
        compSize.push_back(sz);
        ++numComp;
    }

    if (numComp <= 1) {
        emit statusMessage("Mesh đã là 1 khối duy nhất!");
        return false;
    }

    // Find largest component
    int largestComp = (int)(std::max_element(compSize.begin(), compSize.end()) - compSize.begin());

    // Keep only faces where ALL vertices are in the largest component
    std::vector<uint32_t> newFaces;
    newFaces.reserve(_faces.size());
    int removed = 0;
    for (size_t t = 0; t < _faces.size(); t += 3) {
        int a = (int)_faces[t], b = (int)_faces[t+1], c = (int)_faces[t+2];
        if (comp[a] == largestComp && comp[b] == largestComp && comp[c] == largestComp) {
            newFaces.push_back(_faces[t]);
            newFaces.push_back(_faces[t+1]);
            newFaces.push_back(_faces[t+2]);
        } else {
            ++removed;
        }
    }

    if (removed == 0) {
        emit statusMessage("Không có rác nào để dọn!");
        return false;
    }

    pushHistory();
    _faces    = std::move(newFaces);
    _gpuDirty = true;
    computeCenter(); // Recenter the camera on the remaining largest component!
    update();
    emit meshChanged(triangleCount());
    emit statusMessage(QString("Auto Clean: đã xóa %1 tam giác rác. Còn %2 tam giác.")
                       .arg(removed).arg(triangleCount()));
    return true;
}

void MeshViewerWidget::setEraserMode(bool on)
{
    _eraserMode = on;
    _selecting  = false;
    if (_rubberBand) _rubberBand->hide();
    setCursor(on ? Qt::CrossCursor : Qt::ArrowCursor);
    update();
}

void MeshViewerWidget::setWhiteBackground(bool w)
{
    if (_whiteBg == w) return;
    _whiteBg = w;
    update();
}

void MeshViewerWidget::setWireframeMode(bool on)
{
    if (_wireframeMode == on) return;
    _wireframeMode = on;
    update();
}

void MeshViewerWidget::autoCenter()
{
    computeCenter();
}

void MeshViewerWidget::setViewPreset(int preset)
{
    switch (preset) {
    case 0: // Reset / Default
        _yaw = 0.0f;
        _pitch = -30.0f;
        computeCenter();
        break;
    case 1: // Top
        _yaw = 0.0f;
        _pitch = 89.9f;
        break;
    case 2: // Front
        _yaw = 0.0f;
        _pitch = 0.0f;
        break;
    case 3: // Side / Right
        _yaw = 90.0f;
        _pitch = 0.0f;
        break;
    case 4: // Iso
        _yaw = 45.0f;
        _pitch = -35.26f;
        break;
    default:
        break;
    }
    update();
}

bool MeshViewerWidget::captureScreenshot(const QString &savePath)
{
    QImage img = grabFramebuffer();
    QString path = savePath;
    if (path.isEmpty()) {
        QString dir = QDir::currentPath() + "/screenshots";
        QDir().mkpath(dir);
        path = dir + "/mesh_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
    }
    bool ok = img.save(path);
    if (ok) {
        emit statusMessage("Đã lưu ảnh chụp Mesh: " + QFileInfo(path).fileName());
    }
    return ok;
}

void MeshViewerWidget::computeCenter()
{
    if (_faces.empty() || _vx.empty()) return;
    
    std::vector<bool> active(_vx.size(), false);
    for (uint32_t idx : _faces) {
        if (idx < _vx.size()) active[idx] = true;
    }

    double cx = 0, cy = 0, cz = 0;
    int n = 0;
    float xmin = 1e9, xmax = -1e9;
    float ymin = 1e9, ymax = -1e9;
    float zmin = 1e9, zmax = -1e9;
    
    for (size_t i = 0; i < _vx.size(); ++i) {
        if (active[i]) {
            cx += _vx[i];
            cy += _vy[i];
            cz += _vz[i];
            n++;
            if (_vx[i] < xmin) xmin = _vx[i];
            if (_vx[i] > xmax) xmax = _vx[i];
            if (_vy[i] < ymin) ymin = _vy[i];
            if (_vy[i] > ymax) ymax = _vy[i];
            if (_vz[i] < zmin) zmin = _vz[i];
            if (_vz[i] > zmax) zmax = _vz[i];
        }
    }

    if (n > 0) {
        _target = QVector3D(cx/n, cy/n, cz/n);
        float diag = std::sqrt((xmax-xmin)*(xmax-xmin) +
                               (ymax-ymin)*(ymax-ymin) +
                               (zmax-zmin)*(zmax-zmin));
        if (diag < 0.01f) diag = 0.01f;
        _distance = diag * 1.5f;
    }
}

// ─── OBJ Loader ───────────────────────────────────────────────────────────────

bool MeshViewerWidget::loadObjMesh(const QString &path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        emit statusMessage("Không thể mở file OBJ: " + path);
        return false;
    }

    _vx.clear(); _vy.clear(); _vz.clear();
    _vnx.clear(); _vny.clear(); _vnz.clear();
    _vr.clear(); _vg.clear(); _vb.clear();
    _faces.clear();

    QTextStream in(&f);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith("#")) continue;

        QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.empty()) continue;

        if (parts[0] == "v" && parts.size() >= 4) {
            _vx.push_back(parts[1].toFloat());
            _vy.push_back(parts[2].toFloat());
            _vz.push_back(parts[3].toFloat());
            if (parts.size() >= 7) { // v x y z r g b
                _vr.push_back(parts[4].toFloat() * 255.0f);
                _vg.push_back(parts[5].toFloat() * 255.0f);
                _vb.push_back(parts[6].toFloat() * 255.0f);
            } else {
                _vr.push_back(160);
                _vg.push_back(180);
                _vb.push_back(200);
            }
        } else if (parts[0] == "f" && parts.size() >= 4) {
            std::vector<int> faceVerts;
            for (int i = 1; i < parts.size(); ++i) {
                QStringList vParts = parts[i].split('/');
                int vIdx = vParts[0].toInt();
                if (vIdx < 0) vIdx = _vx.size() + vIdx + 1;
                faceVerts.push_back(vIdx - 1);
            }
            if (faceVerts.size() >= 3) {
                // simple fan triangulation for n-gons
                for (size_t i = 1; i < faceVerts.size() - 1; ++i) {
                    _faces.push_back(faceVerts[0]);
                    _faces.push_back(faceVerts[i]);
                    _faces.push_back(faceVerts[i+1]);
                }
            }
        }
    }
    
    // OBJ load doesn't easily map normals per-vertex when indexed, 
    // we'll just let computeCenter / lighting handle recomputing flat normals later
    _vnx.clear(); _vny.clear(); _vnz.clear();
    
    _history.clear();
    _loadedPath = path;
    _gpuDirty   = true;
    computeCenter();
    _gpuDirty = true;
    update();
    emit meshChanged(triangleCount());
    emit statusMessage(QString("Đã tải OBJ: %1 đỉnh, %2 tam giác").arg(_vx.size()).arg(_faces.size()/3));
    return true;
}

// ─── PLY Loader ───────────────────────────────────────────────────────────────

bool MeshViewerWidget::loadMesh(const QString &path)
{
    if (path.endsWith(".obj", Qt::CaseInsensitive)) {
        return loadObjMesh(path);
    }

    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) {
        emit statusMessage("Không thể mở file: " + path);
        return false;
    }

    // Parse header
    bool isBinary = false;
    bool isLittleEndian = true;
    int nVerts = 0, nFaces = 0;
    int propX=-1,propY=-1,propZ=-1,propNX=-1,propNY=-1,propNZ=-1,propR=-1,propG=-1,propB=-1;
    int propX_off=-1,propY_off=-1,propZ_off=-1;
    int propNX_off=-1,propNY_off=-1,propNZ_off=-1;
    int propR_off=-1,propG_off=-1,propB_off=-1;
    // Per-property byte sizes (4=float, 8=double, 1=uchar)
    int propX_sz=4, propY_sz=4, propZ_sz=4;
    int propNX_sz=4, propNY_sz=4, propNZ_sz=4;
    int faceIdxSz = 4; // 4=int/uint, 2=short
    bool inVertex = false, inFace = false;
    int propCount = 0;
    int vertexStride = 0;
    bool hasColors = false, hasNormals = false;

    // Header is always ASCII
    QString line;
    while (true) {
        QByteArray raw = f.readLine();
        line = QString(raw).trimmed();
        if (line == "end_header") break;
        if (line.startsWith("format binary_little_endian")) { isBinary = true; isLittleEndian = true; }
        else if (line.startsWith("format binary_big_endian"))    { isBinary = true; isLittleEndian = false; }
        else if (line.startsWith("element vertex")) { nVerts = line.split(' ').last().toInt(); inVertex=true; inFace=false; propCount=0; }
        else if (line.startsWith("element face"))   { nFaces = line.split(' ').last().toInt(); inFace=true; inVertex=false; }
        else if (line.startsWith("property") && inVertex) {
            QStringList t = line.split(' ', Qt::SkipEmptyParts);
            if (t.size() >= 3) {
                QString type = t[1];
                QString name = t.last();

                int sz = 4;
                if (type == "double" || type == "float64") sz = 8;
                else if (type == "uchar" || type == "char" || type == "uint8" || type == "int8") sz = 1;
                else if (type == "short" || type == "ushort" || type == "int16" || type == "uint16") sz = 2;

                if      (name=="x")                 { propX=propCount;  propX_off=vertexStride;  propX_sz=sz; }
                else if (name=="y")                 { propY=propCount;  propY_off=vertexStride;  propY_sz=sz; }
                else if (name=="z")                 { propZ=propCount;  propZ_off=vertexStride;  propZ_sz=sz; }
                else if (name=="nx")                { propNX=propCount; propNX_off=vertexStride; propNX_sz=sz; hasNormals=true; }
                else if (name=="ny")                { propNY=propCount; propNY_off=vertexStride; propNY_sz=sz; hasNormals=true; }
                else if (name=="nz")                { propNZ=propCount; propNZ_off=vertexStride; propNZ_sz=sz; hasNormals=true; }
                else if (name=="red"||name=="r")   { propR=propCount;  propR_off=vertexStride;                hasColors=true; }
                else if (name=="green"||name=="g") { propG=propCount;  propG_off=vertexStride;                hasColors=true; }
                else if (name=="blue"||name=="b")  { propB=propCount;  propB_off=vertexStride;                hasColors=true; }

                vertexStride += sz;
                ++propCount;
            }
        }
        // Track face index width from: "property list uchar int/uint vertex_indices"
        else if (line.startsWith("property list") && inFace) {
            QStringList t = line.split(' ', Qt::SkipEmptyParts);
            // t[3] is index type: int(4) uint(4) short(2) ...
            if (t.size() >= 4) {
                QString idxType = t[3]; // int, uint, int32, uint32 etc.
                if (idxType == "short" || idxType == "ushort" || idxType == "int16" || idxType == "uint16")
                    faceIdxSz = 2;
                else
                    faceIdxSz = 4;
            }
        }
    }

    if (nVerts == 0 || propX < 0 || propY < 0 || propZ < 0) {
        emit statusMessage("File PLY không hợp lệ: " + path);
        return false;
    }

    _vx.resize(nVerts); _vy.resize(nVerts); _vz.resize(nVerts);
    if (hasNormals) { _vnx.resize(nVerts); _vny.resize(nVerts); _vnz.resize(nVerts); }
    else { _vnx.clear(); _vny.clear(); _vnz.clear(); }
    
    _vr.assign(nVerts, 160); _vg.assign(nVerts, 180); _vb.assign(nVerts, 200);

    if (isBinary) {
        if (vertexStride == 0) vertexStride = 12 + (hasColors ? 3 : 0) + (hasNormals ? 12 : 0); // fallback

        // IMPORTANT: QFile::readLine() uses an internal buffer that may consume
        // bytes beyond the end_header line.  We must use seek() to the correct
        // binary offset that we computed by summing the raw bytes of each header line.
        QByteArray raw = f.readAll();   // reads from current QFile position
        // Verify we have enough data
        qint64 expectedVerts = (qint64)nVerts * vertexStride;
        if (raw.size() < expectedVerts) {
            // QFile internal buffer ate some binary bytes – fall back to re-opening
            f.close();
            f.open(QIODevice::ReadOnly);
            // Skip header by counting raw bytes
            qint64 headerEnd = 0;
            QByteArray all = f.readAll();
            QByteArray needle = QByteArray("end_header\n");
            int idx = all.indexOf(needle);
            if (idx < 0) {
                emit statusMessage("PLY header error");
                return false;
            }
            raw = all.mid(idx + needle.size());
        }
        const char *ptr = raw.constData();
        const char *end = ptr + raw.size();

        for (int i = 0; i < nVerts && ptr + vertexStride <= end; ++i) {
            // Read XYZ — support both float(4) and double(8)
            auto readScalar = [&](int off, int sz) -> float {
                if (sz == 8 && off + 8 <= vertexStride) {
                    double v; memcpy(&v, ptr + off, 8); return (float)v;
                } else if (sz >= 4 && off + 4 <= vertexStride) {
                    float v;  memcpy(&v, ptr + off, 4); return v;
                }
                return 0.f;
            };
            _vx[i] = readScalar(propX_off,  propX_sz);
            _vy[i] = readScalar(propY_off,  propY_sz);
            _vz[i] = readScalar(propZ_off,  propZ_sz);

            if (hasNormals) {
                _vnx[i] = readScalar(propNX_off, propNX_sz);
                _vny[i] = readScalar(propNY_off, propNY_sz);
                _vnz[i] = readScalar(propNZ_off, propNZ_sz);
            }

            if (hasColors) {
                if (propR_off >= 0 && propR_off < vertexStride) _vr[i] = (uint8_t)*(ptr + propR_off);
                if (propG_off >= 0 && propG_off < vertexStride) _vg[i] = (uint8_t)*(ptr + propG_off);
                if (propB_off >= 0 && propB_off < vertexStride) _vb[i] = (uint8_t)*(ptr + propB_off);
            }
            ptr += vertexStride;
        }

        _faces.clear();
        _faces.reserve(nFaces * 3);
        for (int i = 0; i < nFaces && ptr + 1 <= end; ++i) {
            uint8_t cnt = (uint8_t)*ptr; ptr++;
            int needed = cnt * faceIdxSz;
            if (cnt == 3 && ptr + needed <= end) {
                uint32_t a = 0, b = 0, c = 0;
                if (faceIdxSz == 4) {
                    memcpy(&a, ptr,            4);
                    memcpy(&b, ptr+4,          4);
                    memcpy(&c, ptr+8,          4);
                } else { // 2-byte short indices
                    uint16_t sa,sb,sc;
                    memcpy(&sa, ptr,  2); memcpy(&sb, ptr+2, 2); memcpy(&sc, ptr+4, 2);
                    a=(uint32_t)sa; b=(uint32_t)sb; c=(uint32_t)sc;
                }
                _faces.push_back(a);
                _faces.push_back(b);
                _faces.push_back(c);
                ptr += needed;
            } else {
                ptr += needed;
            }
        }
    } else {
        // ASCII
        for (int i = 0; i < nVerts; ++i) {
            QByteArray raw2 = f.readLine();
            QStringList tok = QString(raw2).trimmed().split(' ', Qt::SkipEmptyParts);
            if (propX < tok.size()) _vx[i] = tok[propX].toFloat();
            if (propY < tok.size()) _vy[i] = tok[propY].toFloat();
            if (propZ < tok.size()) _vz[i] = tok[propZ].toFloat();
            if (hasColors) {
                if (propR < tok.size()) _vr[i] = (uint8_t)tok[propR].toInt();
                if (propG < tok.size()) _vg[i] = (uint8_t)tok[propG].toInt();
                if (propB < tok.size()) _vb[i] = (uint8_t)tok[propB].toInt();
            }
        }
        _faces.clear();
        _faces.reserve(nFaces * 3);
        for (int i = 0; i < nFaces; ++i) {
            QByteArray raw2 = f.readLine();
            QStringList tok = QString(raw2).trimmed().split(' ', Qt::SkipEmptyParts);
            if (tok.size() >= 4 && tok[0].toInt() == 3) {
                _faces.push_back(tok[1].toUInt());
                _faces.push_back(tok[2].toUInt());
                _faces.push_back(tok[3].toUInt());
            }
        }
    }
    _history.clear();
    _loadedPath = path;
    _gpuDirty   = true;
    computeCenter();
    _gpuDirty = true;
    update();
    emit meshChanged(triangleCount());
    emit statusMessage(QString("Đã tải: %1").arg(QFileInfo(path).fileName()));
    return true;
}

// ─── PLY Saver ───────────────────────────────────────────────────────────────

bool MeshViewerWidget::saveMesh(const QString &path)
{
    if (_vx.empty()) {
        emit statusMessage("Không có dữ liệu để lưu!");
        return false;
    }

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) {
        emit statusMessage("Không thể lưu file: " + path);
        return false;
    }

    int nVerts = (int)_vx.size();
    int nFaces = (int)(_faces.size() / 3);

    // Write header
    QTextStream hs(&f);
    hs << "ply\n";
    hs << "format binary_little_endian 1.0\n";
    hs << "element vertex " << nVerts << "\n";
    hs << "property float x\nproperty float y\nproperty float z\n";
    hs << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    hs << "element face " << nFaces << "\n";
    hs << "property list uchar int vertex_index\n";
    hs << "end_header\n";
    hs.flush();

    // Write binary vertex data
    for (int i = 0; i < nVerts; ++i) {
        f.write(reinterpret_cast<const char*>(&_vx[i]), 4);
        f.write(reinterpret_cast<const char*>(&_vy[i]), 4);
        f.write(reinterpret_cast<const char*>(&_vz[i]), 4);
        f.write(reinterpret_cast<const char*>(&_vr[i]), 1);
        f.write(reinterpret_cast<const char*>(&_vg[i]), 1);
        f.write(reinterpret_cast<const char*>(&_vb[i]), 1);
    }

    // Write binary face data
    const uint8_t three = 3;
    for (size_t t = 0; t < _faces.size(); t += 3) {
        f.write(reinterpret_cast<const char*>(&three), 1);
        int32_t a = (int32_t)_faces[t];
        int32_t b = (int32_t)_faces[t+1];
        int32_t c = (int32_t)_faces[t+2];
        f.write(reinterpret_cast<const char*>(&a), 4);
        f.write(reinterpret_cast<const char*>(&b), 4);
        f.write(reinterpret_cast<const char*>(&c), 4);
    }

    emit statusMessage("Đã lưu: " + path);
    return true;
}
