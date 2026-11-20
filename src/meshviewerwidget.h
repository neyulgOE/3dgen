#ifndef MESHVIEWERWIDGET_H
#define MESHVIEWERWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMatrix4x4>
#include <QVector3D>
#include <QPoint>
#include <QRubberBand>

#include <vector>

class MeshViewerWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit MeshViewerWidget(QWidget *parent = nullptr);
    ~MeshViewerWidget() override;

    bool loadMesh(const QString &path);
    bool loadObjMesh(const QString &path);
    bool autoClean();
    bool undo();
    bool saveMesh(const QString &path);

    void setWhiteBackground(bool w);


    void setWireframeMode(bool on);
    bool wireframeMode() const { return _wireframeMode; }

    void setViewPreset(int preset); // 0=Reset, 1=Top, 2=Front, 3=Side, 4=Iso
    void autoCenter();
    bool captureScreenshot(const QString &savePath = QString());

    void setEraserMode(bool on);
    bool eraserMode()    const { return _eraserMode; }
    int  triangleCount() const { return static_cast<int>(_faces.size() / 3); }
    int  vertexCount()   const { return static_cast<int>(_vx.size()); }
    bool hasMesh()       const { return !_faces.empty(); }
    QString loadedPath() const { return _loadedPath; }

signals:
    void meshChanged(int triangleCount);
    void statusMessage(const QString &msg);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void paintEvent(QPaintEvent *e) override;
    void keyPressEvent(QKeyEvent *e) override;
    void mousePressEvent  (QMouseEvent  *e) override;
    void mouseMoveEvent   (QMouseEvent  *e) override;
    void mouseReleaseEvent(QMouseEvent  *e) override;
    void wheelEvent       (QWheelEvent  *e) override;

private:
    // Vertex data
    std::vector<float> _vx, _vy, _vz;
    std::vector<float> _vnx, _vny, _vnz;
    std::vector<uint8_t> _vr, _vg, _vb;
    // Face data (3 indices per triangle, flat)
    std::vector<uint32_t> _faces;

    // Undo history
    static constexpr int MAX_HISTORY = 20;
    std::vector<std::vector<uint32_t>> _history;

    // Camera
    float     _yaw      = 0.0f;
    float     _pitch    = -30.0f;
    float     _distance = 0.5f;
    QVector3D _target;
    QPoint    _lastMouse;
    bool      _rotating = false;
    bool      _panning  = false;

    // Eraser
    bool         _eraserMode = false;
    bool         _selecting  = false;
    QPoint       _selStart;
    QRubberBand *_rubberBand = nullptr;

    bool      _whiteBg = false;
    bool      _wireframeMode = false;

    // OpenGL
    GLuint _vboXYZ   = 0;
    GLuint _vboNorm  = 0;
    GLuint _vboRGB   = 0;
    GLuint _ebo      = 0;
    bool   _gpuDirty = true;

    QString _loadedPath;

    void uploadToGPU();
    void eraseInRect(const QRect &screenRect);
    void pushHistory();
    void computeCenter();
    QMatrix4x4 viewMatrix() const;
    QMatrix4x4 projMatrix() const;
};

#endif // MESHVIEWERWIDGET_H
