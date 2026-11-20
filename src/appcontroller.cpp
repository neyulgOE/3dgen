#include "appcontroller.h"
#include "cameraworker.h"
#include "plyexporter.h"
#include "realsenseworker.h"
#include "replayworker.h"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QTextStream>
#include <QThread>
#include <QTimer>

#include <algorithm>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

AppController::AppController(QObject *parent)
    : QObject(parent)
    , _workerThread(new QThread(this))
    , _camera(nullptr)
    , _replayWorker(new ReplayWorker)
    , _recorder(new DatasetRecorder(this))
    , _scanSession(this)
{
    _workerThread->start();
    _replayWorker->moveToThread(_workerThread);
    connect(_workerThread, &QThread::finished, _replayWorker, &QObject::deleteLater);

    // Kết nối DatasetRecorder 1 lần duy nhất (Qt::UniqueConnection)
    connect(_recorder, &DatasetRecorder::sig_log,
            this, &AppController::logMessage, Qt::UniqueConnection);
    connect(_recorder, &DatasetRecorder::sig_recordStats,
            this, &AppController::recordStatsUpdated, Qt::UniqueConnection);
    connect(_recorder, &DatasetRecorder::sig_recordFinished,
            this, [this](const QString &dir, int count) {
                _lastRecordedDir = dir;
                _currentScanDir = dir;
                emit lastDatasetChanged(dir);
                emit logMessage(QString("📁 Dataset mới sẵn sàng cho Mesh Studio: %1 (%2 frames)")
                                    .arg(dir).arg(count));
            }, Qt::UniqueConnection);
    connect(_recorder, &DatasetRecorder::sig_recordError,
            this, &AppController::workerError, Qt::UniqueConnection);

    // Kết nối ReplayWorker finished 1 lần duy nhất
    connect(_replayWorker, &ReplayWorker::fetchCountFinished,
            this, &AppController::onReplayFinished, Qt::UniqueConnection);

    setCameraSource(1); // Mặc định Intel RealSense
    wireScan();
}

AppController::~AppController()
{
    if (_recorder && _recorder->isRecording()) {
        _recorder->stop();
    }
    _scanSession.stop();
    emit sig_disconnectCamera();

    _workerThread->quit();
    _workerThread->wait(3000);
}

// ─── Wiring ───────────────────────────────────────────────────────────────────

void AppController::wireCamera()
{
    if (!_camera) return;

    connect(this, &AppController::sig_connectCamera,    _camera, &ICameraSource::connectHost,     Qt::UniqueConnection);
    connect(this, &AppController::sig_disconnectCamera, _camera, &ICameraSource::disconnectHost,  Qt::UniqueConnection);
    connect(this, &AppController::sig_fetchCount,       _camera, &ICameraSource::fetchCount,      Qt::UniqueConnection);
    connect(this, &AppController::sig_fetchContinuous,  _camera, &ICameraSource::fetchContinuous, Qt::UniqueConnection);
    connect(this, &AppController::sig_stopFetch,        _camera, &ICameraSource::stopFetch,       Qt::UniqueConnection);

    connect(_camera, &ICameraSource::connectionChanged,
            this, &AppController::onCameraConnectionChanged, Qt::UniqueConnection);
    connect(_camera, &ICameraSource::intrinsicsReady,
            this, &AppController::onCameraIntrinsicsReady, Qt::UniqueConnection);
    connect(_camera, &ICameraSource::rgbdFrameReady,
            this, &AppController::onCameraRGBDReady, Qt::UniqueConnection);
    connect(_camera, &ICameraSource::errorOccurred,
            this, &AppController::onCameraError, Qt::UniqueConnection);
    connect(_camera, &ICameraSource::imuDataReady,
            this, &AppController::onImuDataReady, Qt::UniqueConnection);
    connect(_camera, &ICameraSource::logMessage,
            this, &AppController::logMessage, Qt::UniqueConnection);
}

void AppController::wireScan()
{
    connect(&_scanSession, &ScanSession::requestMove,
            this, &AppController::onScanRequestMove);
    connect(&_scanSession, &ScanSession::requestSnapshot,
            this, &AppController::onScanRequestSnapshot);
    connect(&_scanSession, &ScanSession::requestFetch,
            this, &AppController::onScanRequestFetch);
    connect(&_scanSession, &ScanSession::captureStarted,
            this, &AppController::onScanCaptureStarted);
    connect(&_scanSession, &ScanSession::finished,
            this, &AppController::onScanFinished);
    connect(&_scanSession, &ScanSession::progressUpdated,
            this, &AppController::scanProgressUpdated);
    connect(&_scanSession, &ScanSession::logMessage,
            this, &AppController::logMessage);
}

// ─── Camera public slots ──────────────────────────────────────────────────────

void AppController::setCameraSource(int source)
{
    if (_camera) {
        if (_camera != _replayWorker) {
            emit sig_disconnectCamera();
            _camera->disconnect(this);
            this->disconnect(_camera);
            disconnect(_workerThread, &QThread::finished, _camera, &QObject::deleteLater);
            _camera->deleteLater();
        } else {
            _camera->disconnect(this);
            this->disconnect(_camera);
        }
    }

    if (source == 0) {
        _camera = new CameraWorker;
    } else {
        _camera = new RealSenseWorker;
    }

    _camera->moveToThread(_workerThread);
    connect(_workerThread, &QThread::finished, _camera, &QObject::deleteLater);
    QMetaObject::invokeMethod(_camera, &ICameraSource::init, Qt::QueuedConnection);
    wireCamera();
}

void AppController::connectCamera(const QString &ip, quint16 port)
{
    if (_state == AppState::Recording) {
        emit workerError("Đang Record, không thể đổi kết nối camera.");
        return;
    }
    emit logMessage(QString("Connecting camera to %1:%2 …").arg(ip).arg(port));
    emit sig_connectCamera(ip, port);
}

void AppController::disconnectCamera()
{
    if (_state == AppState::Recording) {
        stopRecording();
    }
    emit sig_disconnectCamera();
}

void AppController::setIntrinsics(const CameraIntrinsics &intr)
{
    _intrinsics = intr;
    _currentIntrinsics.width = intr.rgbW > 0 ? intr.rgbW : 640;
    _currentIntrinsics.height = intr.rgbH > 0 ? intr.rgbH : 480;
    _currentIntrinsics.fx = intr.fx;
    _currentIntrinsics.fy = intr.fy;
    _currentIntrinsics.cx = intr.cx;
    _currentIntrinsics.cy = intr.cy;
    _currentIntrinsics.depthScale = 1000.0;

    _projector.setIntrinsics(intr);
    emit sig_setIntrinsics(intr);
}

// ─── Stop-and-scan ────────────────────────────────────────────────────────────

void AppController::addWaypoint(const ScanWaypoint &wp)
{
    _orbitalMode = false;
    _scanSession.addWaypoint(wp);
    emit logMessage(QString("Waypoint added — total: %1").arg(_scanSession.waypointCount()));
}

void AppController::clearWaypoints()
{
    _orbitalMode = false;
    _orbitalWps.clear();
    _currentOrbitalIdx = -1;
    _camTrajectory.clear();
    _scanSession.clearWaypoints();
    emit logMessage("Waypoints cleared");
    emit scanProgressUpdated(0, 0, "Idle — waypoints cleared");
    emit cameraTrajectoryUpdated({});
}

void AppController::startScanSequence(int nFrames, int settleMs, int intervalMs)
{
    if (!_camConnected) {
        emit logMessage("❌ Cannot start scan: Camera is not connected!");
        return;
    }

    _scanFetchPending  = false;
    _currentOrbitalIdx = -1;

    QTimer::singleShot(500, this, [this, nFrames, settleMs, intervalMs]() {
        _scanSession.start(nFrames, settleMs, intervalMs);
    });
}

void AppController::stopScanSequence()
{
    _scanSession.stop();
}

// ─── Orbital scan ─────────────────────────────────────────────────────────────

void AppController::loadOrbitalWaypoints(const QVector<OrbitalWaypoint> &wps,
                                         int nFrames, int settleMs, int intervalMs)
{
    Q_UNUSED(intervalMs);
    if (wps.isEmpty()) {
        emit logMessage("Orbital: no waypoints to load");
        return;
    }

    _orbitalMode        = true;
    _orbitalWps         = wps;
    _currentOrbitalIdx = -1;

    _currentScanDir = QCoreApplication::applicationDirPath() + "/scans/scan_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    _exportedPoses = QJsonObject();

    _scanSession.clearWaypoints();

    {
        ScanWaypoint sw;
        sw.dx = 0.0; sw.dy = 0.0; sw.dz = 0.0;
        sw.droll = 0.0; sw.dpitch = 0.0; sw.dyaw = 0.0;
        sw.nFrames        = nFrames;
        sw.settleMs       = settleMs;
        sw.orbitalWpIndex = 0;
        _scanSession.addWaypoint(sw);
    }

    for (int i = 1; i < wps.size(); ++i) {
        {
            ScanWaypoint home;
            home.dx = 0.0; home.dy = 0.0; home.dz = 0.0;
            home.droll = 0.0; home.dpitch = 0.0; home.dyaw = 0.0;
            home.nFrames        = 0;
            home.settleMs       = settleMs;
            home.orbitalWpIndex = -1;
            _scanSession.addWaypoint(home);
        }
        {
            ScanWaypoint sw;
            sw.dx     = wps[i].dx;
            sw.dy     = wps[i].dy;
            sw.dz     = wps[i].dz;
            sw.droll  = wps[i].droll;
            sw.dpitch = wps[i].dpitch;
            sw.dyaw   = wps[i].dyaw;
            sw.nFrames        = nFrames;
            sw.settleMs       = settleMs;
            sw.orbitalWpIndex = i;
            _scanSession.addWaypoint(sw);
        }
    }

    const int totalSteps = _scanSession.waypointCount();
    emit logMessage(
        QString("Orbital: %1 offset(s) → %2 scan steps")
            .arg(wps.size() - 1).arg(totalSteps));
    emit scanProgressUpdated(0, totalSteps, "Orbital trajectory loaded — press Start");
}

// ─── Continuous Recording (DatasetRecorder Async I/O) ─────────────────────────

void AppController::startRecording(int intervalMs)
{
    if (!_camConnected || _camera == _replayWorker) {
        emit workerError("Cần kết nối RealSense (LivePreview) trước khi bấm Record.");
        emit recordingStateChanged(false);
        return;
    }
    if (_state == AppState::Recording) {
        return;
    }

    const QString baseRecordsDir = QCoreApplication::applicationDirPath() + "/records";
    const QString ts = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    const QString sessionDir = baseRecordsDir + "/record_" + ts;

    if (!_recorder->start(sessionDir, _currentIntrinsics, qMax(0, intervalMs))) {
        emit recordingStateChanged(false);
        return;
    }

    _currentScanDir = sessionDir;
    _state = AppState::Recording;
    emit stateChanged(_state);
    emit recordingStateChanged(true);

    // Đảm bảo luồng camera luôn chạy liên tục (không phụ thuộc VIO)
    emit sig_fetchContinuous(0);
}

void AppController::stopRecording()
{
    if (_state != AppState::Recording) {
        return;
    }

    _recorder->stop();
    if (_imuFile.isOpen()) {
        _imuFile.close();
    }

    _state = _camConnected ? AppState::LivePreview : AppState::Idle;
    emit stateChanged(_state);
    emit recordingStateChanged(false);
    emit logMessage("⏹ Đã dừng Record.");
}

void AppController::startHandheldScan(int intervalMs)
{
    if (!_camConnected) {
        emit logMessage("❌ Cannot start scanning: Camera is not connected!");
        return;
    }

    emit logMessage("🔴 Started continuous handheld scanning...");
    emit recordingStateChanged(true);
    emit sig_fetchCount(999999, intervalMs);
}

void AppController::stopHandheldScan()
{
    emit sig_stopFetch();
    emit recordingStateChanged(false);
    emit logMessage("⏹ Stopped continuous scanning.");
}

// ─── Replay Scan (Kiểm tra thư mục TRƯỚC khi ngắt RealSense) ────────────────

void AppController::startReplay(const QString &dirPath)
{
    if (_state == AppState::Recording) {
        emit workerError("Hãy dừng Record trước khi Replay.");
        return;
    }

    // 1. KIỂM TRA HỢP LỆ TRƯỚC KHI NGẮT REALSENSE
    QDir rgbDir(dirPath + "/rgb");
    QDir depthDir(dirPath + "/depth_raw");
    const QStringList rgbFiles = rgbDir.entryList({"*.png", "*.jpg"}, QDir::Files);
    if (!rgbDir.exists() || !depthDir.exists() || rgbFiles.isEmpty()) {
        emit workerError(
            QString("Thư mục không hợp lệ (thiếu rgb/ hoặc depth_raw/ rỗng): %1").arg(dirPath));
        return; // Giữ nguyên trạng thái hiện tại, KHÔNG ngắt RealSense!
    }

    // 2. Hợp lệ mới ngắt RealSense hiện tại (nếu đang mở) và chuyển sang ReplayWorker
    if (_camera && _camera != _replayWorker) {
        emit sig_disconnectCamera();
        _camera->disconnect(this);
        this->disconnect(_camera);
    }

    _camera = _replayWorker;
    wireCamera();

    _currentScanDir = dirPath;
    _state = AppState::Replaying;
    emit stateChanged(_state);
    emit logMessage(QString("📼 Bắt đầu Replay (%1 frames): %2").arg(rgbFiles.size()).arg(dirPath));

    emit sig_connectCamera(dirPath, 0);
    emit sig_fetchContinuous(33);
}

void AppController::onReplayFinished()
{
    emit sig_stopFetch();
    _state = AppState::Idle;
    emit stateChanged(_state);
    emit logMessage("📼 Replay hoàn tất.");
}

// ─── Camera callbacks ─────────────────────────────────────────────────────────

void AppController::onCameraConnectionChanged(bool connected)
{
    _camConnected = connected;

    if (connected) {
        if (_camera == _replayWorker) {
            _state = AppState::Replaying;
        } else {
            _state = AppState::LivePreview;
            // Tự động bật Live Preview để người dùng thấy PointCloud & sẵn sàng Record ngay
            emit sig_fetchContinuous(0);
        }
    } else {
        if (_state == AppState::Recording) {
            _recorder->stop();
            emit recordingStateChanged(false);
        }
        _state = AppState::Idle;
    }

    emit stateChanged(_state);
    emit cameraConnectionChanged(connected);
}

void AppController::onCameraIntrinsicsReady(int width, int height, double fx, double fy, double cx, double cy)
{
    _currentIntrinsics.width = width;
    _currentIntrinsics.height = height;
    _currentIntrinsics.fx = fx;
    _currentIntrinsics.fy = fy;
    _currentIntrinsics.cx = cx;
    _currentIntrinsics.cy = cy;
    _currentIntrinsics.depthScale = 1000.0;

    _intrinsics.rgbW = width;
    _intrinsics.rgbH = height;
    _intrinsics.depthW = width;
    _intrinsics.depthH = height;
    _intrinsics.fx = static_cast<float>(fx);
    _intrinsics.fy = static_cast<float>(fy);
    _intrinsics.cx = static_cast<float>(cx);
    _intrinsics.cy = static_cast<float>(cy);

    _projector.setIntrinsics(_intrinsics);
    emit intrinsicsUpdated(_intrinsics);
}

void AppController::onCameraRGBDReady(RGBDFrame frame)
{
    if (auto *cw = qobject_cast<CameraWorker*>(_camera))
        cw->_cloudPending.store(false);

    // 1. Đẩy vào DatasetRecorder trên luồng I/O riêng nếu đang Record (không chặn bởi VIO)
    if (_state == AppState::Recording && _recorder->isRecording()
        && frame.rgb.valid && frame.depth.valid)
    {
        const int w = frame.rgb.width;
        const int h = frame.rgb.height;
        const int dw = frame.depth.width;
        const int dh = frame.depth.height;
        if (w > 0 && h > 0 && frame.rgb.data.size() >= w * h * 3
            && dw > 0 && dh > 0 && frame.depth.z_mm_data.size() >= dw * dh)
        {
            cv::Mat colorMat(h, w, CV_8UC3,
                             const_cast<uchar*>(reinterpret_cast<const uchar*>(frame.rgb.data.constData())));
            cv::Mat depthU16(dh, dw, CV_16UC1);
            uint16_t *dst = depthU16.ptr<uint16_t>(0);
            const float *src = frame.depth.z_mm_data.constData();
            const int totalPx = dw * dh;
            for (int i = 0; i < totalPx; ++i) {
                const float v = src[i];
                dst[i] = (v > 0.0f && v < 65535.0f) ? static_cast<uint16_t>(v + 0.5f) : 0;
            }
            _recorder->enqueueFrame(colorMat, depthU16, cv::Mat::eye(4, 4, CV_64F), /*isRgb=*/true);
        }
    }

    // 2. Throttle 3D Live Preview xuống ~12 FPS (80ms) để không làm nghẽn Main UI Thread
    // (Luôn dựng ngay nếu đang chụp Orbital Scan _scanFetchPending)
    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    if (!_scanFetchPending && (nowMs - _lastPreviewMs < 80)) {
        return;
    }
    _lastPreviewMs = nowMs;

    // 3. Project depth to point cloud để hiển thị Live Preview trên UI
    PointCloud cloud_data = _projector.build(frame.rgb, frame.depth, frame.depth.z_mm_data);

    QMatrix4x4 camInvTransform;
    if (_orbitalMode && !_orbitalWps.isEmpty() && _currentOrbitalIdx >= 0) {
        const int idx = qBound(0, _currentOrbitalIdx, _orbitalWps.size() - 1);
        const OrbitalWaypoint &wp = _orbitalWps[idx];
        camInvTransform = _transformer.computeInvTransform(
            wp.dx, wp.dy, wp.dz,
            wp.droll, wp.dpitch, wp.dyaw);
    }

    GeometryFrame gf;
    gf.cloud_raw = cloud_data;
    gf.camInvTransform = camInvTransform;
    gf.capturedAt = static_cast<quint64>(QDateTime::currentMSecsSinceEpoch());

    _scanEngine.addFrame(cloud_data, camInvTransform);

    if (_orbitalMode && !_orbitalWps.isEmpty() && _currentOrbitalIdx >= 0) {
        const int idx = qBound(0, _currentOrbitalIdx, _orbitalWps.size() - 1);
        const OrbitalWaypoint &wp = _orbitalWps[idx];
        gf.pose.x = wp.dx;
        gf.pose.y = wp.dy;
        gf.pose.z = wp.dz;
        gf.pose.rx = wp.droll;
        gf.pose.ry = wp.dpitch;
        gf.pose.rz = wp.dyaw;

        QMatrix4x4 T_base_cam = _transformer.computeBaseCamTransform(
            wp.dx, wp.dy, wp.dz, wp.droll, wp.dpitch, wp.dyaw);

        while (_camTrajectory.size() <= idx)
            _camTrajectory.append(QMatrix4x4());
        _camTrajectory[idx] = T_base_cam;

        QVector<QMatrix4x4> relativePoses;
        QMatrix4x4 T_inv0 = _camTrajectory.isEmpty() ? QMatrix4x4() : _camTrajectory.first().inverted();
        for (const auto &T : _camTrajectory) {
            relativePoses.append(T.isIdentity() ? T : (T_inv0 * T));
        }
        emit cameraTrajectoryUpdated(relativePoses);
    }

    emit geometryFrameReady(gf);

    if (_scanFetchPending) {
        _scanFetchPending = false;

        QDir().mkpath(_currentScanDir);
        QDir().mkpath(_currentScanDir + "/rgb");
        QDir().mkpath(_currentScanDir + "/depth_raw");
        QDir().mkpath(_currentScanDir + "/reconstruction");

        QString rawFilename = QString("%1/reconstruction/raw_%2.ply").arg(_currentScanDir).arg(_currentOrbitalIdx, 3, 10, QChar('0'));

        QFile intrFile(_currentScanDir + "/intrinsics.json");
        if (intrFile.open(QIODevice::WriteOnly)) {
            QJsonObject intrObj;
            intrObj["camera_model"] = _intrinsics.cameraModel;
            intrObj["fx"] = _intrinsics.fx;
            intrObj["fy"] = _intrinsics.fy;
            intrObj["cx"] = _intrinsics.cx;
            intrObj["cy"] = _intrinsics.cy;
            intrObj["width"] = _intrinsics.depthW;
            intrObj["height"] = _intrinsics.depthH;
            intrFile.write(QJsonDocument(intrObj).toJson());
            intrFile.close();
        }

        QJsonArray poseArray;
        QMatrix4x4 T_base_cam = _camTrajectory.value(_currentOrbitalIdx, QMatrix4x4());
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                poseArray.append(T_base_cam(r, c));
            }
        }

        QString errMs;
        PlyExporter::exportToFile(rawFilename, gf.cloud_raw.xyz, gf.cloud_raw.rgb, gf.cloud_raw.count, errMs);

        QString colorFile = QString("%1/rgb/%2.png").arg(_currentScanDir).arg(_currentOrbitalIdx, 6, 10, QChar('0'));
        QString depthRawFile = QString("%1/depth_raw/%2.png").arg(_currentScanDir).arg(_currentOrbitalIdx, 6, 10, QChar('0'));

        if (frame.rgb.valid) {
            QImage img(reinterpret_cast<const uchar*>(frame.rgb.data.constData()), frame.rgb.width, frame.rgb.height, QImage::Format_RGB888);
            img.save(colorFile);
        }
        if (frame.depth.valid) {
            int numPixels = frame.depth.width * frame.depth.height;
            QImage depthRawImg(frame.depth.width, frame.depth.height, QImage::Format_Grayscale16);
            uint16_t *pDstRaw = reinterpret_cast<uint16_t*>(depthRawImg.bits());
            for (int i = 0; i < numPixels; ++i) pDstRaw[i] = static_cast<uint16_t>(frame.depth.z_mm_data[i]);
            depthRawImg.save(depthRawFile);
        }

        _exportedPoses[QString("cam_%1").arg(_currentOrbitalIdx, 3, 10, QChar('0'))] = poseArray;

        QFile file(_currentScanDir + "/poses.json");
        if (file.open(QIODevice::WriteOnly)) {
            QJsonDocument doc(_exportedPoses);
            file.write(doc.toJson());
            file.close();
        }

        QFile tsFile(_currentScanDir + "/timestamps.csv");
        bool isNewFile = !tsFile.exists() || tsFile.size() == 0;
        if (tsFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            QTextStream stream(&tsFile);
            if (isNewFile) stream << "frame,timestamp_ns\n";
            stream << QString("cam_%1").arg(_currentOrbitalIdx, 3, 10, QChar('0')) << "," << frame.rgb.timestamp_ns << "\n";
            tsFile.close();
        }

        emit logMessage("💾 Đã lưu: " + rawFilename);
        _scanSession.notifyFrameCaptured();
    }
}

void AppController::onImuDataReady(const QVector<IMUData> &imuFrames)
{
    if (_state != AppState::Recording && !_scanFetchPending && !_scanSession.isRunning()) {
        if (_imuFile.isOpen()) _imuFile.close();
        return;
    }

    if (!_imuFile.isOpen() && !_currentScanDir.isEmpty()) {
        QDir().mkpath(_currentScanDir);
        _imuFile.setFileName(_currentScanDir + "/imu.csv");
        if (_imuFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QTextStream stream(&_imuFile);
            stream << "timestamp_ns,type,x,y,z\n";
        }
    }

    if (_imuFile.isOpen()) {
        QTextStream stream(&_imuFile);
        for (const auto &imu : imuFrames) {
            stream << imu.timestamp_ns << ","
                   << imu.type << ","
                   << imu.x << ","
                   << imu.y << ","
                   << imu.z << "\n";
        }
    }
}

void AppController::onCameraError(const QString &err)
{
    emit workerError(err);
    emit logMessage("Camera error: " + err);

    if (_scanFetchPending) {
        _scanFetchPending = false;

        if (err.contains("connection lost", Qt::CaseInsensitive) || !_camConnected) {
            emit logMessage("🚨 Scan: Aborting scan sequence due to lost connection!");
            _scanSession.stop();
        } else {
            emit logMessage("Scan: Skipping failed frame to prevent freeze.");
            _scanSession.notifyFrameCaptured();
        }
    }

    if (_state == AppState::Recording && ((!err.isEmpty() && err.contains("connection lost", Qt::CaseInsensitive)) || !_camConnected)) {
        emit logMessage("🚨 Record: Aborting continuous recording due to lost connection!");
        stopRecording();
    }
}

// ─── ScanSession callbacks ────────────────────────────────────────────────────

void AppController::onScanRequestMove(double dx, double dy, double dz,
                                      double droll, double dpitch, double dyaw)
{
    Q_UNUSED(dx); Q_UNUSED(dy); Q_UNUSED(dz);
    Q_UNUSED(droll); Q_UNUSED(dpitch); Q_UNUSED(dyaw);
    _scanSession.notifyMoveDone();
}

void AppController::onScanRequestSnapshot()
{
    emit logMessage("Scan snapshot done");
    _scanSession.notifySnapshotDone();
}

void AppController::onScanCaptureStarted(int orbitalWpIndex)
{
    _currentOrbitalIdx = orbitalWpIndex;
    emit logMessage(QString("Orbital capture ready: idx=%1").arg(orbitalWpIndex));
}

void AppController::onScanRequestFetch()
{
    _scanFetchPending = true;

    auto *cw = qobject_cast<CameraWorker*>(_camera);
    if (cw && cw->_cloudPending.load()) {
        emit logMessage("⏳ [Fetch] Camera busy — retry in 300ms");
        QTimer::singleShot(300, this, &AppController::onScanRequestFetch);
        return;
    }

    emit logMessage("📷 [Fetch] Requesting RGBD frame");
    emit sig_fetchCount(1, 0);
}

void AppController::onScanFinished()
{
    _currentOrbitalIdx = -1;
    emit scanFinished();
    emit logMessage(_orbitalMode
                        ? "Orbital scan complete — ready to export PLY"
                        : "Scan sequence complete — ready to export PLY");
}
