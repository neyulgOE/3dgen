#include "appcontroller.h"
#include "cameraworker.h"
#include "realsenseworker.h"
#include "replayworker.h"
#include "plyexporter.h"

#include <QDateTime>
#include <QCoreApplication>
#include <QProcess>
#include <QDir>
#include <QTimer>
#include <QThread>
#include <QJsonDocument>
#include <QJsonArray>
#include <QFile>
#include <QImage>
#include <QTextStream>
#include <QtConcurrent/QtConcurrentRun>

// ─── Constructor / Destructor ─────────────────────────────────────────────────

AppController::AppController(QObject *parent)
    : QObject(parent)
    , _workerThread(new QThread(this))
    , _vioThread   (new QThread(this))
    , _camera(nullptr)
    , _vioTracker(new VioTracker)
    , _keyframeExtractor(new KeyframeExtractor(this))
    , _fastSam(new FastSamInference(this))
    , _replayWorker(new ReplayWorker)
    , _scanSession(this)
{
    _workerThread->start();
    _replayWorker->moveToThread(_workerThread);
    connect(_workerThread, &QThread::finished, _replayWorker, &QObject::deleteLater);

    setCameraSource(0);

    if (_fastSam) {
        connect(_fastSam, &FastSamInference::logMessage, this, &AppController::logMessage);
        _fastSam->loadModel("../FastSAM-s.onnx"); // Wait, path should be relative to build dir or absolute
    }

    wireScan();
    
    // VIO Tracker Thread
    _vioTracker->moveToThread(_vioThread);
    connect(_vioThread, &QThread::finished, _vioTracker, &QObject::deleteLater);
    
    // Wire VIO Signals
    connect(_vioTracker, &VioTracker::trackingFrameReady,
            this, &AppController::onVioTrackingFrameReady);
    connect(_vioTracker, &VioTracker::logMessage,
            this, &AppController::logMessage);
            
    _vioThread->start();
}

AppController::~AppController()
{
    _scanSession.stop();
    emit sig_disconnectCamera();

    _workerThread->quit();
    _workerThread->wait(3000);
    
    _vioThread->quit();
    _vioThread->wait(3000);
}

// ─── Wiring ───────────────────────────────────────────────────────────────────

void AppController::wireCamera()
{
    connect(this, &AppController::sig_connectCamera,    _camera, &ICameraSource::connectHost);
    connect(this, &AppController::sig_disconnectCamera, _camera, &ICameraSource::disconnectHost);
    connect(this, &AppController::sig_fetchCount,       _camera, &ICameraSource::fetchCount);
    connect(this, &AppController::sig_fetchContinuous,  _camera, &ICameraSource::fetchContinuous);
    connect(this, &AppController::sig_stopFetch,        _camera, &ICameraSource::stopFetch);

    connect(_camera, &ICameraSource::connectionChanged,
            this, &AppController::onCameraConnectionChanged);
    connect(_camera, &ICameraSource::rgbdFrameReady,
            this, &AppController::onCameraRGBDReady);
    connect(_camera, &ICameraSource::errorOccurred,     this, &AppController::onCameraError);
    connect(_camera, &ICameraSource::imuDataReady,      this, &AppController::onImuDataReady);
    connect(_camera, &ICameraSource::logMessage,        this, &AppController::logMessage);
    
    // Connect Camera Source -> Vio Tracker
    connect(_camera, &ICameraSource::rgbdFrameReady,    _vioTracker, &VioTracker::onRgbdFrameReady);
    connect(_camera, &ICameraSource::imuDataReady,      _vioTracker, &VioTracker::onImuDataReady);
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
        emit sig_disconnectCamera();
        _camera->disconnect(this);
        this->disconnect(_camera);
        disconnect(_workerThread, &QThread::finished, _camera, &QObject::deleteLater);
        _camera->deleteLater();
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
    emit logMessage(QString("Connecting camera to %1:%2 …").arg(ip).arg(port));
    emit sig_connectCamera(ip, port);
}

void AppController::disconnectCamera() { emit sig_disconnectCamera(); }

void AppController::setIntrinsics(const CameraIntrinsics &intr)
{
    _projector.setIntrinsics(intr);
    emit logMessage("Intrinsics updated.");
    emit sig_setIntrinsics(intr);
    emit logMessage(QString("Intrinsics: fx=%1 fy=%2 cx=%3 cy=%4")
                        .arg(intr.fx).arg(intr.fy).arg(intr.cx).arg(intr.cy));
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

    // Khởi tạo thư mục export cho scan session mới
    _currentScanDir = QCoreApplication::applicationDirPath() + "/scans/scan_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    _exportedPoses = QJsonObject(); // Reset poses

    _scanSession.clearWaypoints();

    // Waypoint 0: capture tại Home
    {
        ScanWaypoint sw;
        sw.dx = 0.0; sw.dy = 0.0; sw.dz = 0.0;
        sw.droll = 0.0; sw.dpitch = 0.0; sw.dyaw = 0.0;
        sw.nFrames        = nFrames;
        sw.settleMs       = settleMs;
        sw.orbitalWpIndex = 0;
        _scanSession.addWaypoint(sw);
    }

    // Waypoint 1..N: về Home → đến offset → capture
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

// ─── Continuous Recording ─────────────────────────────────────────────────────

void AppController::startRecording(int intervalMs)
{
    if (!_camConnected) {
        emit logMessage("❌ Cannot start recording: Camera is not connected!");
        return;
    }
    _currentScanDir = QCoreApplication::applicationDirPath() + "/records/record_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir().mkpath(_currentScanDir);

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

    _exportedPoses = QJsonObject();
    _recordFrameIdx = 0;
    _isRecording = true;
    _recordIntervalMs = intervalMs;
    _lastRecordTimeMs = 0;

    emit logMessage("🔴 Started continuous recording to: " + _currentScanDir);
    emit recordingStateChanged(true);
    
    QMetaObject::invokeMethod(_vioTracker, "startTracking", Qt::QueuedConnection);
    
    // Always fetch at 30 FPS (interval = 0) so VIO Tracker works reliably
    // We will throttle disk saving in onCameraRGBDReady
    emit sig_fetchContinuous(0);
}

void AppController::stopRecording()
{
    if (!_isRecording) return;
    _isRecording = false;
    if (_imuFile.isOpen()) _imuFile.close();
    emit sig_stopFetch();
    QMetaObject::invokeMethod(_vioTracker, "stopTracking", Qt::QueuedConnection);
    emit recordingStateChanged(false);
    emit logMessage("⏹ Stopped recording.");
}

void AppController::startHandheldScan(int intervalMs)
{
    if (!_camConnected) {
        emit logMessage("❌ Cannot start scanning: Camera is not connected!");
        return;
    }
    
    _isRecording = false;
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

// ─── Replay Scan ─────────────────────────────────────────────────────────────

void AppController::startReplay(const QString &dirPath)
{
    emit logMessage("📼 Starting replay from: " + dirPath);
    _currentScanDir = dirPath;
    
    // Switch camera source to ReplayWorker
    if (_camera) {
        _camera->disconnectHost();
        _camera->disconnect(this);
    }
    
    _camera = _replayWorker;
    wireCamera();
    
    // Connect fetch finished to trigger stopVioScan
    connect(_replayWorker, &ReplayWorker::fetchCountFinished, this, &AppController::onReplayFinished, Qt::UniqueConnection);
    
    // Connect to dir
    emit sig_connectCamera(dirPath, 0);
    
    // Emulate starting VIO scan
    _isRecording = false; // We are not recording
    QMetaObject::invokeMethod(_vioTracker, "startTracking", Qt::QueuedConnection);
    if (_keyframeExtractor) _keyframeExtractor->reset();
    
    // Start fetching
    emit sig_fetchContinuous(33); // 30fps
}

void AppController::onReplayFinished()
{
    emit logMessage("📼 Replay finished. Automatically stopping VIO scan to process Phase 2 & 3.");
    stopVioScan();
}

// ─── VIO Free Scan ─────────────────────────────────────────────────────────────

void AppController::startVioScan()
{
    if (!_camConnected) {
        emit logMessage("❌ Cannot start VIO Scan: Camera is not connected!");
        return;
    }
    
    _isRecording = false; // We don't save to disk immediately, or maybe we do depending on requirements.
    
    emit logMessage("🔴 Started VIO Free Scanning...");
    
    if (_keyframeExtractor) _keyframeExtractor->reset();
    
    QMetaObject::invokeMethod(_vioTracker, "startTracking", Qt::QueuedConnection);
    
    // Yêu cầu worker bắn liên tục RGBD + IMU
    emit sig_fetchContinuous(0); // 0 = no interval delay, stream as fast as possible
}

void AppController::stopVioScan()
{
    emit sig_stopFetch();
    QMetaObject::invokeMethod(_vioTracker, "stopTracking", Qt::QueuedConnection);
    
    if (_keyframeExtractor) {
        auto keyframes = _keyframeExtractor->extractKeyframes(24);
        emit logMessage(QString("🔑 Phase 2: Extracted %1 sparse keyframes from scan session.").arg(keyframes.size()));
        
        QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
        
        // --- Phase 3: FastSAM Background Removal ---
        if (_fastSam) {
            emit logMessage("🧠 Phase 3: AI is processing 24 frames to remove background...");
            QString kfDir = _currentScanDir.isEmpty() ? "keyframes_debug" : _currentScanDir + "/keyframes";
            
            // Xóa sạch các ảnh cũ từ những lần Replay trước
            QDir dir(kfDir);
            if (dir.exists()) {
                dir.removeRecursively();
            }
            QDir().mkpath(kfDir);
            
            // Step 1: Save unmasked RGB to disk
            for (int i = 0; i < keyframes.size(); ++i) {
                auto &kf = keyframes[i];
                if (kf.rgbd.rgb.data.isEmpty()) continue;
                QImage img((const uchar*)kf.rgbd.rgb.data.constData(), 
                           kf.rgbd.rgb.width, kf.rgbd.rgb.height, QImage::Format_RGB888);
                img.save(kfDir + QString("/keyframe_%1_%2.png").arg(timestamp).arg(i, 2, 10, QChar('0')));
            }
            
            // Step 2: Run Python FastSAM script to process all images in the folder
            emit logMessage(QString("🐍 Spawning Python FastSAM process with text prompt '%1'...").arg(_aiPrompt));
            int ret = QProcess::execute("python3", {"../scripts/fastsam_batch.py", kfDir, timestamp, _aiPrompt});
            if (ret != 0) {
                emit logMessage("❌ Python FastSAM process failed!");
            } else {
                emit logMessage("✨ Phase 3: AI Background Removal finished. Check '" + kfDir + "' folder.");
            }
            
            // Step 3: Load the masked images back into memory
            for (int i = 0; i < keyframes.size(); ++i) {
                auto &kf = keyframes[i];
                if (kf.rgbd.rgb.data.isEmpty()) continue;
                QImage img(kfDir + QString("/keyframe_%1_%2.png").arg(timestamp).arg(i, 2, 10, QChar('0')));
                img = img.convertToFormat(QImage::Format_RGB888);
                kf.rgbd.rgb.data = QByteArray(reinterpret_cast<const char*>(img.constBits()), img.sizeInBytes());
            }
            
            // --- Phase 4: 3D Reconstruction ---
            emit logMessage("🏗 Phase 4: Reconstructing 3D Point Cloud from masked keyframes...");
            PointCloud finalCloud;
            for (int i = 0; i < keyframes.size(); ++i) {
                const auto &kf = keyframes[i];
                if (kf.rgbd.rgb.data.isEmpty()) continue;
                
                // Build local point cloud from masked RGBD
                PointCloud localCloud = _projector.build(kf.rgbd.rgb, kf.rgbd.depth, kf.rgbd.depth.z_mm_data);
                QMatrix4x4 T_world_cam = kf.pose;
                
                // Filter black pixels and apply transform
                for (int j = 0; j < localCloud.count; ++j) {
                    uint8_t r = localCloud.rgb[j * 3];
                    uint8_t g = localCloud.rgb[j * 3 + 1];
                    uint8_t b = localCloud.rgb[j * 3 + 2];
                    
                    // Skip FastSAM background (black)
                    if (r == 0 && g == 0 && b == 0) continue;
                    
                    // Skip far away points (ghosting behind cup surface)
                    if (localCloud.xyz[j * 3 + 2] > 800.0f) continue;
                    
                    QVector3D p(localCloud.xyz[j * 3], localCloud.xyz[j * 3 + 1], localCloud.xyz[j * 3 + 2]);
                    QVector3D p_world = T_world_cam.map(p);
                    
                    finalCloud.xyz.push_back(p_world.x());
                    finalCloud.xyz.push_back(p_world.y());
                    finalCloud.xyz.push_back(p_world.z());
                    
                    finalCloud.rgb.push_back(r);
                    finalCloud.rgb.push_back(g);
                    finalCloud.rgb.push_back(b);
                }
                
                // Export the individual transformed point cloud for ICP Registration later
                PointCloud transformedLocalCloud;
                transformedLocalCloud.xyz.resize(localCloud.count * 3);
                transformedLocalCloud.rgb.resize(localCloud.count * 3);
                int validCount = 0;
                for (int j = 0; j < localCloud.count; ++j) {
                    uint8_t r = localCloud.rgb[j * 3];
                    uint8_t g = localCloud.rgb[j * 3 + 1];
                    uint8_t b = localCloud.rgb[j * 3 + 2];
                    if (r == 0 && g == 0 && b == 0) continue;
                    
                    // Skip far away points (ghosting behind cup surface)
                    if (localCloud.xyz[j * 3 + 2] > 800.0f) continue;
                    
                    QVector3D p(localCloud.xyz[j * 3], localCloud.xyz[j * 3 + 1], localCloud.xyz[j * 3 + 2]);
                    QVector3D p_world = T_world_cam.map(p);
                    
                    transformedLocalCloud.xyz[validCount * 3] = p_world.x();
                    transformedLocalCloud.xyz[validCount * 3 + 1] = p_world.y();
                    transformedLocalCloud.xyz[validCount * 3 + 2] = p_world.z();
                    
                    transformedLocalCloud.rgb[validCount * 3] = r;
                    transformedLocalCloud.rgb[validCount * 3 + 1] = g;
                    transformedLocalCloud.rgb[validCount * 3 + 2] = b;
                    validCount++;
                }
                transformedLocalCloud.count = validCount;
                transformedLocalCloud.xyz.resize(validCount * 3);
                transformedLocalCloud.rgb.resize(validCount * 3);
                
                QString kfPlyPath = kfDir + QString("/cloud_%1.ply").arg(i, 2, 10, QChar('0'));
                QString errStr;
                PlyExporter::exportToFile(kfPlyPath, transformedLocalCloud.xyz, transformedLocalCloud.rgb, transformedLocalCloud.count, errStr);
            }
            finalCloud.count = finalCloud.xyz.size() / 3;
            emit logMessage(QString("🎯 Phase 4: Reconstruction complete! Final Cloud has %1 points.").arg(finalCloud.count));
            
            // Export to PLY
            QString plyPath = _currentScanDir.isEmpty() ? "scanned_object_" + timestamp + ".ply" : _currentScanDir + "/scanned_object_" + timestamp + ".ply";
            QString errStr;
            if (PlyExporter::exportToFile(plyPath, finalCloud.xyz, finalCloud.rgb, finalCloud.count, errStr)) {
                emit logMessage("💾 Exported 3D Point Cloud to: " + plyPath);
            } else {
                emit logMessage("❌ Failed to export PLY: " + errStr);
            }
        }
    }
    
    emit logMessage("⏹ Stopped VIO Free Scanning.");
}

// ─── Camera callbacks ─────────────────────────────────────────────────────────

void AppController::onCameraConnectionChanged(bool connected)
{
    _camConnected = connected;
    emit cameraConnectionChanged(connected);
}

void AppController::onCameraRGBDReady(RGBDFrame frame)
{
    if (auto *cw = qobject_cast<CameraWorker*>(_camera))
        cw->_cloudPending.store(false);

    // Project depth to point cloud
    PointCloud cloud_data = _projector.build(frame.rgb, frame.depth, frame.depth.z_mm_data);

    QMatrix4x4 camInvTransform;

    if (_orbitalMode && !_orbitalWps.isEmpty() && _currentOrbitalIdx >= 0) {
        const int idx = qBound(0, _currentOrbitalIdx, _orbitalWps.size() - 1);
        const OrbitalWaypoint &wp = _orbitalWps[idx];
        camInvTransform = _transformer.computeInvTransform(
            wp.dx, wp.dy, wp.dz,
            wp.droll, wp.dpitch, wp.dyaw);
    }
    // DEBUG — log transform matrix
    emit logMessage(QString("DEBUG T_inv: [%1 %2 %3 %4 | %5 %6 %7 %8 | %9 %10 %11 %12]")
                        .arg(camInvTransform(0,0),0,'f',3).arg(camInvTransform(0,1),0,'f',3)
                        .arg(camInvTransform(0,2),0,'f',3).arg(camInvTransform(0,3),0,'f',2)
                        .arg(camInvTransform(1,0),0,'f',3).arg(camInvTransform(1,1),0,'f',3)
                        .arg(camInvTransform(1,2),0,'f',3).arg(camInvTransform(1,3),0,'f',2)
                        .arg(camInvTransform(2,0),0,'f',3).arg(camInvTransform(2,1),0,'f',3)
                        .arg(camInvTransform(2,2),0,'f',3).arg(camInvTransform(2,3),0,'f',2));
    GeometryFrame gf;
    gf.cloud_raw = cloud_data;
    gf.camInvTransform = camInvTransform;

    // Feed to ScanEngine
    _scanEngine.addFrame(cloud_data, camInvTransform);

    emit geometryFrameReady(gf);
    gf.capturedAt      = (quint64)QDateTime::currentMSecsSinceEpoch();

    if (_orbitalMode && !_orbitalWps.isEmpty() && _currentOrbitalIdx >= 0) {
        const int idx = qBound(0, _currentOrbitalIdx, _orbitalWps.size() - 1);
        const OrbitalWaypoint &wp = _orbitalWps[idx];
        gf.pose.x = wp.dx;
        gf.pose.y = wp.dy;
        gf.pose.z = wp.dz;
        gf.pose.rx = wp.droll;
        gf.pose.ry = wp.dpitch;
        gf.pose.rz = wp.dyaw;

        // ── Cập nhật camera trajectory ──────────────────────────────────────────
        // T_base_cam tại waypoint này (phục vụ visualizer)
        QMatrix4x4 T_base_cam = _transformer.computeBaseCamTransform(
            wp.dx, wp.dy, wp.dz, wp.droll, wp.dpitch, wp.dyaw);

        // Đảm bảo mảng trajectory có đủ chỗ cho idx này
        while (_camTrajectory.size() <= idx)
            _camTrajectory.append(QMatrix4x4());
        _camTrajectory[idx] = T_base_cam;

        emit logMessage(QString("[Trajectory] WP%1: t=(%2, %3, %4)")
            .arg(idx)
            .arg(T_base_cam(0,3),0,'f',1)
            .arg(T_base_cam(1,3),0,'f',1)
            .arg(T_base_cam(2,3),0,'f',1));
            
        // Transform the trajectory to Camera 0's frame before sending to PointCloudWidget
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
        
        // Export point cloud for offline ICP (Local Camera Frame)
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
        
        // Cập nhật ma trận Pose vào file poses.json
        // NOTE: Xuất T_base_cam (tuyệt đối) để phục vụ SLAM thay vì camInvTransform (tương đối)
        QJsonArray poseArray;
        // Chúng ta đã có T_base_cam tính ở trên và lưu trong _camTrajectory
        QMatrix4x4 T_base_cam = _camTrajectory.value(_currentOrbitalIdx, QMatrix4x4());
        
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                poseArray.append(T_base_cam(r, c));
            }
        }
        
        QString errMs;
        // Xuất nguyên bản Local Point Cloud!
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

        
        // Print camera pose info
        if (!camInvTransform.isIdentity()) {
            float tx = camInvTransform(0,3), ty = camInvTransform(1,3), tz = camInvTransform(2,3);
            emit logMessage(QString("📍 [Camera Pose %1] T: x=%.1f, y=%.1f, z=%.1f").arg(_currentOrbitalIdx).arg(tx).arg(ty).arg(tz));
        } else {
            emit logMessage(QString("📍 [Camera Pose %1] Origin (0,0,0)").arg(_currentOrbitalIdx));
        }
        
        emit logMessage("💾 Đã lưu: " + rawFilename);

        _scanSession.notifyFrameCaptured();
    }
    
    if (_isRecording) {
        qint64 currentMs = QDateTime::currentMSecsSinceEpoch();
        if (currentMs - _lastRecordTimeMs < _recordIntervalMs) {
            return; // Skip saving this frame, but VIO Tracker still processed it
        }
        _lastRecordTimeMs = currentMs;

        QDir().mkpath(_currentScanDir);
        QDir().mkpath(_currentScanDir + "/rgb");
        QDir().mkpath(_currentScanDir + "/depth_raw");
        QDir().mkpath(_currentScanDir + "/reconstruction");

        QString rawFilename = QString("%1/reconstruction/raw_%2.ply").arg(_currentScanDir).arg(_recordFrameIdx, 3, 10, QChar('0'));

        QString errMs;
        PlyExporter::exportToFile(rawFilename, gf.cloud_raw.xyz, gf.cloud_raw.rgb, gf.cloud_raw.count, errMs);
        
        QString colorFile = QString("%1/rgb/%2.png").arg(_currentScanDir).arg(_recordFrameIdx, 6, 10, QChar('0'));
        QString depthRawFile = QString("%1/depth_raw/%2.png").arg(_currentScanDir).arg(_recordFrameIdx, 6, 10, QChar('0'));
        
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
        
        emit logMessage(QString("⏺ [Record] Saved frame %1 -> %2").arg(_recordFrameIdx).arg(rawFilename));
        
        QJsonArray poseArray;
        QMatrix4x4 currentVioPose = _vioTracker->getCurrentPose();
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) poseArray.append(currentVioPose(r, c));
        }
        _exportedPoses[QString("cam_%1").arg(_recordFrameIdx, 3, 10, QChar('0'))] = poseArray;
        
        QFile file(_currentScanDir + "/poses.json");
        if (file.open(QIODevice::WriteOnly)) {
            file.write(QJsonDocument(_exportedPoses).toJson());
            file.close();
        }
        
        QFile tsFile(_currentScanDir + "/timestamps.csv");
        bool isNewFile = !tsFile.exists() || tsFile.size() == 0;
        if (tsFile.open(QIODevice::WriteOnly | QIODevice::Append | QIODevice::Text)) {
            QTextStream stream(&tsFile);
            if (isNewFile) stream << "frame,timestamp_ns\n";
            stream << QString("cam_%1").arg(_recordFrameIdx, 3, 10, QChar('0')) << "," << frame.rgb.timestamp_ns << "\n";
            tsFile.close();
        }


        _recordFrameIdx++;
    }
}

void AppController::onImuDataReady(const QVector<IMUData> &imuFrames)
{
    // Chỉ ghi file khi đang ở chế độ Record hoặc đang quét SLAM (có thư mục _currentScanDir hợp lệ)
    if (!_isRecording && !_scanFetchPending && !_scanSession.isRunning()) {
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
            _scanSession.notifyFrameCaptured(); // skip the frame and continue
        }
    }
    
    if (_isRecording && ((!err.isEmpty() && err.contains("connection lost", Qt::CaseInsensitive)) || !_camConnected)) {
        emit logMessage("🚨 Record: Aborting continuous recording due to lost connection!");
        stopRecording();
    }
}

void AppController::onVioTrackingFrameReady(const TrackingFrame &frame)
{
    // Cập nhật lên UI (CameraTrajectory hoặc 3D view)
    // frame.pose chính là T_world_cam (4x4) tính từ VIO
    
    if (_keyframeExtractor) {
        _keyframeExtractor->addFrame(frame);
    }
    
    // emit logMessage(QString("📍 [VIO] T: x=%.1f, y=%.1f, z=%.1f")
    //                 .arg(frame.pose(0,3)).arg(frame.pose(1,3)).arg(frame.pose(2,3)));
                    
    // Có thể emit signal để UI vẽ quỹ đạo
    QVector<QMatrix4x4> trajectory;
    trajectory.append(frame.pose);
    emit cameraTrajectoryUpdated(trajectory);
}

// ─── ScanSession callbacks ────────────────────────────────────────────────────

void AppController::onScanRequestMove(double dx, double dy, double dz,
                                      double droll, double dpitch, double dyaw)
{
    // Robot connection removed, immediately notify move done
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
