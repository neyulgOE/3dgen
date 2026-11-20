#ifndef APPCONTROLLER_H
#define APPCONTROLLER_H

#include "coordinatetransformer.h"
#include "icamerasource.h"
#include "depthprojector.h"
#include "orbitaltrajectory.h"
#include "scansession.h"
#include "scanengine.h"
#include "types.h"
#include "vio_tracker.h"
#include "keyframeextractor.h"
#include "fastsaminference.h"

#include <QFile>
#include <QJsonObject>
#include <QMatrix3x3>
#include <QMatrix4x4>
#include <QObject>
#include <QThread>
#include <QVector>
#include <QVector3D>
#include <QtConcurrent/QtConcurrentRun>

class AppController : public QObject
{
    Q_OBJECT

public:
    explicit AppController(QObject *parent = nullptr);
    ~AppController() override;

    bool cameraConnected() const { return _camConnected; }

public slots:
    // Camera
    void setCameraSource (int source);
    void connectCamera   (const QString &ip, quint16 port);
    void disconnectCamera();
    void setIntrinsics   (const CameraIntrinsics &intr);

    // Stop-and-scan
    void addWaypoint      (const ScanWaypoint &wp);
    void clearWaypoints   ();
    void startScanSequence(int nFrames, int settleMs, int intervalMs);
    void stopScanSequence ();

    // Orbital scan
    void loadOrbitalWaypoints(const QVector<OrbitalWaypoint> &wps,
                              int nFrames, int settleMs, int intervalMs);

    // Continuous Recording / Handheld Scan
    void startRecording(int intervalMs);
    void stopRecording();

    // Replay a previously recorded scan
    void startReplay(const QString &dirPath);
    void startHandheldScan(int intervalMs);
    void stopHandheldScan();
    
    // VIO Free Scan
    void startVioScan();
    void stopVioScan();
    
    // AI Prompt Configuration
    void setAiPrompt(const QString &prompt) { _aiPrompt = prompt.isEmpty() ? "cup" : prompt; }
    QString aiPrompt() const { return _aiPrompt; }

    ScanEngine* scanEngine() { return &_scanEngine; }

    void setCurrentScanDir(const QString &dir) { _currentScanDir = dir; }
    QString currentScanDir() const { return _currentScanDir; }

signals:
    void geometryFrameReady(GeometryFrame frame);

    void cameraConnectionChanged(bool connected);

    void scanProgressUpdated(int current, int total, const QString &status);
    void scanFinished();

    void recordingStateChanged(bool isRecording);

    /// Phát sau mỗi waypoint: danh sách các pose T_base_cam để visualize trajectory.
    void cameraTrajectoryUpdated(QVector<QMatrix4x4> poses);

    void logMessage (const QString &msg);
    void workerError(const QString &err);

    // Internal → CameraWorker (worker thread)
    void sig_connectCamera   (const QString &ip, quint16 port);
    void sig_disconnectCamera();
    void sig_fetchCount      (int n, int intervalMs);
    void sig_fetchContinuous (int intervalMs);
    void sig_stopFetch       ();
    void sig_setIntrinsics   (const CameraIntrinsics &intr);

private slots:
    void onCameraConnectionChanged(bool connected);
    void onCameraRGBDReady(RGBDFrame frame);
    void onImuDataReady(const QVector<IMUData> &imuFrames);
    void onCameraError            (const QString &err);
    void onVioTrackingFrameReady  (const TrackingFrame &frame);

    void onScanRequestMove    (double dx, double dy, double dz,
                           double droll, double dpitch, double dyaw);
    void onScanRequestSnapshot();
    void onScanRequestFetch   ();
    void onScanCaptureStarted (int orbitalWpIndex);
    void onScanFinished       ();
    void onReplayFinished();

private:
    void wireCamera();
    void wireScan  ();

    QThread        *_workerThread = nullptr;
    QThread        *_vioThread    = nullptr;
    ICameraSource  *_camera       = nullptr;
    VioTracker     *_vioTracker  = nullptr;
    KeyframeExtractor *_keyframeExtractor = nullptr;
    FastSamInference *_fastSam = nullptr;
    class ReplayWorker *_replayWorker = nullptr;
    ScanSession     _scanSession;
    ScanEngine      _scanEngine;
    CoordinateTransformer _transformer;
    DepthProjector        _projector;
    CameraIntrinsics      _intrinsics;

    bool _camConnected     = false;
    bool _scanFetchPending = false;

    bool                     _orbitalMode       = false;
    QVector<OrbitalWaypoint> _orbitalWps;
    int                      _currentOrbitalIdx = -1;

    bool                     _isRecording = false;
    qint64                   _recordIntervalMs = 200;
    qint64                   _lastRecordTimeMs = 0;
    int                      _recordFrameIdx = 0;

    QJsonObject              _exportedPoses;
    QString                  _currentScanDir;
    QString                  _aiPrompt = "cup";

    /// Camera trajectory: mọi phần tử là T_base_cam tại waypoint tương ứng.
    QVector<QMatrix4x4>      _camTrajectory;

    QFile _imuFile;
};

#endif // APPCONTROLLER_H
