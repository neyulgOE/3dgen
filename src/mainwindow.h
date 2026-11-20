#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "appcontroller.h"
#include "meshviewerwidget.h"
#include "orbitalscandialog.h"
#include "pointcloudwidget.h"
#include "types.h"

#include <QMainWindow>
#include <QTextCursor>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QPlainTextEdit;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTimer;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    // Camera UI
    void onCameraConnectClicked   ();
    void onCameraDisconnectClicked();
    void onCameraConnectionChanged(bool connected);

    // Mesh Builder tab
    void onBuildMeshClicked   ();
    void onViewLoadMeshClicked();
    void onMeshAutoClean      ();
    void onMeshUndo           ();
    void onMeshEraserToggle   (bool checked);
    void onMeshSave           ();
    void onMeshExportObj      ();
    void onMeshPoissonRecon   ();
    void onMeshRemoveFile     ();
    void onMeshChanged        (int triCount);
    void onMeshStatus         (const QString &msg);

    // ICP Python integration
    void onRunIcpClicked();
    void onExportMeshClicked();
    void onKillProcessesClicked();
    void onViewMeshClicked();

    // Geometry
    void onGeometryFrameReady(const GeometryFrame &gf);
    void onWorkerError       (const QString &err);

    // Accumulator stats
    void onAccumStatsUpdated(int voxelCount, int pointCount);

    // Scan UI
    void onScanProgressUpdated(int current, int total, const QString &status);
    void onScanFinished       ();

    // Orbital scan UI
    void onOrbitalWaypointsReady  (const QVector<OrbitalWaypoint> &wps);
    void onStartOrbitalScanClicked();

    // Recording
    void onRecordButtonClicked    ();
    void onRecordingStateChanged  (bool isRecording);

    // Viewport & Tools
    void onCloudSnapshotClicked   ();
    void onMeshSnapshotClicked    ();
    void onOpenDataFolderClicked  ();

    // Log
    void appendLog(const QString &msg);
    void flushLog ();

private:
    void setupUi            ();
    void wireSelf           ();
    void setCameraConnectedState(bool connected);
    void pushIntrinsics     ();
    void pushCamToTool      ();
    void exportPointCloudPLY();
    void setScanRunningState(bool running);

    // ── Backend ───────────────────────────────────────────────────────────────
    AppController    *_ctrl  = nullptr;

    QString _centroidLogFilename;
    int _centroidWpCount = 0;

    // ── Camera Connection ─────────────────────────────────────────────────────
    QComboBox   *_camSourceCombo = nullptr;
    QPushButton *_camConnBtn   = nullptr;
    QPushButton *_camDiscBtn   = nullptr;
    QLabel      *_camStatusLbl = nullptr;

    // ── Mesh Builder ──────────────────────────────────────────────────────────
    MeshViewerWidget *_meshView       = nullptr;
    QPushButton      *_buildMeshBtn2  = nullptr;  // Build Mesh in Mesh tab
    QPushButton      *_loadMeshBtn    = nullptr;  // Load PLY into viewer
    QPushButton      *_meshAutoBtn    = nullptr;
    QPushButton      *_meshEraserBtn  = nullptr;
    QPushButton      *_meshUndoBtn    = nullptr;
    QPushButton      *_meshSaveBtn    = nullptr;
    QPushButton      *_exportObjBtn   = nullptr;  // Export current PLY as OBJ
    QPushButton      *_meshPoissonBtn = nullptr;
    QPushButton      *_meshRemoveBtn  = nullptr;
    QLabel           *_meshInfoLbl    = nullptr;
    QListWidget      *_meshFileList   = nullptr;
    QCheckBox        *_meshWhiteBgCheck = nullptr;

    // ── 3D View ───────────────────────────────────────────────────────────────
    PointCloudWidget *_cloudView      = nullptr;
    QLabel           *_cloudInfoLbl   = nullptr;
    QComboBox        *_cameraModelCombo = nullptr;
    QDoubleSpinBox   *_fxSpin         = nullptr;
    QDoubleSpinBox   *_fySpin         = nullptr;
    QDoubleSpinBox   *_cxSpin         = nullptr;
    QDoubleSpinBox   *_cySpin         = nullptr;
    QPushButton      *_applyIntrBtn   = nullptr;
    QCheckBox        *_accumCheck     = nullptr;
    QCheckBox        *_showFovCheck   = nullptr;
    QCheckBox        *_icpCheck       = nullptr;
    QPushButton      *_clearCloudBtn  = nullptr;
    QPushButton      *_exportCloudBtn = nullptr;
    QPushButton      *_runIcpBtn      = nullptr;
    QPushButton      *_killProcessesBtn = nullptr;
    QPushButton      *_exportMeshBtn  = nullptr;
    QPushButton      *_viewMeshBtn    = nullptr;
    QLabel           *_accumStatsLbl  = nullptr;
    QDoubleSpinBox   *_voxelSizeSpin  = nullptr;
    QCheckBox        *_scanMaskCheck  = nullptr;
    QDoubleSpinBox   *_maskCenterXSpin = nullptr;
    QDoubleSpinBox   *_maskCenterYSpin = nullptr;
    QDoubleSpinBox   *_maskCenterZSpin = nullptr;
    QDoubleSpinBox   *_maskRadiusSpin  = nullptr;

    // ── Continuous Recording ──────────────────────────────────────────────────
    QPushButton      *_recordBtn       = nullptr;
    QSpinBox         *_recordIntervalSpin = nullptr;

    QPushButton      *_vioScanBtn      = nullptr;
    QPushButton      *_replayBtn       = nullptr;

    // ── Scan Timing ───────────────────────────────────────────────────────────
    QSpinBox *_scanFramesSpin   = nullptr;
    QSpinBox *_scanSettleSpin   = nullptr;
    QSpinBox *_scanIntervalSpin = nullptr;

    // ── Orbital Scan ──────────────────────────────────────────────────────────
    OrbitalScanWidget *_orbitalWidget    = nullptr;
    QPushButton       *_startOrbitalBtn  = nullptr;
    QPushButton       *_stopOrbitalBtn   = nullptr;
    QLabel            *_orbitalStatusLbl = nullptr;
    QProgressBar      *_orbitalProgress  = nullptr;
    QLineEdit   *_aiPromptEdit  = nullptr;
    QGroupBox         *_orbGroup         = nullptr;
    QLabel            *_orbModeBadge     = nullptr;

    // ── Navigation & Extended UI ─────────────────────────────────────────────
    QTabWidget       *_mainTabs           = nullptr;
    QPushButton      *_meshWireframeBtn   = nullptr;
    QPushButton      *_cloudSnapshotBtn   = nullptr;
    QPushButton      *_meshSnapshotBtn    = nullptr;
    QCheckBox        *_gridCheck          = nullptr;
    QDoubleSpinBox   *_pointSizeSpin      = nullptr;
    QLabel           *_logCountLbl        = nullptr;

    // ── Log ───────────────────────────────────────────────────────────────────
    QPlainTextEdit *_logEdit = nullptr;

    // ── UI state ──────────────────────────────────────────────────────────────
    bool    _camConnected = false;
    bool    _scanRunning  = false;
    QString _captureDir;

    // ── Log throttle ──────────────────────────────────────────────────────────
    QStringList _pendingLogLines;
    QTimer     *_logFlushTimer = nullptr;
    static constexpr int LOG_MAX_LINES = 400;
    static constexpr int LOG_FLUSH_MS  = 150;
};

#endif // MAINWINDOW_H