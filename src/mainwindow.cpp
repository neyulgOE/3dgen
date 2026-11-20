#include "mainwindow.h"
#include "cloudaccumulator.h"
#include "orbitalscandialog.h"
#include "plyexporter.h"
#include "pointcloudwidget.h"
#include "exporter.h"

#include <QApplication>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPalette>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QStatusBar>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QDesktopServices>
#include <QUrl>
#include <QClipboard>

// ─── Constructor / Destructor
// ─────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), _ctrl(new AppController(this)) {
  setWindowTitle("TMVisionary 3D Studio v4.8 Pro");
  resize(1380, 880);

  setupUi();
  wireSelf();

  setCameraConnectedState(false);

  _captureDir = QDir::currentPath() + "/captures";
  statusBar()->showMessage("Ready — not connected");

  _orbitalWidget->setTimingParams(_scanFramesSpin->value(),
                                  _scanSettleSpin->value());

  _logFlushTimer = new QTimer(this);
  _logFlushTimer->setInterval(LOG_FLUSH_MS);
  connect(_logFlushTimer, &QTimer::timeout, this, &MainWindow::flushLog);
  _logFlushTimer->start();


  QTimer::singleShot(300, this, [this]() { pushIntrinsics(); });
}

MainWindow::~MainWindow() {}

// ─── UI construction
// ──────────────────────────────────────────────────────────

void MainWindow::setupUi() {
  qApp->setStyle("Fusion");

  QPalette darkPalette;
  darkPalette.setColor(QPalette::Window, QColor("#171b24"));
  darkPalette.setColor(QPalette::WindowText, QColor("#f1f5f9"));
  darkPalette.setColor(QPalette::Base, QColor("#11141a"));
  darkPalette.setColor(QPalette::AlternateBase, QColor("#171b24"));
  darkPalette.setColor(QPalette::ToolTipBase, QColor("#1e2430"));
  darkPalette.setColor(QPalette::ToolTipText, QColor("#f1f5f9"));
  darkPalette.setColor(QPalette::Text, QColor("#f1f5f9"));
  darkPalette.setColor(QPalette::Button, QColor("#242b3a"));
  darkPalette.setColor(QPalette::ButtonText, QColor("#f8fafc"));
  darkPalette.setColor(QPalette::BrightText, Qt::red);
  darkPalette.setColor(QPalette::Link, QColor("#3b82f6"));
  darkPalette.setColor(QPalette::Highlight, QColor("#2563eb"));
  darkPalette.setColor(QPalette::HighlightedText, Qt::white);
  qApp->setPalette(darkPalette);

  // Modern Dark Studio Theme Stylesheet
  qApp->setStyleSheet(
      "QMainWindow { background-color: #11141a; }"
      "QDialog, QMessageBox { background-color: #171b24; color: #f8fafc; }"
      "QMessageBox QLabel { color: #f8fafc; font-size: 12px; background-color: transparent; min-height: 28px; }"
      "QDialog QLabel { color: #f8fafc; background-color: transparent; }"
      "QWidget { font-family: 'Segoe UI', 'Ubuntu', 'Roboto', sans-serif; color: #e2e8f0; font-size: 12px; }"
      "QToolTip { background-color: #1e2430; color: #f1f5f9; border: 1px solid #3b82f6; border-radius: 4px; padding: 4px 8px; }"
      "QGroupBox { background-color: #171b24; border: 1px solid #283040; border-radius: 8px; margin-top: 14px; padding-top: 14px; font-weight: bold; }"
      "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 6px; color: #60a5fa; }"
      "QPushButton { background-color: #242b3a; color: #f8fafc; border: 1px solid #374151; border-radius: 5px; padding: 6px 12px; font-weight: 600; font-size: 11px; }"
      "QPushButton:hover { background-color: #2f384c; border-color: #4b5563; }"
      "QPushButton:pressed { background-color: #1a202c; }"
      "QPushButton:checked { background-color: #1e3a8a; border: 1px solid #3b82f6; color: #93c5fd; }"
      "QPushButton:disabled { background-color: #141720; color: #475569; border-color: #1e2430; }"
      "QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #13161f; color: #f1f5f9; border: 1px solid #2a3344; border-radius: 5px; padding: 4px 7px; font-size: 11px; }"
      "QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border-color: #3b82f6; }"
      "QComboBox::drop-down { border: none; width: 18px; }"
      "QTabWidget::pane { border: 1px solid #242c3b; background-color: #141720; border-radius: 0 0 6px 6px; }"
      "QTabBar::tab { background-color: #181d27; color: #94a3b8; padding: 8px 18px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; font-weight: bold; font-size: 12px; }"
      "QTabBar::tab:selected { background-color: #242c3b; color: #60a5fa; border-bottom: 2px solid #3b82f6; }"
      "QTabBar::tab:hover:!selected { background-color: #1f2533; color: #cbd5e1; }"
      "QScrollBar:vertical { border: none; background: #11141a; width: 8px; margin: 0px; }"
      "QScrollBar:handle:vertical { background: #283142; min-height: 20px; border-radius: 4px; }"
      "QScrollBar:handle:vertical:hover { background: #3b82f6; }"
      "QScrollBar:add-line:vertical, QScrollBar:sub-line:vertical { height: 0px; }"
      "QScrollBar:horizontal { border: none; background: #11141a; height: 8px; margin: 0px; }"
      "QScrollBar:handle:horizontal { background: #283142; min-width: 20px; border-radius: 4px; }"
      "QScrollBar:handle:horizontal:hover { background: #3b82f6; }"
      "QScrollBar:add-line:horizontal, QScrollBar:sub-line:horizontal { width: 0px; }"
      "QSplitter::handle { background-color: #202735; }"
      "QSplitter::handle:hover { background-color: #3b82f6; }"
      "QCheckBox { spacing: 6px; font-size: 11px; color: #cbd5e1; }"
      "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #374151; border-radius: 3px; background: #13161f; }"
      "QCheckBox::indicator:checked { background: #2563eb; border-color: #3b82f6; }"
      "QListWidget { background-color: #131720; border: 1px solid #262f3f; border-radius: 6px; color: #cbd5e1; }"
      "QListWidget::item { padding: 4px 6px; border-radius: 4px; }"
      "QListWidget::item:selected { background-color: #1e3a8a; color: #ffffff; }"
      "QListWidget::item:hover:!selected { background-color: #1c2230; }"
  );

  auto *central = new QWidget(this);
  setCentralWidget(central);
  auto *rootVl = new QVBoxLayout(central);
  rootVl->setContentsMargins(6, 6, 6, 6);
  rootVl->setSpacing(6);

  auto wrapScroll = [](QWidget *w) {
    auto *sc = new QScrollArea;
    sc->setWidgetResizable(true);
    sc->setFrameShape(QFrame::NoFrame);
    sc->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    sc->setWidget(w);
    return sc;
  };

  // ─── Top Header Bar ────────────────────────────────────────────────────────
  auto *headerBar = new QFrame;
  headerBar->setStyleSheet(
      "QFrame { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a202c, stop:1 #131722); "
      "border: 1px solid #283144; border-radius: 6px; padding: 3px 8px; }");
  auto *headerLayout = new QHBoxLayout(headerBar);
  headerLayout->setContentsMargins(4, 2, 4, 2);
  headerLayout->setSpacing(10);

  auto *logoTitleLbl = new QLabel("🔷 TMVisionary 3D Studio");
  logoTitleLbl->setStyleSheet("font-size: 14px; font-weight: bold; color: #60a5fa;");
  auto *verBadge = new QLabel("v4.8 Pro");
  verBadge->setStyleSheet("background-color: #1e3a8a; color: #bfdbfe; font-size: 10px; font-weight: bold; border-radius: 3px; padding: 1px 5px;");

  _camStatusLbl = new QLabel("● Disconnected");
  _camStatusLbl->setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px; padding: 2px 8px; background: #261619; border-radius: 4px;");

  auto *openDataDirBtn = new QPushButton("📁 Thư Mục Dữ Liệu");
  openDataDirBtn->setStyleSheet(
      "QPushButton { background: #1f2737; border: 1px solid #334057; color: #cbd5e1; padding: 4px 10px; border-radius: 4px; }"
      "QPushButton:hover { background: #2b364c; color: #fff; }");
  connect(openDataDirBtn, &QPushButton::clicked, this, &MainWindow::onOpenDataFolderClicked);

  headerLayout->addWidget(logoTitleLbl);
  headerLayout->addWidget(verBadge);
  headerLayout->addStretch(1);
  headerLayout->addWidget(_camStatusLbl);
  headerLayout->addWidget(openDataDirBtn);

  rootVl->addWidget(headerBar);

  // ===========================================================================
  // WORKSPACE 1: 📡 SCAN & 3D POINT CLOUD
  // ===========================================================================

  auto *scanCloudPage = new QWidget;
  auto *scanCloudLayout = new QHBoxLayout(scanCloudPage);
  scanCloudLayout->setContentsMargins(0, 0, 0, 0);
  scanCloudLayout->setSpacing(0);

  auto *cloudSplitter = new QSplitter(Qt::Horizontal);

  // ── Left Sidebar Controls ──────────────────────────────────────────────────
  auto *scanSidebarContent = new QWidget;
  auto *scanSidebarVl = new QVBoxLayout(scanSidebarContent);
  scanSidebarVl->setContentsMargins(6, 6, 8, 6);
  scanSidebarVl->setSpacing(10);

  // 1. Camera & Intrinsics Card
  auto *camGroup = new QGroupBox("📷 Thiết Bị Camera & Hiệu Chuẩn");
  auto *camVl = new QVBoxLayout(camGroup);
  camVl->setContentsMargins(8, 8, 8, 8);
  camVl->setSpacing(6);

  auto *camRow = new QHBoxLayout;
  _camSourceCombo = new QComboBox;
  _camSourceCombo->addItem("Intel RealSense (USB)", 1);
  _camSourceCombo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
  camRow->addWidget(new QLabel("Nguồn:"));
  camRow->addWidget(_camSourceCombo, 1);
  camVl->addLayout(camRow);

  auto *camBtnRow = new QHBoxLayout;
  camBtnRow->setSpacing(6);
  _camConnBtn = new QPushButton("⚡ Kết Nối");
  _camConnBtn->setStyleSheet(
      "QPushButton { background: #166534; color: white; border: 1px solid #22c55e; border-radius: 4px; padding: 5px 10px; font-weight: bold; }"
      "QPushButton:hover { background: #15803d; }"
      "QPushButton:disabled { background: #141720; color: #475569; border-color: #1e2430; }");
  _camDiscBtn = new QPushButton("Ngắt");
  _camDiscBtn->setEnabled(false);
  _camDiscBtn->setFixedWidth(75);
  _camDiscBtn->setStyleSheet(
      "QPushButton { background: #7f1d1d; color: white; border: 1px solid #ef4444; border-radius: 4px; padding: 5px 10px; }"
      "QPushButton:hover { background: #991b1b; }"
      "QPushButton:disabled { background: #141720; color: #475569; border-color: #1e2430; }");
  camBtnRow->addWidget(_camConnBtn, 1);
  camBtnRow->addWidget(_camDiscBtn);
  camVl->addLayout(camBtnRow);

  // Model & Apply Row
  auto *modelRow = new QHBoxLayout;
  _cameraModelCombo = new QComboBox;
  _cameraModelCombo->addItem("d435f", 0);
  _cameraModelCombo->addItem("d435", 1);
  _cameraModelCombo->addItem("d435if", 2);

  _applyIntrBtn = new QPushButton("💾 Áp Dụng");
  _applyIntrBtn->setFixedWidth(85);
  _applyIntrBtn->setStyleSheet(
      "QPushButton { background: #0284c7; color: white; border-radius: 4px; padding: 4px 8px; font-weight: bold; }"
      "QPushButton:hover { background: #0369a1; }");
  modelRow->addWidget(new QLabel("Model:"));
  modelRow->addWidget(_cameraModelCombo, 1);
  modelRow->addWidget(_applyIntrBtn);
  camVl->addLayout(modelRow);

  auto mkDSpin = [&](double val, double lo, double hi) {
    auto *s = new QDoubleSpinBox;
    s->setRange(lo, hi);
    s->setValue(val);
    s->setDecimals(3);
    s->setFixedWidth(78);
    return s;
  };
  _fxSpin = mkDSpin(386.652, 100, 2000);
  _fySpin = mkDSpin(386.652, 100, 2000);
  _cxSpin = mkDSpin(326.089, 0, 1920);
  _cySpin = mkDSpin(240.785, 0, 1080);

  auto *intrGrid = new QGridLayout;
  intrGrid->setContentsMargins(0, 2, 0, 0);
  intrGrid->setSpacing(6);
  intrGrid->addWidget(new QLabel("fx:"), 0, 0);
  intrGrid->addWidget(_fxSpin, 0, 1);
  intrGrid->addWidget(new QLabel("fy:"), 0, 2);
  intrGrid->addWidget(_fySpin, 0, 3);

  intrGrid->addWidget(new QLabel("cx:"), 1, 0);
  intrGrid->addWidget(_cxSpin, 1, 1);
  intrGrid->addWidget(new QLabel("cy:"), 1, 2);
  intrGrid->addWidget(_cySpin, 1, 3);
  intrGrid->setColumnStretch(1, 1);
  intrGrid->setColumnStretch(3, 1);
  camVl->addLayout(intrGrid);

  scanSidebarVl->addWidget(camGroup);

  // 2. Scan Modes Card
  auto *scanModesGroup = new QGroupBox("🚀 Chế Độ Quét & Thu Thập Dữ Liệu");
  auto *scanModesVl = new QVBoxLayout(scanModesGroup);
  scanModesVl->setContentsMargins(6, 8, 6, 8);
  scanModesVl->setSpacing(6);

  auto *scanModeTabs = new QTabWidget;
  scanModeTabs->setDocumentMode(false);

  // Subtab A: Handheld & Free Scan
  auto *handheldTab = new QWidget;
  auto *handheldVl = new QVBoxLayout(handheldTab);
  handheldVl->setContentsMargins(6, 6, 6, 6);
  handheldVl->setSpacing(6);

  _recordIntervalSpin = new QSpinBox;
  _recordIntervalSpin->setRange(50, 5000);
  _recordIntervalSpin->setValue(200);
  _recordIntervalSpin->setSuffix(" ms");
  _recordIntervalSpin->setFixedWidth(75);

  _recordBtn = new QPushButton("🔴 Bắt Đầu Record");
  _recordBtn->setStyleSheet(
      "QPushButton { background: #991b1b; color: white; font-weight: bold; border-radius: 4px; padding: 7px 12px; }"
      "QPushButton:hover { background: #b91c1c; }"
      "QPushButton:checked { background: #ef4444; border: 2px solid #fecaca; }");
  _recordBtn->setCheckable(true);

  auto *recOptRow = new QHBoxLayout;
  recOptRow->addWidget(new QLabel("Chu kỳ ghi:"));
  recOptRow->addWidget(_recordIntervalSpin);
  recOptRow->addStretch(1);

  _vioScanBtn = new QPushButton("🚀 Quét VIO (Free Scan)");
  _vioScanBtn->setStyleSheet(
      "QPushButton { background: #1d4ed8; color: white; font-weight: bold; border-radius: 4px; padding: 6px 10px; }"
      "QPushButton:hover { background: #2563eb; }");

  _replayBtn = new QPushButton("📼 Load Record (Replay)");
  _replayBtn->setStyleSheet(
      "QPushButton { background: #6b21a8; color: white; font-weight: bold; border-radius: 4px; padding: 6px 10px; }"
      "QPushButton:hover { background: #7e22ce; }");

  auto *promptRow = new QHBoxLayout;
  _aiPromptEdit = new QLineEdit("cup");
  _aiPromptEdit->setPlaceholderText("Vật thể tách AI (e.g. cup, mouse, box)");
  promptRow->addWidget(new QLabel("AI Object:"));
  promptRow->addWidget(_aiPromptEdit, 1);

  handheldVl->addWidget(_recordBtn);
  handheldVl->addLayout(recOptRow);
  handheldVl->addWidget(_vioScanBtn);
  handheldVl->addWidget(_replayBtn);
  handheldVl->addLayout(promptRow);

  scanModeTabs->addTab(handheldTab, "Cầm Tay / VIO");

  // Subtab B: Orbital Scan
  auto *orbitalTab = new QWidget;
  auto *orbitalVl = new QVBoxLayout(orbitalTab);
  orbitalVl->setContentsMargins(6, 6, 6, 6);
  orbitalVl->setSpacing(6);

  _scanFramesSpin = new QSpinBox;
  _scanFramesSpin->setRange(1, 50);
  _scanFramesSpin->setValue(5);
  _scanFramesSpin->setFixedWidth(65);

  _scanSettleSpin = new QSpinBox;
  _scanSettleSpin->setRange(50, 5000);
  _scanSettleSpin->setValue(400);
  _scanSettleSpin->setFixedWidth(70);

  _scanIntervalSpin = new QSpinBox;
  _scanIntervalSpin->setRange(50, 2000);
  _scanIntervalSpin->setValue(150);
  _scanIntervalSpin->setFixedWidth(70);

  auto *orbTimingGrid = new QGridLayout;
  orbTimingGrid->setContentsMargins(0, 2, 0, 0);
  orbTimingGrid->setSpacing(6);
  orbTimingGrid->addWidget(new QLabel("Frames/wp:"), 0, 0);
  orbTimingGrid->addWidget(_scanFramesSpin, 0, 1);
  orbTimingGrid->addWidget(new QLabel("Settle (ms):"), 0, 2);
  orbTimingGrid->addWidget(_scanSettleSpin, 0, 3);
  orbTimingGrid->addWidget(new QLabel("Interval (ms):"), 1, 0);
  orbTimingGrid->addWidget(_scanIntervalSpin, 1, 1);
  orbTimingGrid->setColumnStretch(1, 1);
  orbTimingGrid->setColumnStretch(3, 1);
  orbitalVl->addLayout(orbTimingGrid);

  _orbModeBadge = new QLabel("⬤ Mode: Spherical orbit");
  _orbModeBadge->setStyleSheet(
      "QLabel { background: #1a3a5a; color: #7ec8e3; font-weight: bold; border-radius: 4px; padding: 3px 8px; font-size: 11px; }");
  orbitalVl->addWidget(_orbModeBadge);

  _orbitalWidget = new OrbitalScanWidget;
  _orbitalWidget->setMinimumHeight(150);
  orbitalVl->addWidget(_orbitalWidget);

  _startOrbitalBtn = new QPushButton("▶ Bắt Đầu Quét Robot");
  _startOrbitalBtn->setEnabled(false);
  _startOrbitalBtn->setStyleSheet(
      "QPushButton { background: #15803d; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; }"
      "QPushButton:hover { background: #16a34a; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  _stopOrbitalBtn = new QPushButton("⏹ Dừng");
  _stopOrbitalBtn->setEnabled(false);
  _stopOrbitalBtn->setFixedWidth(75);
  _stopOrbitalBtn->setStyleSheet(
      "QPushButton { background: #b91c1c; color: white; border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  _orbitalStatusLbl = new QLabel("Tạo quỹ đạo rồi bấm bắt đầu");
  _orbitalStatusLbl->setStyleSheet("color: #94a3b8; font-size: 10px;");

  _orbitalProgress = new QProgressBar;
  _orbitalProgress->setRange(0, 100);
  _orbitalProgress->setValue(0);
  _orbitalProgress->setTextVisible(true);
  _orbitalProgress->setFixedHeight(12);
  _orbitalProgress->setStyleSheet(
      "QProgressBar { border: none; background: #1e2430; border-radius: 3px; font-size: 9px; }"
      "QProgressBar::chunk { background: #16a34a; border-radius: 3px; }");

  auto *orbBtnRow = new QHBoxLayout;
  orbBtnRow->addWidget(_startOrbitalBtn, 1);
  orbBtnRow->addWidget(_stopOrbitalBtn);
  orbitalVl->addLayout(orbBtnRow);
  orbitalVl->addWidget(_orbitalStatusLbl);
  orbitalVl->addWidget(_orbitalProgress);

  scanModeTabs->addTab(orbitalTab, "Robot (Orbital)");

  scanModesVl->addWidget(scanModeTabs);
  scanSidebarVl->addWidget(scanModesGroup);

  // 3. Point Cloud & Masking Card
  auto *filterGroup = new QGroupBox("☁ Bộ Lọc & Mask Mây Điểm");
  auto *filterVl = new QVBoxLayout(filterGroup);
  filterVl->setContentsMargins(8, 8, 8, 8);
  filterVl->setSpacing(6);

  _accumStatsLbl = new QLabel("0 voxels  |  0 pts");
  _accumStatsLbl->setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 11px;");

  _voxelSizeSpin = new QDoubleSpinBox;
  _voxelSizeSpin->setRange(1.0, 50.0);
  _voxelSizeSpin->setValue(15.0);
  _voxelSizeSpin->setSingleStep(1.0);
  _voxelSizeSpin->setDecimals(1);
  _voxelSizeSpin->setSuffix(" mm");
  _voxelSizeSpin->setFixedWidth(75);
  _voxelSizeSpin->setToolTip("Kích thước voxel — áp dụng khi quét tích lũy");

  _clearCloudBtn = new QPushButton("🗑 Clear");
  _clearCloudBtn->setFixedWidth(68);
  _clearCloudBtn->setStyleSheet(
      "QPushButton { background: #991b1b; color: white; border-radius: 4px; padding: 4px 8px; }"
      "QPushButton:hover { background: #b91c1c; }");

  _showFovCheck = new QCheckBox("Hiện FOV / Quỹ Đạo");
  _showFovCheck->setChecked(true);

  auto *statRow = new QHBoxLayout;
  statRow->addWidget(_accumStatsLbl, 1);
  statRow->addWidget(_clearCloudBtn);
  filterVl->addLayout(statRow);

  auto *voxelRow = new QHBoxLayout;
  voxelRow->addWidget(new QLabel("Voxel:"));
  voxelRow->addWidget(_voxelSizeSpin);
  voxelRow->addStretch(1);
  voxelRow->addWidget(_showFovCheck);
  filterVl->addLayout(voxelRow);

  _scanMaskCheck = new QCheckBox("🎯 Scan Mask Bounding (Hình Cầu)");
  auto mkMaskSpin = [&](double val, double lo, double hi) {
    auto *s = new QDoubleSpinBox;
    s->setRange(lo, hi);
    s->setValue(val);
    s->setSuffix(" mm");
    s->setDecimals(0);
    s->setFixedWidth(68);
    s->setEnabled(false);
    return s;
  };
  _maskCenterXSpin = mkMaskSpin(0, -5000, 5000);
  _maskCenterYSpin = mkMaskSpin(0, -5000, 5000);
  _maskCenterZSpin = mkMaskSpin(500, -5000, 5000);
  _maskRadiusSpin  = mkMaskSpin(300, 10, 5000);

  auto *maskGrid = new QGridLayout;
  maskGrid->setContentsMargins(0, 0, 0, 0);
  maskGrid->setSpacing(4);
  maskGrid->addWidget(new QLabel("Cx:"), 0, 0);
  maskGrid->addWidget(_maskCenterXSpin, 0, 1);
  maskGrid->addWidget(new QLabel("Cy:"), 0, 2);
  maskGrid->addWidget(_maskCenterYSpin, 0, 3);
  maskGrid->addWidget(new QLabel("Cz:"), 1, 0);
  maskGrid->addWidget(_maskCenterZSpin, 1, 1);
  maskGrid->addWidget(new QLabel("R:"),  1, 2);
  maskGrid->addWidget(_maskRadiusSpin,  1, 3);
  maskGrid->setColumnStretch(1, 1);
  maskGrid->setColumnStretch(3, 1);

  filterVl->addWidget(_scanMaskCheck);
  filterVl->addLayout(maskGrid);
  scanSidebarVl->addWidget(filterGroup);

  // 4. Point Cloud SLAM & Export Card
  auto *slamGroup = new QGroupBox("⚡ Tối Ưu Hóa & Xuất Mây Điểm");
  auto *slamVl = new QVBoxLayout(slamGroup);
  slamVl->setContentsMargins(8, 8, 8, 8);
  slamVl->setSpacing(6);

  _runIcpBtn = new QPushButton("🐍 Run SLAM (Tối Ưu Mây Điểm)");
  _runIcpBtn->setStyleSheet(
      "QPushButton { background: #15803d; color: white; font-weight: bold; border-radius: 4px; padding: 8px 10px; }"
      "QPushButton:hover { background: #16a34a; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  _exportCloudBtn = new QPushButton("💾 Xuất Mây Điểm (PLY)");
  _exportCloudBtn->setEnabled(false);
  _exportCloudBtn->setStyleSheet(
      "QPushButton { background: #0369a1; color: white; font-weight: bold; border-radius: 4px; padding: 8px 10px; }"
      "QPushButton:hover { background: #0284c7; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  auto *goToMeshBtn = new QPushButton("🧱 Sang Tab Mesh Studio Tách Vật Thể →");
  goToMeshBtn->setStyleSheet(
      "QPushButton { background: #1e293b; color: #93c5fd; font-weight: bold; border: 1px solid #334155; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #1e3a8a; color: white; border-color: #3b82f6; }");
  connect(goToMeshBtn, &QPushButton::clicked, this, [this]() {
    _mainTabs->setCurrentIndex(1);
  });

  slamVl->addWidget(_runIcpBtn);
  slamVl->addWidget(_exportCloudBtn);
  slamVl->addWidget(goToMeshBtn);
  scanSidebarVl->addWidget(slamGroup);

  // Hidden buttons kept for signal wiring compatibility
  _exportMeshBtn = new QPushButton(scanSidebarContent);
  _exportMeshBtn->setVisible(false);
  _viewMeshBtn = new QPushButton(scanSidebarContent);
  _viewMeshBtn->setVisible(false);

  scanSidebarVl->addStretch(1);

  auto *scanSidebarScroll = wrapScroll(scanSidebarContent);
  scanSidebarScroll->setMinimumWidth(320);
  scanSidebarScroll->setMaximumWidth(550);

  // Hidden elements needed for signal wiring
  _accumCheck = new QCheckBox(scanSidebarContent);
  _accumCheck->setVisible(false);
  _icpCheck = new QCheckBox(scanSidebarContent);
  _icpCheck->setVisible(false);
  _orbGroup = new QGroupBox(scanSidebarContent);
  _orbGroup->setVisible(false);

  // ── Right Main Viewport Area ───────────────────────────────────────────────
  auto *cloudViewportArea = new QWidget;
  auto *cloudVl = new QVBoxLayout(cloudViewportArea);
  cloudVl->setContentsMargins(4, 4, 4, 4);
  cloudVl->setSpacing(4);

  // Viewport Control Toolbar
  auto *cloudToolbar = new QFrame;
  cloudToolbar->setStyleSheet(
      "QFrame { background: #181d27; border: 1px solid #283142; border-radius: 6px; padding: 3px 6px; }");
  auto *ctbLayout = new QHBoxLayout(cloudToolbar);
  ctbLayout->setContentsMargins(4, 2, 4, 2);
  ctbLayout->setSpacing(6);

  auto mkTbBtn = [](const QString &txt, const QString &tooltip) {
    auto *b = new QPushButton(txt);
    b->setToolTip(tooltip);
    b->setStyleSheet(
        "QPushButton { background: #202735; color: #e2e8f0; border: 1px solid #333f54; border-radius: 4px; padding: 3px 8px; font-size: 11px; }"
        "QPushButton:hover { background: #2a3447; border-color: #3b82f6; }");
    return b;
  };

  auto *pcResetBtn  = mkTbBtn("🔄 Reset Cam", "Khôi phục góc nhìn mặc định");
  auto *pcCenterBtn = mkTbBtn("🎯 Center", "Căn giữa tâm mây điểm");
  auto *pcTopBtn    = mkTbBtn("🔼 Top", "Nhìn từ trên xuống (Top view)");
  auto *pcFrontBtn  = mkTbBtn("◀ Front", "Nhìn chính diện (Front view)");
  auto *pcSideBtn   = mkTbBtn("▶ Side", "Nhìn ngang cạnh (Side view)");
  auto *pcIsoBtn    = mkTbBtn("📐 Iso", "Góc nhìn phối cảnh Isometric");

  ctbLayout->addWidget(pcResetBtn);
  ctbLayout->addWidget(pcCenterBtn);
  ctbLayout->addWidget(pcTopBtn);
  ctbLayout->addWidget(pcFrontBtn);
  ctbLayout->addWidget(pcSideBtn);
  ctbLayout->addWidget(pcIsoBtn);

  auto *sep1 = new QFrame; sep1->setFrameShape(QFrame::VLine); sep1->setStyleSheet("color:#333f54;");
  ctbLayout->addWidget(sep1);

  _pointSizeSpin = new QDoubleSpinBox;
  _pointSizeSpin->setRange(1.0, 10.0);
  _pointSizeSpin->setValue(2.0);
  _pointSizeSpin->setSingleStep(0.5);
  _pointSizeSpin->setSuffix(" px");
  _pointSizeSpin->setFixedWidth(66);
  _pointSizeSpin->setToolTip("Kích thước hạt mây điểm");
  ctbLayout->addWidget(new QLabel("Pt Size:"));
  ctbLayout->addWidget(_pointSizeSpin);

  _gridCheck = new QCheckBox("Lưới Tọa Độ");
  _gridCheck->setChecked(true);
  ctbLayout->addWidget(_gridCheck);

  auto *sep2 = new QFrame; sep2->setFrameShape(QFrame::VLine); sep2->setStyleSheet("color:#333f54;");
  ctbLayout->addWidget(sep2);

  _cloudSnapshotBtn = new QPushButton("📸 Snapshot");
  _cloudSnapshotBtn->setToolTip("Chụp ảnh độ nét cao màn hình 3D hiện tại");
  _cloudSnapshotBtn->setStyleSheet(
      "QPushButton { background: #1e3a8a; color: white; border: 1px solid #3b82f6; border-radius: 4px; padding: 3px 10px; font-weight: bold; }"
      "QPushButton:hover { background: #2563eb; }");
  ctbLayout->addWidget(_cloudSnapshotBtn);
  ctbLayout->addStretch(1);

  _cloudView = new PointCloudWidget;
  _cloudView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  _cloudView->setMinimumSize(400, 300);

  _cloudInfoLbl = new QLabel("3D View — Bắt đầu quét hoặc load dữ liệu để tích lũy mây điểm");
  _cloudInfoLbl->setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 4px;");

  cloudVl->addWidget(cloudToolbar);
  cloudVl->addWidget(_cloudView, 1);
  cloudVl->addWidget(_cloudInfoLbl);

  cloudSplitter->addWidget(scanSidebarScroll);
  cloudSplitter->addWidget(cloudViewportArea);
  cloudSplitter->setStretchFactor(0, 0);
  cloudSplitter->setStretchFactor(1, 1);
  cloudSplitter->setCollapsible(0, false);
  cloudSplitter->setCollapsible(1, false);
  cloudSplitter->setSizes({400, 980});

  scanCloudLayout->addWidget(cloudSplitter);

  // Wire viewport toolbar buttons
  connect(pcResetBtn,  &QPushButton::clicked, this, [this]() { _cloudView->setViewPreset(0); });
  connect(pcCenterBtn, &QPushButton::clicked, this, [this]() { _cloudView->autoCenter(); });
  connect(pcTopBtn,    &QPushButton::clicked, this, [this]() { _cloudView->setViewPreset(1); });
  connect(pcFrontBtn,  &QPushButton::clicked, this, [this]() { _cloudView->setViewPreset(2); });
  connect(pcSideBtn,   &QPushButton::clicked, this, [this]() { _cloudView->setViewPreset(3); });
  connect(pcIsoBtn,    &QPushButton::clicked, this, [this]() { _cloudView->setViewPreset(4); });
  connect(_pointSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this](double val) {
    _cloudView->setPointSize(float(val));
  });
  connect(_gridCheck, &QCheckBox::toggled, _cloudView, &PointCloudWidget::setShowGrid);
  connect(_cloudSnapshotBtn, &QPushButton::clicked, this, &MainWindow::onCloudSnapshotClicked);

  // ===========================================================================
  // WORKSPACE 2: 🧱 MESH STUDIO & CLEANUP
  // ===========================================================================

  auto *meshStudioPage = new QWidget;
  auto *meshStudioLayout = new QHBoxLayout(meshStudioPage);
  meshStudioLayout->setContentsMargins(0, 0, 0, 0);
  meshStudioLayout->setSpacing(0);

  auto *meshSplitter = new QSplitter(Qt::Horizontal);

  // ── Left/Center Mesh Viewport ──────────────────────────────────────────────
  auto *meshViewportArea = new QWidget;
  auto *meshViewportVl = new QVBoxLayout(meshViewportArea);
  meshViewportVl->setContentsMargins(4, 4, 4, 4);
  meshViewportVl->setSpacing(4);

  auto *meshToolbar = new QFrame;
  meshToolbar->setStyleSheet(
      "QFrame { background: #181d27; border: 1px solid #283142; border-radius: 6px; padding: 3px 6px; }");
  auto *mtbLayout = new QHBoxLayout(meshToolbar);
  mtbLayout->setContentsMargins(4, 2, 4, 2);
  mtbLayout->setSpacing(6);

  auto *mResetBtn  = mkTbBtn("🔄 Reset Cam", "Khôi phục góc nhìn");
  auto *mCenterBtn = mkTbBtn("🎯 Center", "Căn giữa vật thể");
  auto *mTopBtn    = mkTbBtn("🔼 Top", "Góc nhìn từ đỉnh");
  auto *mFrontBtn  = mkTbBtn("◀ Front", "Góc nhìn chính diện");
  auto *mIsoBtn    = mkTbBtn("📐 Iso", "Góc nhìn Isometric");

  mtbLayout->addWidget(mResetBtn);
  mtbLayout->addWidget(mCenterBtn);
  mtbLayout->addWidget(mTopBtn);
  mtbLayout->addWidget(mFrontBtn);
  mtbLayout->addWidget(mIsoBtn);

  auto *msep1 = new QFrame; msep1->setFrameShape(QFrame::VLine); msep1->setStyleSheet("color:#333f54;");
  mtbLayout->addWidget(msep1);

  _meshWireframeBtn = new QPushButton("🔲 Wireframe");
  _meshWireframeBtn->setCheckable(true);
  _meshWireframeBtn->setToolTip("Bật/tắt chế độ hiển thị khung lưới đa giác");
  mtbLayout->addWidget(_meshWireframeBtn);

  _meshWhiteBgCheck = new QCheckBox("Nền Sáng");
  _meshWhiteBgCheck->setToolTip("Chuyển đổi màu nền Trắng / Tối");
  mtbLayout->addWidget(_meshWhiteBgCheck);

  auto *msep2 = new QFrame; msep2->setFrameShape(QFrame::VLine); msep2->setStyleSheet("color:#333f54;");
  mtbLayout->addWidget(msep2);

  _meshSnapshotBtn = new QPushButton("📸 Snapshot");
  _meshSnapshotBtn->setToolTip("Lưu ảnh render Mesh chất lượng cao");
  _meshSnapshotBtn->setStyleSheet(
      "QPushButton { background: #1e3a8a; color: white; border: 1px solid #3b82f6; border-radius: 4px; padding: 3px 10px; font-weight: bold; }"
      "QPushButton:hover { background: #2563eb; }");
  mtbLayout->addWidget(_meshSnapshotBtn);
  mtbLayout->addStretch(1);

  _meshView = new MeshViewerWidget;
  _meshView->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  _meshView->setMinimumSize(400, 300);

  _meshInfoLbl = new QLabel("Chưa tải mesh nào");
  _meshInfoLbl->setStyleSheet("color: #94a3b8; font-size: 11px; padding: 2px 4px;");

  meshViewportVl->addWidget(meshToolbar);
  meshViewportVl->addWidget(_meshView, 1);
  meshViewportVl->addWidget(_meshInfoLbl);

  // ── Right Mesh Sidebar ─────────────────────────────────────────────────────
  auto *meshSidebarContent = new QWidget;
  auto *meshSidebarVl = new QVBoxLayout(meshSidebarContent);
  meshSidebarVl->setContentsMargins(6, 6, 8, 6);
  meshSidebarVl->setSpacing(10);

  // 1. AI Mesh Reconstruction
  auto *meshGenGroup = new QGroupBox("⚙ Tái Tạo Mesh 3D (AI & SLAM)");
  auto *meshGenVl = new QVBoxLayout(meshGenGroup);
  meshGenVl->setContentsMargins(8, 8, 8, 8);
  meshGenVl->setSpacing(6);

  _buildMeshBtn2 = new QPushButton("🐍 Xây Mesh Object (AI)");
  _buildMeshBtn2->setStyleSheet(
      "QPushButton { background: #166534; color: white; font-weight: bold; border-radius: 5px; padding: 8px 12px; }"
      "QPushButton:hover { background: #15803d; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  _meshPoissonBtn = new QPushButton("📦 Bọc Mesh (Poisson Recon)");
  _meshPoissonBtn->setStyleSheet(
      "QPushButton { background: #c2410c; color: white; font-weight: bold; border-radius: 5px; padding: 8px 12px; }"
      "QPushButton:hover { background: #ea580c; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  _killProcessesBtn = new QPushButton("🛑 Dừng / Kill Tiến Trình SLAM Ngầm");
  _killProcessesBtn->setStyleSheet(
      "QPushButton { background: #991b1b; color: white; font-weight: bold; border-radius: 5px; padding: 8px 12px; }"
      "QPushButton:hover { background: #dc2626; }");

  meshGenVl->addWidget(_buildMeshBtn2);
  meshGenVl->addWidget(_meshPoissonBtn);
  meshGenVl->addWidget(_killProcessesBtn);
  meshSidebarVl->addWidget(meshGenGroup);

  // 2. Mesh Cleanup & Editing
  auto *meshEditGroup = new QGroupBox("✂ Công Cụ Chỉnh Sửa & Làm Sạch");
  auto *meshEditVl = new QVBoxLayout(meshEditGroup);
  meshEditVl->setContentsMargins(8, 8, 8, 8);
  meshEditVl->setSpacing(6);

  _meshAutoBtn = new QPushButton("🧹 Tẩy Tự Động (Lọc Nhiễu Rời)");
  _meshAutoBtn->setStyleSheet(
      "QPushButton { background: #854d0e; color: white; font-weight: bold; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #a16207; }");

  _meshEraserBtn = new QPushButton("✂ Tẩy Thủ Công (Vẽ Vùng Chọn)");
  _meshEraserBtn->setCheckable(true);
  _meshEraserBtn->setStyleSheet(
      "QPushButton { background: #86198f; color: white; font-weight: bold; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #a21caf; }"
      "QPushButton:checked { background: #c026d3; border: 2px solid #f5d0fe; }");

  _meshUndoBtn = new QPushButton("↩ Hoàn Tác (Undo)");
  _meshUndoBtn->setStyleSheet(
      "QPushButton { background: #334155; color: white; font-weight: bold; border-radius: 4px; padding: 6px 10px; }"
      "QPushButton:hover { background: #475569; }");

  _meshSaveBtn = new QPushButton("💾 Lưu Mesh PLY");
  _meshSaveBtn->setStyleSheet(
      "QPushButton { background: #0369a1; color: white; font-weight: bold; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #0284c7; }");

  _exportObjBtn = new QPushButton("📤 Xuất File OBJ (CAD / 3D)");
  _exportObjBtn->setStyleSheet(
      "QPushButton { background: #3f6212; color: white; font-weight: bold; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #4d7c0f; }");

  meshEditVl->addWidget(_meshAutoBtn);
  meshEditVl->addWidget(_meshEraserBtn);
  meshEditVl->addWidget(_meshUndoBtn);
  meshEditVl->addWidget(_meshSaveBtn);
  meshEditVl->addWidget(_exportObjBtn);
  meshSidebarVl->addWidget(meshEditGroup);

  // 3. File Management & Recent List
  auto *meshFileGroup = new QGroupBox("📂 Quản Lý File & Debug Stages");
  auto *meshFileVl = new QVBoxLayout(meshFileGroup);
  meshFileVl->setContentsMargins(8, 8, 8, 8);
  meshFileVl->setSpacing(6);

  _loadMeshBtn = new QPushButton("📂 Mở File Mesh PLY/OBJ...");
  _loadMeshBtn->setStyleSheet(
      "QPushButton { background: #1e3a8a; color: white; font-weight: bold; border-radius: 4px; padding: 7px 10px; }"
      "QPushButton:hover { background: #2563eb; }");

  _meshFileList = new QListWidget;
  _meshFileList->setMinimumHeight(150);

  _meshRemoveBtn = new QPushButton("❌ Bỏ Khỏi Danh Sách");
  _meshRemoveBtn->setEnabled(false);
  _meshRemoveBtn->setStyleSheet(
      "QPushButton { background: #7f1d1d; color: white; border-radius: 4px; padding: 5px 10px; }"
      "QPushButton:hover { background: #991b1b; }"
      "QPushButton:disabled { background: #141720; color: #475569; }");

  meshFileVl->addWidget(_loadMeshBtn);
  meshFileVl->addWidget(new QLabel("Danh sách mesh gần đây:"));
  meshFileVl->addWidget(_meshFileList, 1);
  meshFileVl->addWidget(_meshRemoveBtn);
  meshSidebarVl->addWidget(meshFileGroup);

  meshSidebarVl->addStretch(1);

  auto *meshSidebarScroll = wrapScroll(meshSidebarContent);
  meshSidebarScroll->setMinimumWidth(280);
  meshSidebarScroll->setMaximumWidth(360);

  meshSplitter->addWidget(meshViewportArea);
  meshSplitter->addWidget(meshSidebarScroll);
  meshSplitter->setStretchFactor(0, 1);
  meshSplitter->setStretchFactor(1, 0);
  meshSplitter->setCollapsible(0, false);
  meshSplitter->setCollapsible(1, false);

  meshStudioLayout->addWidget(meshSplitter);

  // Wire Mesh Viewport toolbar buttons
  connect(mResetBtn,  &QPushButton::clicked, this, [this]() { _meshView->setViewPreset(0); });
  connect(mCenterBtn, &QPushButton::clicked, this, [this]() { _meshView->autoCenter(); });
  connect(mTopBtn,    &QPushButton::clicked, this, [this]() { _meshView->setViewPreset(1); });
  connect(mFrontBtn,  &QPushButton::clicked, this, [this]() { _meshView->setViewPreset(2); });
  connect(mIsoBtn,    &QPushButton::clicked, this, [this]() { _meshView->setViewPreset(4); });
  connect(_meshWireframeBtn, &QPushButton::toggled, _meshView, &MeshViewerWidget::setWireframeMode);
  connect(_meshSnapshotBtn,  &QPushButton::clicked, this, &MainWindow::onMeshSnapshotClicked);

  // ===========================================================================
  // ASSEMBLE MAIN WORKSPACE TABS
  // ===========================================================================

  _mainTabs = new QTabWidget;
  _mainTabs->setDocumentMode(true);
  _mainTabs->addTab(scanCloudPage,  "📡 Scan & 3D Point Cloud");
  _mainTabs->addTab(meshStudioPage, "🧱 Mesh Studio & Cleanup");

  // ===========================================================================
  // TERMINAL & LOGGING DRAWER
  // ===========================================================================

  _logEdit = new QPlainTextEdit;
  _logEdit->setReadOnly(true);
  _logEdit->setMaximumBlockCount(LOG_MAX_LINES + 50);
  _logEdit->setStyleSheet(
      "QPlainTextEdit { background: #0c0f16; color: #94a3b8; "
      "font-family: 'Consolas', 'DejaVu Sans Mono', monospace; font-size: 11px; "
      "border: 1px solid #1e2535; border-radius: 4px; padding: 4px; }");

  auto *logWidget = new QWidget;
  auto *logLayout = new QVBoxLayout(logWidget);
  logLayout->setContentsMargins(0, 2, 0, 0);
  logLayout->setSpacing(2);

  auto *logHeaderLayout = new QHBoxLayout;
  logHeaderLayout->setContentsMargins(6, 2, 6, 2);

  auto *logTitle = new QLabel("🖥 Terminal Output & Diagnostics");
  logTitle->setStyleSheet("color: #60a5fa; font-weight: bold; font-size: 11px;");

  _logCountLbl = new QLabel("0 dòng");
  _logCountLbl->setStyleSheet("color: #64748b; font-size: 10px; margin-left: 8px;");

  auto *copyLogBtn = new QPushButton("📋 Copy Log");
  copyLogBtn->setStyleSheet(
      "QPushButton { background: #1e2430; border: 1px solid #333d52; color: #cbd5e1; padding: 2px 8px; border-radius: 3px; font-size: 11px; }"
      "QPushButton:hover { background: #283142; color: #fff; }");
  connect(copyLogBtn, &QPushButton::clicked, this, [this]() {
    QApplication::clipboard()->setText(_logEdit->toPlainText());
    statusBar()->showMessage("Đã sao chép toàn bộ log vào clipboard!", 3000);
  });

  auto *clearBtn = new QPushButton("🗑 Xóa Log");
  clearBtn->setStyleSheet(
      "QPushButton { background: #1e2430; border: 1px solid #333d52; color: #cbd5e1; padding: 2px 8px; border-radius: 3px; font-size: 11px; }"
      "QPushButton:hover { background: #283142; color: #fff; }");
  connect(clearBtn, &QPushButton::clicked, _logEdit, &QPlainTextEdit::clear);

  logHeaderLayout->addWidget(logTitle);
  logHeaderLayout->addWidget(_logCountLbl);
  logHeaderLayout->addStretch(1);
  logHeaderLayout->addWidget(copyLogBtn);
  logHeaderLayout->addWidget(clearBtn);

  logLayout->addLayout(logHeaderLayout);
  logLayout->addWidget(_logEdit);

  // Splitter: Workspace on top, Terminal on bottom
  auto *mainSplitter = new QSplitter(Qt::Vertical);
  mainSplitter->addWidget(_mainTabs);
  mainSplitter->addWidget(logWidget);
  mainSplitter->setStretchFactor(0, 5);
  mainSplitter->setStretchFactor(1, 1);
  mainSplitter->setCollapsible(1, false);

  rootVl->addWidget(mainSplitter, 1);
}

// ─── Signal wiring
// ────────────────────────────────────────────────────────────

void MainWindow::wireSelf() {
  // AppController → MainWindow
  connect(_ctrl, &AppController::geometryFrameReady, this,
          &MainWindow::onGeometryFrameReady);
  connect(_ctrl, &AppController::workerError, this, &MainWindow::onWorkerError);
  connect(_ctrl, &AppController::logMessage, this, &MainWindow::appendLog);
  connect(_ctrl, &AppController::cameraConnectionChanged, this,
          &MainWindow::onCameraConnectionChanged);
  connect(_ctrl, &AppController::scanProgressUpdated, this,
          &MainWindow::onScanProgressUpdated);
  connect(_ctrl, &AppController::scanFinished, this,
          &MainWindow::onScanFinished);
  connect(_ctrl, &AppController::recordingStateChanged, this,
          &MainWindow::onRecordingStateChanged);

  // Camera trajectory → PointCloudWidget (SLAM-like visualization)
  connect(_ctrl, &AppController::cameraTrajectoryUpdated,
          _cloudView, &PointCloudWidget::onCameraTrajectoryUpdated);

  // ScanEngine → PointCloudWidget
  connect(_ctrl->scanEngine(), &ScanEngine::partialUpdateReady, _cloudView,
          &PointCloudWidget::onDeltaReady);
  connect(_ctrl->scanEngine(), &ScanEngine::objectUpdated, this,
          [this](const Object3D &obj) {
              _cloudView->onFullRebuildReady(obj.pointCloud);
          });
  connect(_ctrl->scanEngine(), &ScanEngine::statsUpdated, this,
          &MainWindow::onAccumStatsUpdated);

  // Camera Model Selection
  connect(_cameraModelCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int index) {
      if (index == 0) { // d435f
          _fxSpin->setValue(386.652);
          _fySpin->setValue(386.652);
          _cxSpin->setValue(326.089);
          _cySpin->setValue(240.785);
      } else if (index == 1) { // d435
          _fxSpin->setValue(389.142);
          _fySpin->setValue(389.142);
          _cxSpin->setValue(322.228);
          _cySpin->setValue(241.031);
      } else if (index == 2) { // d435if
          _fxSpin->setValue(389.242);
          _fySpin->setValue(389.242);
          _cxSpin->setValue(322.072);
          _cySpin->setValue(237.180);
      }
      pushIntrinsics(); // apply immediately when changed
  });

  // Camera UI
  connect(_camConnBtn, &QPushButton::clicked, this,
          &MainWindow::onCameraConnectClicked);
  connect(_camDiscBtn, &QPushButton::clicked, this,
          &MainWindow::onCameraDisconnectClicked);

  // Robot UI

  // Calibration
  connect(_applyIntrBtn, &QPushButton::clicked, this,
          [this] { pushIntrinsics(); });

  // 3D view controls
  connect(_accumCheck, &QCheckBox::toggled, this, [this](bool checked) {
    _cloudView->setAccumulate(checked);
    _exportCloudBtn->setEnabled(checked); // Simple check
    appendLog(QString("Cloud Accumulator: %1").arg(checked ? "ON" : "OFF"));
  });

  connect(_showFovCheck, &QCheckBox::toggled, _cloudView, &PointCloudWidget::setShowFov);

  connect(_icpCheck, &QCheckBox::toggled, this, [this](bool checked) {
    _ctrl->scanEngine()->setIcpEnabled(checked);
    appendLog(QString("ICP Alignment: %1").arg(checked ? "ON" : "OFF"));
  });

  connect(_clearCloudBtn, &QPushButton::clicked, this, [this]() {
    _ctrl->scanEngine()->clear();
    _cloudView->clearAccumulated();
    _exportCloudBtn->setEnabled(false);
    _voxelSizeSpin->setEnabled(true);
    appendLog(QString("Cleared — voxel size: %1 mm")
                  .arg(_voxelSizeSpin->value(), 0, 'f', 1));
  });

  connect(_runIcpBtn, &QPushButton::clicked, this, &MainWindow::onRunIcpClicked);
  connect(_exportMeshBtn, &QPushButton::clicked, this, &MainWindow::onExportMeshClicked);
  connect(_viewMeshBtn, &QPushButton::clicked, this, &MainWindow::onViewMeshClicked);

  connect(_voxelSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, [this](double v) { _ctrl->scanEngine()->setVoxelSize(float(v)); });
  connect(_exportCloudBtn, &QPushButton::clicked, this,
          &MainWindow::exportPointCloudPLY);

  connect(_scanMaskCheck, &QCheckBox::toggled, this, [this](bool on) {
    _maskCenterXSpin->setEnabled(on);
    _maskCenterYSpin->setEnabled(on);
    _maskCenterZSpin->setEnabled(on);
    _maskRadiusSpin->setEnabled(on);
    float cx = float(_maskCenterXSpin->value());
    float cy = float(_maskCenterYSpin->value());
    float cz = float(_maskCenterZSpin->value());
    float r = float(_maskRadiusSpin->value());
    _ctrl->scanEngine()->setScanMask(on, cx, cy, cz, r);
    _cloudView->setScanMask(on, cx, cy, cz, r);
  });
  auto pushMask = [this]() {
    if (!_scanMaskCheck->isChecked())
      return;
    float cx = float(_maskCenterXSpin->value());
    float cy = float(_maskCenterYSpin->value());
    float cz = float(_maskCenterZSpin->value());
    float r = float(_maskRadiusSpin->value());
    _ctrl->scanEngine()->setScanMask(true, cx, cy, cz, r);
    _cloudView->setScanMask(true, cx, cy, cz, r);
  };
  connect(_maskCenterXSpin,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [pushMask](double) { pushMask(); });
  connect(_maskCenterYSpin,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [pushMask](double) { pushMask(); });
  connect(_maskCenterZSpin,
          QOverload<double>::of(&QDoubleSpinBox::valueChanged), this,
          [pushMask](double) { pushMask(); });
  connect(_maskRadiusSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
          this, [pushMask](double) { pushMask(); });

  connect(_recordBtn, &QPushButton::clicked, this, &MainWindow::onRecordButtonClicked);

  connect(_killProcessesBtn, &QPushButton::clicked, this, &MainWindow::onKillProcessesClicked);

  connect(_vioScanBtn, &QPushButton::clicked, this, [this]() {
    if (_vioScanBtn->text().contains("Dừng")) {
      _ctrl->stopVioScan();
      _vioScanBtn->setText("🚀 Quét VIO (Free Scan)");
      _vioScanBtn->setStyleSheet(
          "QPushButton { background-color: #2196F3; color: white; "
          "font-weight: bold; border-radius: 4px; padding: 6px 12px; }"
          "QPushButton:hover { background-color: #1976D2; }");
    } else {
      _ctrl->setAiPrompt(_aiPromptEdit->text());
      _ctrl->startVioScan();
      _vioScanBtn->setText("⏹ Dừng Quét VIO");
      _vioScanBtn->setStyleSheet(
          "QPushButton { background-color: #F44336; color: white; "
          "font-weight: bold; border-radius: 4px; padding: 6px 12px; }"
          "QPushButton:hover { background-color: #D32F2F; }");
    }
  });
  
  connect(_replayBtn, &QPushButton::clicked, this, [this]() {
      QString dir = QFileDialog::getExistingDirectory(this, "Select Record Directory", QCoreApplication::applicationDirPath() + "/records");
      if (!dir.isEmpty()) {
          _ctrl->setAiPrompt(_aiPromptEdit->text());
          _ctrl->startReplay(dir);
      }
  });

  // Scan timing → OrbitalScanWidget
  connect(_scanFramesSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int v) {
            _orbitalWidget->setTimingParams(v, _scanSettleSpin->value());
          });
  connect(_scanSettleSpin, QOverload<int>::of(&QSpinBox::valueChanged), this,
          [this](int v) {
            _orbitalWidget->setTimingParams(_scanFramesSpin->value(), v);
          });

  // Robot Debug

  // Orbital scan
  connect(_orbitalWidget, &OrbitalScanWidget::waypointsReady, this,
          &MainWindow::onOrbitalWaypointsReady);
  connect(_orbitalWidget, &OrbitalScanWidget::logMessage, this,
          &MainWindow::appendLog);
  connect(
      _orbitalWidget, &OrbitalScanWidget::scanModeChanged, this,
      [this](const QString &mode) {
        const bool spherical = (mode == "Spherical");
        _orbGroup->setTitle(spherical ? "Orbital Scan — Spherical Trajectory"
                                      : "Orbital Scan — Random 6-DOF");
        _orbModeBadge->setText(spherical ? "⬤  Mode: Spherical orbit"
                                         : "⬤  Mode: Random 6-DOF");
        _orbModeBadge->setStyleSheet(
            spherical
                ? "QLabel{background:#1a3a5a;color:#7ec8e3;font-weight:bold;"
                  "border-radius:4px;padding:3px 10px;font-size:12px;}"
                : "QLabel{background:#3a2a1a;color:#ffd6a5;font-weight:bold;"
                  "border-radius:4px;padding:3px 10px;font-size:12px;}");
        appendLog(QString("Orbital mode → %1").arg(mode));
      });
  connect(_startOrbitalBtn, &QPushButton::clicked, this,
          &MainWindow::onStartOrbitalScanClicked);
  connect(_stopOrbitalBtn, &QPushButton::clicked, this, [this] {
    _ctrl->stopScanSequence();
    setScanRunningState(false);
    _orbitalStatusLbl->setText("Stopped");
  });

  // ── Mesh Builder ─────────────────────────────────────────────────────────
  connect(_buildMeshBtn2, &QPushButton::clicked, this, &MainWindow::onBuildMeshClicked);
  connect(_loadMeshBtn,   &QPushButton::clicked, this, &MainWindow::onViewLoadMeshClicked);
  connect(_exportObjBtn,  &QPushButton::clicked, this, &MainWindow::onMeshExportObj);
  connect(_meshAutoBtn,   &QPushButton::clicked, this, &MainWindow::onMeshAutoClean);
  connect(_meshUndoBtn,   &QPushButton::clicked, this, &MainWindow::onMeshUndo);
  connect(_meshSaveBtn,   &QPushButton::clicked, this, &MainWindow::onMeshSave);
  connect(_meshPoissonBtn,&QPushButton::clicked, this, &MainWindow::onMeshPoissonRecon);
  connect(_meshRemoveBtn, &QPushButton::clicked, this, &MainWindow::onMeshRemoveFile);
  connect(_meshEraserBtn, &QPushButton::toggled, this, &MainWindow::onMeshEraserToggle);
  
  connect(_meshWhiteBgCheck, &QCheckBox::toggled, _meshView, &MeshViewerWidget::setWhiteBackground);

  connect(_meshView, &MeshViewerWidget::meshChanged,   this, &MainWindow::onMeshChanged);
  connect(_meshView, &MeshViewerWidget::statusMessage, this, &MainWindow::onMeshStatus);

  // Enable remove button only when an item is selected
  connect(_meshFileList, &QListWidget::itemSelectionChanged, this, [this]() {
      _meshRemoveBtn->setEnabled(_meshFileList->currentItem() != nullptr);
  });

  // Double-click on file list → load mesh
  connect(_meshFileList, &QListWidget::itemDoubleClicked, this, [this](QListWidgetItem *item) {
      QString path = item->data(Qt::UserRole).toString();
      if (!path.isEmpty()) _meshView->loadMesh(path);
  });
}

// ─── Camera UI slots
// ──────────────────────────────────────────────────────────

void MainWindow::onRecordButtonClicked() {
    if (_recordBtn->text().contains("Dừng")) {
        _ctrl->stopRecording();
    } else {
        // Tắt tính toán live để đỡ nặng máy và sai lệch khi cầm tay (để dành tính toán sau bằng nút Run ICP)
        if (_accumCheck->isChecked()) _accumCheck->setChecked(false);
        if (_icpCheck->isChecked()) _icpCheck->setChecked(false);
        
        _ctrl->startRecording(_recordIntervalSpin->value());
    }
}

void MainWindow::onRecordingStateChanged(bool isRecording) {
    if (isRecording) {
        _recordBtn->setText("⏹ Dừng Record");
        _recordBtn->setStyleSheet(
            "QPushButton{background:#333333;color:#ff5555;font-weight:bold;border-radius:3px;padding:4px 10px; border: 1px solid #ff5555;}"
            "QPushButton:hover{background:#553333;}");
        _recordIntervalSpin->setEnabled(false);
    } else {
        _recordBtn->setText("🔴 Record");
        _recordBtn->setStyleSheet(
            "QPushButton{background:#7a2020;color:white;font-weight:bold;border-radius:3px;padding:4px 10px;}"
            "QPushButton:hover{background:#a03030;}");
        _recordIntervalSpin->setEnabled(true);
    }
}

void MainWindow::onCameraConnectClicked() {
  _ctrl->setCameraSource(_camSourceCombo->currentData().toInt());
  _ctrl->connectCamera("", 0);
}

void MainWindow::onCameraDisconnectClicked() { _ctrl->disconnectCamera(); }

void MainWindow::onCameraConnectionChanged(bool connected) {
  _camConnected = connected;
  setCameraConnectedState(connected);
  statusBar()->showMessage(connected ? "Camera connected" : "Camera disconnected");
}

void MainWindow::setCameraConnectedState(bool connected) {
  _camConnBtn->setEnabled(!connected);
  _camDiscBtn->setEnabled(connected);
  _camSourceCombo->setEnabled(!connected);
  if (connected) {
    _camStatusLbl->setText("● Connected");
    _camStatusLbl->setStyleSheet("color:#3ddc84;font-weight:bold;");
  } else {
    _camStatusLbl->setText("● Disconnected");
    _camStatusLbl->setStyleSheet("color:#e05c5c;font-weight:bold;");
  }
}

// ─── Geometry frame
// ───────────────────────────────────────────────────────────

void MainWindow::onGeometryFrameReady(const GeometryFrame &gf) {
  // Calculate centroid for debugging
  double cx = 0, cy = 0, cz = 0;
  int pts = gf.cloud_raw.count;
  if (pts > 0) {
    for (int i = 0; i < pts; ++i) {
      cx += gf.cloud_raw.xyz[i * 3 + 0];
      cy += gf.cloud_raw.xyz[i * 3 + 1];
      cz += gf.cloud_raw.xyz[i * 3 + 2];
    }
    cx /= pts;
    cy /= pts;
    cz /= pts;

    // Transform the centroid
    float tx = gf.camInvTransform(0, 0) * cx + gf.camInvTransform(0, 1) * cy +
               gf.camInvTransform(0, 2) * cz + gf.camInvTransform(0, 3);
    float ty = gf.camInvTransform(1, 0) * cx + gf.camInvTransform(1, 1) * cy +
               gf.camInvTransform(1, 2) * cz + gf.camInvTransform(1, 3);
    float tz = gf.camInvTransform(2, 0) * cx + gf.camInvTransform(2, 1) * cy +
               gf.camInvTransform(2, 2) * cz + gf.camInvTransform(2, 3);

    // Comment out to prevent 30fps UI freeze
    // appendLog(QString::asprintf("📊 [Centroid Log] RAW: (%.1f, %.1f, %.1f)  ==>  TX: (%.1f, %.1f, %.1f)",
    //                              cx, cy, cz, tx, ty, tz));

    // Write to CSV
    if (_scanRunning) {
      if (_centroidLogFilename.isEmpty()) {
        _centroidLogFilename =
            QString("centroid_log_%1.csv")
                .arg(QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss"));
      }

    _centroidWpCount++;
    QFile file(_centroidLogFilename);
    if (file.open(QIODevice::Append | QIODevice::Text)) {
      QTextStream out(&file);
      if (_centroidWpCount == 1) {
        out << "Waypoint,dx,dy,dz,droll,dpitch,dyaw,RawX,RawY,RawZ,TxX,TxY,TxZ,"
               "MatTX,MatTY,MatTZ\n";
      }
      out << QString::asprintf("%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%."
                               "2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                               _centroidWpCount, gf.pose.x, gf.pose.y,
                               gf.pose.z, gf.pose.rx, gf.pose.ry, gf.pose.rz,
                               cx, cy, cz, tx, ty, tz, gf.camInvTransform(0, 3),
                               gf.camInvTransform(1, 3),
                               gf.camInvTransform(2, 3));
      file.close();
    }
    }
  }

  if (_accumCheck->isChecked()) {
    // Note: accumulation is now handled inside AppController -> ScanEngine
    // This UI logic is just for fallback or immediate drawing.
  } else {
    _cloudView->updatePointCloud(gf.cloud_raw);
    if (_recordBtn->text().contains("Dừng")) {
        _cloudInfoLbl->setText(QString("🔴 ĐANG RECORD... %1 pts").arg(gf.cloud_raw.count));
        _cloudInfoLbl->setStyleSheet("color:#ff4444;font-weight:bold;font-size:12px;");
    } else {
        _cloudInfoLbl->setText(QString("Points: %1  [Camera frame]").arg(gf.cloud_raw.count));
        _cloudInfoLbl->setStyleSheet("color:#aaa;");
    }
  }
}

void MainWindow::onAccumStatsUpdated(int voxelCount, int pointCount) {
  _accumStatsLbl->setText(
      QString("%1 voxels  |  %2 pts").arg(voxelCount).arg(pointCount));
  _cloudInfoLbl->setText(QString("🔴 Accumulating: %1 voxels").arg(voxelCount));
  _exportCloudBtn->setEnabled(pointCount > 0);
  _voxelSizeSpin->setEnabled(pointCount == 0);
}

void MainWindow::onWorkerError(const QString &err) {
  appendLog("ERROR: " + err);
  statusBar()->showMessage("Error: " + err, 5000);
}

// ─── Scan slots
// ───────────────────────────────────────────────────────────────

void MainWindow::onScanProgressUpdated(int current, int total,
                                       const QString &status) {
  _orbitalStatusLbl->setText(status);
  if (total > 0)
    _orbitalProgress->setValue(int(double(current) / total * 100));
}

void MainWindow::onScanFinished() {
  setScanRunningState(false);
  _orbitalProgress->setValue(100);
  _orbitalStatusLbl->setText("Complete — export PLY when ready");
  _centroidLogFilename.clear();
  _centroidWpCount = 0;
  appendLog("Scan complete — export PLY when ready");
  statusBar()->showMessage("Scan complete", 5000);
}

void MainWindow::onOrbitalWaypointsReady(const QVector<OrbitalWaypoint> &wps) {
  if (wps.isEmpty())
    return;
  _orbitalWidget->setTimingParams(_scanFramesSpin->value(),
                                  _scanSettleSpin->value());
  _ctrl->loadOrbitalWaypoints(wps, _scanFramesSpin->value(),
                              _scanSettleSpin->value(),
                              _scanIntervalSpin->value());
  _startOrbitalBtn->setEnabled(true);
  _orbitalStatusLbl->setText(
      QString("%1 waypoints loaded — press ▶ to start").arg(wps.size()));
  _orbitalProgress->setValue(0);
  appendLog(QString("Orbital: %1 waypoints ready").arg(wps.size()));
}

void MainWindow::onStartOrbitalScanClicked() {
  setScanRunningState(true);
  _startOrbitalBtn->setEnabled(false);

  // Auto-enable accumulate on scan start
  _accumCheck->setChecked(true);
  _ctrl->scanEngine()->setEnabled(true);
  _cloudView->setAccumulate(true);

  // Reset centroid log
  _centroidLogFilename.clear();
  _centroidWpCount = 0;

  _ctrl->setAiPrompt(_aiPromptEdit->text());
  _ctrl->startScanSequence(_scanFramesSpin->value(), _scanSettleSpin->value(),
                           _scanIntervalSpin->value());

  _orbitalStatusLbl->setText("Orbital scan running…");
}

void MainWindow::setScanRunningState(bool running) {
  _scanRunning = running;
  _stopOrbitalBtn->setEnabled(running);
  _startOrbitalBtn->setEnabled(!running && true &&
                               !_orbitalWidget->waypoints().isEmpty());
}

// ─── Log ─────────────────────────────────────────────────────────────────────

void MainWindow::appendLog(const QString &msg) {
  _pendingLogLines.append(QString("[%1] %2").arg(
      QDateTime::currentDateTime().toString("HH:mm:ss.zzz"), msg));
}

void MainWindow::flushLog() {
  if (_pendingLogLines.isEmpty())
    return;
  _logEdit->appendPlainText(_pendingLogLines.join('\n'));
  _pendingLogLines.clear();

  const int blockCount = _logEdit->document()->blockCount();
  if (blockCount > LOG_MAX_LINES) {
    QTextCursor cur(_logEdit->document());
    cur.movePosition(QTextCursor::Start);
    cur.movePosition(QTextCursor::Down, QTextCursor::KeepAnchor,
                     blockCount - LOG_MAX_LINES);
    cur.removeSelectedText();
  }

  if (_logCountLbl) {
    _logCountLbl->setText(QString("%1 dòng").arg(_logEdit->document()->blockCount()));
  }
}

// ─── Extended Tool Slots ──────────────────────────────────────────────────────

void MainWindow::onCloudSnapshotClicked() {
  QString dir = QDir::currentPath() + "/screenshots";
  QDir().mkpath(dir);
  QString filename = "cloud_snapshot_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
  QString fullPath = dir + "/" + filename;
  if (_cloudView->captureScreenshot(fullPath)) {
    appendLog("📸 Đã lưu ảnh chụp 3D Point Cloud: " + filename);
    statusBar()->showMessage("Đã lưu ảnh chụp: " + filename, 5000);
  } else {
    appendLog("❌ Lưu ảnh chụp Point Cloud thất bại!");
  }
}

void MainWindow::onMeshSnapshotClicked() {
  QString dir = QDir::currentPath() + "/screenshots";
  QDir().mkpath(dir);
  QString filename = "mesh_snapshot_" + QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".png";
  QString fullPath = dir + "/" + filename;
  if (_meshView->captureScreenshot(fullPath)) {
    appendLog("📸 Đã lưu ảnh chụp 3D Mesh: " + filename);
    statusBar()->showMessage("Đã lưu ảnh chụp Mesh: " + filename, 5000);
  } else {
    appendLog("❌ Lưu ảnh chụp Mesh thất bại!");
  }
}

void MainWindow::onOpenDataFolderClicked() {
  QString dir = QCoreApplication::applicationDirPath() + "/records";
  if (!QDir(dir).exists()) {
    dir = QDir::currentPath() + "/captures";
    QDir().mkpath(dir);
  }
  QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
  appendLog("📂 Đã mở thư mục dữ liệu: " + dir);
}

// ─── Helpers
// ──────────────────────────────────────────────────────────────────

void MainWindow::pushIntrinsics() {
  CameraIntrinsics intr;
  intr.cameraModel = _cameraModelCombo->currentText();
  intr.fx = float(_fxSpin->value());
  intr.fy = float(_fySpin->value());
  intr.cx = float(_cxSpin->value());
  intr.cy = float(_cySpin->value());
  _ctrl->setIntrinsics(intr);
  _cloudView->setIntrinsics(intr);
}



// ─── Export
// ───────────────────────────────────────────────────────────────────

void MainWindow::exportPointCloudPLY() {
  Object3D obj = _ctrl->scanEngine()->getObject3D();
  if (obj.isEmpty()) {
    QMessageBox::information(this, "Export PLY", "No accumulated points.");
    return;
  }
  QDir().mkpath(_captureDir);
  const QString path = QFileDialog::getSaveFileName(
      this, "Export PLY",
      _captureDir + "/pointcloud_" +
          QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss") + ".ply",
      "PLY Files (*.ply);;All (*)");
  if (path.isEmpty())
    return;

  QString errMsg;
  if (!Exporter::exportPLY(path, obj, errMsg)) {
    QMessageBox::warning(this, "Export Failed", errMsg);
  } else {
    QMessageBox::information(this, "Export Success",
                             "Saved point cloud to:\n" + path);
  }
}

#include <QProcess>
#include <QDir>
#include <QCoreApplication>
void MainWindow::onRunIcpClicked()
{
    QString currentScan = _ctrl->currentScanDir();
    if (currentScan.isEmpty() || !QDir(currentScan).exists()) {
        currentScan = QFileDialog::getExistingDirectory(this, "Chọn thư mục chứa các file PLY (Scan / Record)", QCoreApplication::applicationDirPath());
        if (currentScan.isEmpty()) {
            return;
        }
        _ctrl->setCurrentScanDir(currentScan);
    } else {
        QMessageBox::StandardButton reply = QMessageBox::question(this, "Chạy SLAM", "Bạn muốn chạy SLAM trên thư mục vừa quét?\n" + currentScan + "\n\n(Chọn No để mở thư mục Scan khác trong quá khứ)", QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        if (reply == QMessageBox::Cancel) return;
        if (reply == QMessageBox::No) {
            currentScan = QFileDialog::getExistingDirectory(this, "Chọn thư mục chứa các file PLY (Scan / Record)", QCoreApplication::applicationDirPath());
            if (currentScan.isEmpty()) {
                return;
            }
            _ctrl->setCurrentScanDir(currentScan);
        }
    }

    QDir dir(QCoreApplication::applicationDirPath());
    // Lùi ra ngoài thư mục build để tìm thư mục gốc chứa file py
    while (!dir.exists("scripts/slam_engine.py") && !dir.isRoot()) {
        dir.cdUp();
    }

    QString pyScript = dir.absoluteFilePath("scripts/slam_engine.py");
    QFileInfo fi(pyScript);
    if (!fi.exists()) {
        QMessageBox::warning(this, "Lỗi ICP", "Không tìm thấy file Python ICP tại:\n" + pyScript);
        return;
    }

    _runIcpBtn->setEnabled(false);
    _runIcpBtn->setText("Đang chạy SLAM...");

    QProcess *process = new QProcess(this);
    process->setWorkingDirectory(QDir::currentPath());
    process->start("python", QStringList() << pyScript << currentScan
                                           << "--low-memory"
                                           << "--voxel-size" << "4.0"
                                           << "--no-vis");

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process, currentScan](int exitCode, QProcess::ExitStatus exitStatus) {
        _runIcpBtn->setEnabled(true);
        _runIcpBtn->setText("🐍 Run SLAM");
        
        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            QString outPath = currentScan + "/final_optimized_model.ply";
            QString binPath = currentScan + "/final_optimized_model.bin";
            
            // Load binary array directly into CloudAccumulator
            QFile f(binPath);
            if (f.open(QIODevice::ReadOnly)) {
                int N = 0;
                f.read((char*)&N, sizeof(int));
                if (N > 0 && N < 50000000) {
                    std::vector<float> xyz(N * 3);
                    std::vector<uint8_t> rgb(N * 3);
                    f.read((char*)xyz.data(), N * 3 * sizeof(float));
                    f.read((char*)rgb.data(), N * 3 * sizeof(uint8_t));
                    
                    _ctrl->scanEngine()->loadRawCloud(xyz, rgb, N);
                    _cloudView->onFullRebuildReady(_ctrl->scanEngine()->getObject3D().pointCloud);
                    _accumCheck->setChecked(true); // Force show accumulated cloud
                }
                f.close();
            }

            QMessageBox::information(this, "SLAM Hoàn thành", "Đã tối ưu hóa và dung hợp mây điểm thành công!\nKết quả lưu tại:\n" + outPath);
        } else {
            QString err = process->readAllStandardError();
            QString out = process->readAllStandardOutput();
            QMessageBox::warning(this, "Lỗi SLAM", "Chạy SLAM Engine thất bại.\nError:\n" + err + "\nOutput:\n" + out);
        }
        process->deleteLater();
    });
}

void MainWindow::onExportMeshClicked()
{
    QString currentScan = _ctrl->currentScanDir();
    QString plyPath;

    if (!currentScan.isEmpty() && QDir(currentScan).exists() && QFile::exists(currentScan + "/final_optimized_model.ply")) {
        QMessageBox::StandardButton reply = QMessageBox::question(this, "Xuất Mesh", 
            "Bạn muốn xuất Mesh từ thư mục Scan vừa chạy SLAM hiện tại?\n\n(Chọn No để chọn một file PLY bất kỳ trên máy)", 
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
            
        if (reply == QMessageBox::Cancel) return;
        
        if (reply == QMessageBox::Yes) {
            plyPath = currentScan + "/final_optimized_model.ply";
        } else {
            plyPath = QFileDialog::getOpenFileName(this, "Chọn file PLY", currentScan, "PLY Files (*.ply)");
            if (plyPath.isEmpty()) return;
        }
    } else {
        plyPath = QFileDialog::getOpenFileName(this, "Chọn file PLY", QCoreApplication::applicationDirPath(), "PLY Files (*.ply)");
        if (plyPath.isEmpty()) return;
    }

    QDir dir(QCoreApplication::applicationDirPath());
    while (!dir.exists("scripts/mesh_builder.py") && !dir.isRoot()) {
        dir.cdUp();
    }

    QFileInfo plyInfo(plyPath);
    QString pyScript = dir.absoluteFilePath("scripts/mesh_builder.py");
    bool useSlam = false;
    if (!plyInfo.baseName().contains("_mesh")) {
        pyScript = dir.absoluteFilePath("scripts/slam_object_builder.py");
        useSlam = true;
    }

    if (!QFile::exists(pyScript)) {
        QMessageBox::warning(this, "Lỗi Xuất Mesh", "Không tìm thấy script python:\n" + pyScript);
        return;
    }

    _exportMeshBtn->setEnabled(false);
    _exportMeshBtn->setText("Đang tạo Mesh...");

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    
    // Lưu mesh vào thư mục ExportedMeshes ngay bên trong record folder
    QDir recordDir(plyInfo.absolutePath());
    if (!recordDir.exists("ExportedMeshes")) {
        recordDir.mkdir("ExportedMeshes");
    }
    QString exportDir = recordDir.absoluteFilePath("ExportedMeshes");
    
    QString outPath = exportDir + "/" + plyInfo.baseName() + "_mesh_" + timestamp + ".ply";

    QStringList args;
    if (useSlam) {
        // Dùng SLAM + AI trên thư mục chứa bộ dataset
        args << pyScript << plyInfo.absolutePath() << "--prompt" << _aiPromptEdit->text() << "--output" << outPath;
    } else {
        // Dùng Mesh Builder (Poisson) để làm mịn Mesh đã có sẵn
        args << pyScript << plyPath << "--output" << outPath;
    }

    QProcess *process = new QProcess(this);
    process->setWorkingDirectory(dir.absolutePath());
    process->start("python3", args);

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process, outPath](int exitCode, QProcess::ExitStatus exitStatus) {
        _exportMeshBtn->setEnabled(true);
        _exportMeshBtn->setText("📦 Xuất Mesh");

        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            QMessageBox::information(this, "Tạo Mesh Thành công", "Đã xuất Mesh (PLY) thành công tại:\n" + outPath);
        } else {
            QString err = process->readAllStandardError();
            QString out = process->readAllStandardOutput();
            QMessageBox::warning(this, "Lỗi Tạo Mesh", "Tạo Mesh thất bại.\nError:\n" + err + "\nOutput:\n" + out);
        }
        process->deleteLater();
    });
}

void MainWindow::onViewMeshClicked()
{
    QString currentScan = _ctrl->currentScanDir();
    QString meshPath;

    // Tìm file mesh mới nhất trong thư mục scan hiện tại
    QString defaultMesh;
    if (!currentScan.isEmpty() && QDir(currentScan).exists()) {
        QDir scanDir(currentScan);
        QStringList filters;
        filters << "*_mesh*.obj" << "*_mesh*.stl" << "*_mesh*.ply";
        scanDir.setNameFilters(filters);
        scanDir.setSorting(QDir::Time); // Sắp xếp theo thời gian mới nhất (Newest first)
        
        QFileInfoList list = scanDir.entryInfoList();
        if (!list.isEmpty()) {
            defaultMesh = list.first().absoluteFilePath();
        }
    }

    if (!defaultMesh.isEmpty()) {
        QMessageBox::StandardButton reply = QMessageBox::question(this, "Xem Mesh", 
            "Bạn muốn xem file Mesh vừa xuất gần đây nhất từ thư mục Scan hiện tại?\n(" + QFileInfo(defaultMesh).fileName() + ")\n\n(Chọn No để duyệt và mở file khác trên máy)", 
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
            
        if (reply == QMessageBox::Cancel) return;
        
        if (reply == QMessageBox::Yes) {
            meshPath = defaultMesh;
        } else {
            meshPath = QFileDialog::getOpenFileName(this, "Chọn file Mesh", currentScan, "Mesh Files (*.obj *.stl *.ply);;All Files (*)");
            if (meshPath.isEmpty()) return;
        }
    } else {
        meshPath = QFileDialog::getOpenFileName(this, "Chọn file Mesh", QCoreApplication::applicationDirPath(), "Mesh Files (*.obj *.stl *.ply);;All Files (*)");
        if (meshPath.isEmpty()) return;
    }

    QDir dir(QCoreApplication::applicationDirPath());
    while (!dir.exists("scripts/mesh_viewer.py") && !dir.isRoot()) {
        dir.cdUp();
    }

    QString pyScript = dir.absoluteFilePath("scripts/mesh_viewer.py");
    if (!QFile::exists(pyScript)) {
        QMessageBox::warning(this, "Lỗi Xem Mesh", "Không tìm thấy script python:\n" + pyScript);
        return;
    }

    _viewMeshBtn->setEnabled(false);
    _viewMeshBtn->setText("Đang mở...");

    QProcess *process = new QProcess(this);
    process->setWorkingDirectory(dir.absolutePath());
    process->setProcessChannelMode(QProcess::MergedChannels);
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process]() {
        const QString output = QString::fromUtf8(process->readAllStandardOutput()).trimmed();
        if (!output.isEmpty()) appendLog("[Handheld SLAM] " + output);
    });
    
    // Forward stdout and stderr to parent console for debugging
    process->setProcessChannelMode(QProcess::ForwardedChannels);
    
    process->start("python3", QStringList() << pyScript << meshPath);

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process](int exitCode, QProcess::ExitStatus exitStatus) {
        _viewMeshBtn->setEnabled(true);
        _viewMeshBtn->setText("👀 Xem Mesh");

        if (exitStatus != QProcess::NormalExit || exitCode != 0) {
            QString err = process->readAllStandardError();
            QString out = process->readAllStandardOutput();
            QMessageBox::warning(this, "Lỗi Xem Mesh", "Mở Mesh Viewer thất bại.\nError:\n" + err + "\nOutput:\n" + out);
        }
        process->deleteLater();
    });
}

// ─── Mesh Builder slots ────────────────────────────────────────────────────────

void MainWindow::onBuildMeshClicked()
{
    QString currentScan = _ctrl->currentScanDir();
    if (currentScan.isEmpty() || !QDir(currentScan).exists()) {
        currentScan = QFileDialog::getExistingDirectory(this, "Chọn thư mục Record", QCoreApplication::applicationDirPath());
        if (currentScan.isEmpty()) return;
        _ctrl->setCurrentScanDir(currentScan);
    } else {
        QMessageBox::StandardButton reply = QMessageBox::question(this, "Xây Mesh Object",
            "Bạn muốn tách vật thể từ thư mục Record hiện tại?\n" + currentScan + "\n\n(Chọn No để chọn thư mục Record khác)",
            QMessageBox::Yes | QMessageBox::No | QMessageBox::Cancel);
        if (reply == QMessageBox::Cancel) return;
        if (reply == QMessageBox::No) {
            currentScan = QFileDialog::getExistingDirectory(this, "Chọn thư mục Record", QCoreApplication::applicationDirPath());
            if (currentScan.isEmpty()) return;
            _ctrl->setCurrentScanDir(currentScan);
        }
    }
    
    // Ensure currentScan is an absolute path
    if (QDir::isRelativePath(currentScan)) {
        currentScan = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(currentScan);
    }

    QDir dir(QCoreApplication::applicationDirPath());
    while (!dir.exists("scripts/slam_object_builder.py") && !dir.isRoot()) {
        dir.cdUp();
    }
    QString pyScript = dir.absoluteFilePath("scripts/slam_object_builder.py");
    if (!QFile::exists(pyScript)) {
        QMessageBox::warning(this, "Lỗi", "Không tìm thấy script:\n" + pyScript);
        return;
    }

    QString timestamp = QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");
    QDir recordDir(currentScan);
    if (!recordDir.exists("ExportedMeshes")) {
        recordDir.mkdir("ExportedMeshes");
    }
    QString outPath = recordDir.absoluteFilePath("ExportedMeshes/object_mesh_" + timestamp + ".ply");

    QMessageBox motionDialog(this);
    motionDialog.setWindowTitle("Kiểu quét");
    motionDialog.setText("Dữ liệu được ghi theo cách nào?");
    motionDialog.setInformativeText(
        "Camera cầm tay: camera di chuyển quanh vật.\n"
        "Vật quay đều 1 vòng: camera giữ cố định và vật quay gần đều đủ 360°.");
    motionDialog.addButton("Camera cầm tay", QMessageBox::AcceptRole);
    QPushButton *turntableButton = motionDialog.addButton("Vật quay đều 1 vòng", QMessageBox::ActionRole);
    QPushButton *cancelButton = motionDialog.addButton(QMessageBox::Cancel);
    motionDialog.exec();
    if (!motionDialog.clickedButton() || motionDialog.clickedButton() == cancelButton) return;
    const bool manualTurntable = motionDialog.clickedButton() == turntableButton;

    _buildMeshBtn2->setEnabled(false);
    _buildMeshBtn2->setText(manualTurntable ? "Đang căn góc quay vật..." : "Đang chạy AI + SLAM...");

    QProcess *process = new QProcess(this);
    process->setWorkingDirectory(dir.absolutePath());
    
    QString prompt = _aiPromptEdit ? _aiPromptEdit->text() : "cup";
    // Handheld scan: allow up to 160 keyframes so TSDF gets full 360° coverage
    // regardless of scan direction. 100 left the view density too sparse in
    // regions the operator only passed over once (no loop-closure revisit):
    // select_motion_keyframes() uniformly subsamples all motion-triggered
    // candidates down to this cap, so a low cap thins out those no-loop
    // regions specifically -- exactly where the multi-view support filter
    // (--min-view-support) then had the fewest overlapping views to work
    // with. Turntable: retain all 48 uniformly sampled yaw views (one every
    // ~7.5 degrees).  The previous cap of 24 discarded every other valid
    // view after tracking, needlessly opening faces that are visible only in
    // a narrow angular interval.
    const QString maxViews = manualTurntable ? "48" : "160";
    QStringList builderArgs = {pyScript, currentScan,
                                "--prompt", prompt, "--output", outPath,
                                "--max-views", maxViews, "--voxel", "0.002",
                                // A pose-graph component that could not be anchored
                                // belongs to a different world frame. Fusing it as
                                // "soft" creates a duplicate/offset surface even
                                // when its depth and mask look valid. The builder
                                // logs every strict exclusion for diagnosis.
                                "--depth-filter", "strict",
                                // 100 Taubin iterations visibly rounds box edges and
                                // closes thin, partially observed surfaces.  Keep the
                                // builder's feature-preserving default instead.
                                "--smooth-iters", "20",
                                "--max-input-frames", "1000",
                                "--tracking-report", recordDir.absoluteFilePath("tracking_report.json")};
    if (manualTurntable) {
        // Fixed-camera rotation needs angular coverage, not hundreds of near-
        // duplicate video frames.  This keeps FastSAM/Open3D within laptop
        // RAM/VRAM while retaining about one view every 7.5 degrees.
        builderArgs << "--manual-turntable"
                    << "--turntable-yaw-mode" << "uniform"
                    << "--turntable-max-tracking-frames" << "48";
    }
    process->setProcessChannelMode(QProcess::MergedChannels);
    connect(process, &QProcess::readyReadStandardOutput, this, [this, process]() {
        const QString output = QString::fromUtf8(process->readAllStandardOutput()).trimmed();
        if (!output.isEmpty()) appendLog("[Mesh Builder] " + output);
    });
    process->start("python3", builderArgs);

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process, outPath, recordDir](int exitCode, QProcess::ExitStatus exitStatus) {
        _buildMeshBtn2->setEnabled(true);
        _buildMeshBtn2->setText("🐍 Xây Mesh Object (AI)");

        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            // Function to add a single mesh path to the list
            auto addMeshToList = [this](const QString& path, const QString& displayName = "") {
                bool found = false;
                for (int i = 0; i < _meshFileList->count(); ++i) {
                    if (_meshFileList->item(i)->data(Qt::UserRole).toString() == path) {
                        found = true; break;
                    }
                }
                if (!found) {
                    QString name = displayName.isEmpty() ? QFileInfo(path).fileName() : displayName;
                    auto *item = new QListWidgetItem(name);
                    item->setData(Qt::UserRole, path);
                    _meshFileList->addItem(item);
                }
            };

            // Add main output
            addMeshToList(outPath);

            // Also search for debug stages and add them to the list
            QDir debugDir(recordDir.absoluteFilePath("reconstruction/debug_stages"));
            if (debugDir.exists()) {
                QStringList filters;
                filters << "*.ply" << "*.obj";
                QFileInfoList fileList = debugDir.entryInfoList(filters, QDir::Files | QDir::NoDotAndDotDot);
                for (const QFileInfo &fileInfo : fileList) {
                    addMeshToList(fileInfo.absoluteFilePath(), "[Debug] " + fileInfo.fileName());
                }
            }

            // Select the main output
            for (int i = 0; i < _meshFileList->count(); ++i) {
                if (_meshFileList->item(i)->data(Qt::UserRole).toString() == outPath) {
                    _meshFileList->setCurrentItem(_meshFileList->item(i));
                    break;
                }
            }

            // Load result directly into C++ viewer
            _meshView->loadMesh(outPath);
        } else {
            // Note: process uses MergedChannels so stderr is part of stdout.
            // readAllStandardError() would be empty; we captured output via
            // the readyReadStandardOutput signal, so show a generic message.
            QString err = QString::fromUtf8(process->readAllStandardOutput()).trimmed();
            if (err.isEmpty()) err = "(xem log bên dưới để biết chi tiết)";
            // Only show last 800 chars to avoid giant dialog
            if (err.length() > 800) err = "..." + err.right(800);
            QMessageBox::warning(this, "Lỗi", "Xây Mesh thất bại.\nError:\n" + err);
        }
        process->deleteLater();
    });
}

void MainWindow::onViewLoadMeshClicked()
{
    QString startDir = QCoreApplication::applicationDirPath() + "/records";
    QString path = QFileDialog::getOpenFileName(
        this, "Mở File Mesh PLY/OBJ", startDir, "Mesh Files (*.ply *.obj)");
    if (path.isEmpty()) return;

    // Add to recent list
    bool found = false;
    for (int i = 0; i < _meshFileList->count(); ++i) {
        if (_meshFileList->item(i)->data(Qt::UserRole).toString() == path) {
            found = true; break;
        }
    }
    if (!found) {
        auto *item = new QListWidgetItem(QFileInfo(path).fileName());
        item->setData(Qt::UserRole, path);
        _meshFileList->addItem(item);
        _meshFileList->setCurrentItem(item);
    }

    // Load directly into C++ viewer
    _meshView->loadMesh(path);
}

void MainWindow::onMeshAutoClean()
{
    _meshView->autoClean();
}

void MainWindow::onMeshUndo()
{
    _meshView->undo();
}

void MainWindow::onMeshEraserToggle(bool checked)
{
    _meshView->setEraserMode(checked);
    if (checked) {
        _meshEraserBtn->setText("✂ Tẩy Thủ Công [BẬT]");
        _meshEraserBtn->setStyleSheet(
            "QPushButton{background:#b02080;color:white;font-weight:bold;"
            "border-radius:5px;padding:8px 14px;font-size:12px;"
            "border:2px solid #ff60c0;}"
            "QPushButton:hover{background:#c030a0;}");
    } else {
        _meshEraserBtn->setText("✂ Tẩy Thủ Công");
        _meshEraserBtn->setStyleSheet(
            "QPushButton{background:#6a2060;color:white;font-weight:bold;"
            "border-radius:5px;padding:8px 14px;font-size:12px;}"
            "QPushButton:hover{background:#8a3080;}"
            "QPushButton:disabled{background:#333;color:#666;}");
    }
}

void MainWindow::onMeshSave()
{
    if (!_meshView->hasMesh()) {
        QMessageBox::information(this, "Thông báo", "Chưa có mesh để lưu!");
        return;
    }
    QString path = _meshView->loadedPath();
    if (path.isEmpty()) {
        path = QFileDialog::getSaveFileName(
            this, "Lưu Mesh", QCoreApplication::applicationDirPath(),
            "PLY Files (*.ply)");
    }
    if (!path.isEmpty()) _meshView->saveMesh(path);
}

void MainWindow::onMeshExportObj()
{
    // Pick source PLY
    QString plyPath = _meshView->loadedPath();
    if (plyPath.isEmpty()) {
        plyPath = QFileDialog::getOpenFileName(
            this, "Chọn file PLY nguồn",
            QCoreApplication::applicationDirPath(),
            "PLY Files (*.ply)");
    }
    if (plyPath.isEmpty()) return;

    // Pick destination OBJ
    QString defaultObj = plyPath;
    defaultObj.replace(QRegularExpression("\\.ply$", QRegularExpression::CaseInsensitiveOption), ".obj");
    QString objPath = QFileDialog::getSaveFileName(
        this, "Lưu file OBJ", defaultObj, "OBJ Files (*.obj)");
    if (objPath.isEmpty()) return;

    // Convert via Python / open3d (single-line, no dependency on slam_object_builder)
    QString script = QString(
        "import open3d as o3d; "
        "m = o3d.io.read_triangle_mesh('%1'); "
        "m.orient_triangles(); "
        "m.compute_vertex_normals(); "
        "ok = o3d.io.write_triangle_mesh('%2', m, write_ascii=False); "
        "print('OK' if ok else 'FAIL')"
    ).arg(plyPath, objPath);

    QString python = QStandardPaths::findExecutable("python3");
    if (python.isEmpty()) python = QStandardPaths::findExecutable("python");
    if (python.isEmpty()) {
        QMessageBox::warning(this, "Lỗi", "Không tìm thấy python3 / python trong PATH.");
        return;
    }

    QProcess proc;
    proc.start(python, {"-c", script});
    proc.waitForFinished(30000);
    QString out = proc.readAllStandardOutput().trimmed();
    QString err = proc.readAllStandardError().trimmed();

    if (proc.exitCode() == 0 && out.contains("OK")) {
        QMessageBox::information(this, "Xuất OBJ",
            QString("Đã lưu thành công:\n%1").arg(objPath));
    } else {
        QMessageBox::warning(this, "Xuất OBJ thất bại",
            QString("Exit code: %1\nstdout: %2\nstderr: %3")
                .arg(proc.exitCode()).arg(out).arg(err.left(400)));
    }
}

void MainWindow::onMeshRemoveFile()
{
    QListWidgetItem *item = _meshFileList->currentItem();
    if (item) {
        delete _meshFileList->takeItem(_meshFileList->row(item));
        _meshInfoLbl->setText("Đã xóa khỏi danh sách.");
    }
}

void MainWindow::onMeshPoissonRecon()
{
    if (!_meshView->hasMesh()) {
        QMessageBox::information(this, "Thông báo", "Chưa có point cloud để bọc Mesh!");
        return;
    }
    QString plyPath = _meshView->loadedPath();
    if (plyPath.isEmpty()) {
        QMessageBox::warning(this, "Thông báo", "Hãy lưu file hiện tại xuống đĩa trước (bấm Lưu Mesh).");
        return;
    }

    QDir dir(QCoreApplication::applicationDirPath());
    while (!dir.exists("scripts/mesh_builder.py") && !dir.isRoot()) {
        dir.cdUp();
    }
    QString pyScript = dir.absoluteFilePath("scripts/mesh_builder.py");
    if (!QFile::exists(pyScript)) {
        QMessageBox::warning(this, "Lỗi", "Không tìm thấy script:\n" + pyScript);
        return;
    }

    QFileInfo fi(plyPath);
    QString outPath = fi.absolutePath() + "/" + fi.baseName() + "_poisson.ply";

    _meshPoissonBtn->setEnabled(false);
    _meshPoissonBtn->setText("Đang chạy Poisson...");

    QProcess *process = new QProcess(this);
    process->setWorkingDirectory(dir.absolutePath());
    process->start("python3", QStringList() << pyScript << plyPath << "--output" << outPath << "--depth" << "9");

    connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this, process, outPath](int exitCode, QProcess::ExitStatus exitStatus) {
        _meshPoissonBtn->setEnabled(true);
        _meshPoissonBtn->setText("📦 Bọc Mesh (Poisson)");
        
        if (exitStatus == QProcess::NormalExit && exitCode == 0) {
            QMessageBox::information(this, "Hoàn thành", "Tạo Poisson Mesh thành công!\n" + outPath);
            _meshView->loadMesh(outPath);
            // Thêm vào danh sách nếu chưa có
            bool found = false;
            for (int i = 0; i < _meshFileList->count(); ++i) {
                if (_meshFileList->item(i)->data(Qt::UserRole).toString() == outPath) {
                    found = true; break;
                }
            }
            if (!found) {
                auto *item = new QListWidgetItem(QFileInfo(outPath).fileName());
                item->setData(Qt::UserRole, outPath);
                _meshFileList->addItem(item);
                _meshFileList->setCurrentItem(item);
            }
        } else {
            QString err = process->readAllStandardError();
            QMessageBox::warning(this, "Lỗi", "Bọc Mesh thất bại.\nError:\n" + err);
        }
        process->deleteLater();
    });
}

void MainWindow::onMeshChanged(int triCount)
{
    _meshInfoLbl->setText(QString("Tam giác: %1").arg(triCount));
}

void MainWindow::onMeshStatus(const QString &msg)
{
    appendLog("[Mesh] " + msg);
    _meshInfoLbl->setText(msg);
}

void MainWindow::onKillProcessesClicked() {
    auto processes = this->findChildren<QProcess*>();
    int killedCount = 0;
    for (QProcess* p : processes) {
        if (p->state() != QProcess::NotRunning) {
            p->kill();
            killedCount++;
        }
    }
    
    // Also reset buttons just in case
    if (_runIcpBtn) {
        _runIcpBtn->setEnabled(true);
        _runIcpBtn->setText("🐍 Run SLAM");
    }
    if (_exportMeshBtn) {
        _exportMeshBtn->setEnabled(true);
        _exportMeshBtn->setText("📦 Xuất Mesh");
    }
    if (_buildMeshBtn2) {
        _buildMeshBtn2->setEnabled(true);
        _buildMeshBtn2->setText("🐍 Xây Mesh Object (AI)");
    }
    if (_meshPoissonBtn) {
        _meshPoissonBtn->setEnabled(true);
        _meshPoissonBtn->setText("📦 Bọc Mesh (Poisson Recon)");
    }
    
    if (killedCount > 0) {
        appendLog(QString("Killed %1 background Python processes.").arg(killedCount));
        QMessageBox::information(this, "Stopped", QString("Đã dừng %1 tiến trình chạy ngầm.").arg(killedCount));
    } else {
        appendLog("No running background processes found.");
    }
}
