#include "orbitalscandialog.h"
#include "orbitaltrajectory.h"

#include <QButtonGroup>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QSpinBox>
#include <QSplitter>
#include <QStackedWidget>
#include <QTableWidget>
#include <QVBoxLayout>

// ─── Constructor ──────────────────────────────────────────────────────────────

OrbitalScanWidget::OrbitalScanWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *mainVl = new QVBoxLayout(this);
    mainVl->setContentsMargins(0, 0, 0, 0);
    mainVl->setSpacing(6);

    buildModeSelector();
    buildSphericalPanel();
    buildRandomPanel();
    buildTablePanel();

    // ── Stats + buttons ───────────────────────────────────────────────────────
    _statsLabel = new QLabel;
    _statsLabel->setStyleSheet("color:#888;font-size:11px;");

    _generateBtn = new QPushButton("⟳  Generate Trajectory");
    _generateBtn->setStyleSheet(
        "QPushButton{background:#1a5a8a;color:white;font-weight:bold;"
        "border-radius:4px;padding:5px 12px;}"
        "QPushButton:hover{background:#2a7abf;}");

    _loadScanBtn = new QPushButton("▶  Load into Scan Sequence");
    _loadScanBtn->setEnabled(false);
    _loadScanBtn->setStyleSheet(
        "QPushButton{background:#3a7d44;color:white;font-weight:bold;"
        "border-radius:4px;padding:5px 12px;}"
        "QPushButton:hover{background:#4a9d54;}"
        "QPushButton:disabled{background:#333;color:#555;}");

    // ── Mode stack ────────────────────────────────────────────────────────────
    _modeStack = new QStackedWidget;
    _modeStack->addWidget(_sphericalPanel);  // index 0
    _modeStack->addWidget(_randomPanel);     // index 1

    // ── Assemble param group ──────────────────────────────────────────────────
    auto *paramGroup = new QGroupBox("Orbital Trajectory Parameters");
    auto *paramVl    = new QVBoxLayout(paramGroup);
    paramVl->setContentsMargins(8, 8, 8, 8);
    paramVl->setSpacing(4);

    // Mode selector row
    auto *modeRow = new QHBoxLayout;
    modeRow->addWidget(_sphericalRadio);
    modeRow->addWidget(_randomRadio);
    modeRow->addStretch(1);
    paramVl->addLayout(modeRow);

    paramVl->addWidget(_modeStack);

    // Stats row — timing đến từ MainWindow
    auto *statsRow = new QHBoxLayout;
    statsRow->addWidget(_statsLabel, 1);
    paramVl->addLayout(statsRow);

    // Buttons — stacked vertically for sidebar compactness
    auto *btnVl = new QVBoxLayout;
    btnVl->setSpacing(4);
    btnVl->addWidget(_generateBtn);
    btnVl->addWidget(_loadScanBtn);
    paramVl->addLayout(btnVl);

    mainVl->addWidget(paramGroup);
    mainVl->addWidget(_table, 1);

    // ── Connections ───────────────────────────────────────────────────────────
    connect(_generateBtn, &QPushButton::clicked, this, &OrbitalScanWidget::onGenerate);
    connect(_loadScanBtn, &QPushButton::clicked, this, &OrbitalScanWidget::onLoadToScan);
    connect(_table, &QTableWidget::cellClicked,
            this, &OrbitalScanWidget::onTableCellClicked);
    connect(_sphericalRadio, &QRadioButton::toggled, this, &OrbitalScanWidget::onModeChanged);

    // Preview label auto-update
    auto updateStats = [this]{ updatePreviewLabel(); };
    connect(_yawStepSizeSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, updateStats);
    connect(_pitchStepSizeSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, updateStats);
    connect(_halfYawSpin,       QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, updateStats);
    connect(_halfPitchSpin,     QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, updateStats);
    connect(_nPosesSpin,        QOverload<int>::of(&QSpinBox::valueChanged),          this, updateStats);

    _sphericalRadio->setChecked(true);
    updatePreviewLabel();
    QMetaObject::invokeMethod(this, [this]{
        emit scanModeChanged("Spherical");
    }, Qt::QueuedConnection);
}

// ─── Public API ───────────────────────────────────────────────────────────────

void OrbitalScanWidget::setTimingParams(int nFrames, int settleMs)
{
    _nFrames  = nFrames;
    _settleMs = settleMs;
    updatePreviewLabel();
}

// ─── Mode selector ────────────────────────────────────────────────────────────

void OrbitalScanWidget::buildModeSelector()
{
    _sphericalRadio = new QRadioButton("Spherical orbit");
    _sphericalRadio->setChecked(true);
    _sphericalRadio->setStyleSheet("font-weight:bold;");

    _randomRadio = new QRadioButton("Random 6-DOF");
    _randomRadio->setStyleSheet("font-weight:bold;");

    auto *grp = new QButtonGroup(this);
    grp->addButton(_sphericalRadio);
    grp->addButton(_randomRadio);
}

// ─── Spherical panel ──────────────────────────────────────────────────────────

void OrbitalScanWidget::buildSphericalPanel()
{
    _sphericalPanel = new QWidget;
    auto *gl = new QGridLayout(_sphericalPanel);
    gl->setContentsMargins(0, 4, 0, 4);
    gl->setSpacing(4);

    auto mkDSpin = [](double lo, double hi, double val, const QString &suf = "") {
        auto *s = new QDoubleSpinBox;
        s->setRange(lo, hi); s->setValue(val); s->setDecimals(1);
        if (!suf.isEmpty()) s->setSuffix(suf);
        s->setFixedWidth(78);
        return s;
    };

    _radiusSpin        = mkDSpin(10, 2000,  300.0, " mm");
    _halfPitchSpin     = mkDSpin( 0,   89,   45.0, " °");
    _halfYawSpin       = mkDSpin( 0,   89,   45.0, " °");
    _halfRollSpin      = mkDSpin(-45,  45,    0.0, " °");
    _yawStepSizeSpin   = mkDSpin( 1,   90,   15.0, " °");
    _pitchStepSizeSpin = mkDSpin( 1,   90,   15.0, " °");

    gl->addWidget(new QLabel("Radius (mm)"),    0, 0); gl->addWidget(_radiusSpin,        0, 1);
    gl->addWidget(new QLabel("Roll off (°)"),   0, 2); gl->addWidget(_halfRollSpin,      0, 3);
    gl->addWidget(new QLabel("±Pitch (°)"),     1, 0); gl->addWidget(_halfPitchSpin,      1, 1);
    gl->addWidget(new QLabel("±Yaw (°)"),       1, 2); gl->addWidget(_halfYawSpin,        1, 3);
    gl->addWidget(new QLabel("Pitch Δ°"),       2, 0); gl->addWidget(_pitchStepSizeSpin,  2, 1);
    gl->addWidget(new QLabel("Yaw Δ°"),         2, 2); gl->addWidget(_yawStepSizeSpin,    2, 3);
    gl->setColumnStretch(1, 1); gl->setColumnStretch(3, 1);
}

// ─── Random Window panel ──────────────────────────────────────────────────────

void OrbitalScanWidget::buildRandomPanel()
{
    _randomPanel = new QWidget;
    auto *gl = new QGridLayout(_randomPanel);
    gl->setContentsMargins(0, 4, 0, 4);
    gl->setSpacing(4);

    auto mkD = [](double lo, double hi, double val, const QString &suf = "") {
        auto *s = new QDoubleSpinBox;
        s->setRange(lo, hi); s->setValue(val); s->setDecimals(1);
        if (!suf.isEmpty()) s->setSuffix(suf);
        s->setFixedWidth(78);
        return s;
    };
    auto mkI = [](int lo, int hi, int val, const QString &suf = "") {
        auto *s = new QSpinBox;
        s->setRange(lo, hi); s->setValue(val);
        if (!suf.isEmpty()) s->setSuffix(suf);
        s->setFixedWidth(78);
        return s;
    };

    _nPosesSpin  = mkI(1, 5000, 20);
    _seedSpin    = mkI(0, 99999, 0);
    _seedSpin->setToolTip("0 = time-based (non-deterministic)");

    _xyzStepSpin = mkI(1, 500, 10, " mm");
    _xyzStepSpin->setToolTip("Grid step for X/Y/Z sampling");
    _rotStepSpin = mkI(1, 45,  5,  " °");
    _rotStepSpin->setToolTip("Grid step for Roll/Pitch/Yaw sampling");

    _winXSpin    = mkD(0, 2000, 100.0, " mm");
    _winYSpin    = mkD(0, 2000, 100.0, " mm");
    _winZSpin    = mkD(0, 2000, 100.0, " mm");

    _winRollSpin  = mkD(0, 45, 45.0, " °");
    _winPitchSpin = mkD(0, 45, 45.0, " °");
    _winYawSpin   = mkD(0, 45, 45.0, " °");

    gl->addWidget(new QLabel("Poses"),        0, 0); gl->addWidget(_nPosesSpin,    0, 1);
    gl->addWidget(new QLabel("Seed"),         0, 2); gl->addWidget(_seedSpin,      0, 3);
    gl->addWidget(new QLabel("Step XYZ"),     1, 0); gl->addWidget(_xyzStepSpin,   1, 1);
    gl->addWidget(new QLabel("Step Rot"),     1, 2); gl->addWidget(_rotStepSpin,   1, 3);
    gl->addWidget(new QLabel("±X (mm)"),      2, 0); gl->addWidget(_winXSpin,      2, 1);
    gl->addWidget(new QLabel("±Y (mm)"),      2, 2); gl->addWidget(_winYSpin,      2, 3);
    gl->addWidget(new QLabel("±Z (mm)"),      3, 0); gl->addWidget(_winZSpin,      3, 1);
    gl->addWidget(new QLabel("±Roll (°)"),    3, 2); gl->addWidget(_winRollSpin,   3, 3);
    gl->addWidget(new QLabel("±Pitch (°)"),   4, 0); gl->addWidget(_winPitchSpin,  4, 1);
    gl->addWidget(new QLabel("±Yaw (°)"),     4, 2); gl->addWidget(_winYawSpin,    4, 3);

    gl->setColumnStretch(1, 1); gl->setColumnStretch(3, 1);
}

// ─── Table (shared, 10 cols) ──────────────────────────────────────────────────

void OrbitalScanWidget::buildTablePanel()
{
    _table = new QTableWidget(0, 10);
    _table->setHorizontalHeaderLabels({
        "#", "Pitch°", "Yaw°",
        "dx (mm)", "dy (mm)", "dz (mm)",
        "dRoll°", "dPitch°", "dYaw°",
        "Dist (mm)"
    });
    _table->horizontalHeader()->setSectionResizeMode(QHeaderView::Interactive);
    _table->horizontalHeader()->setDefaultSectionSize(52);
    _table->horizontalHeader()->setStretchLastSection(true);
    _table->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    _table->verticalHeader()->setVisible(false);
    _table->setSelectionBehavior(QAbstractItemView::SelectRows);
    _table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    _table->setAlternatingRowColors(true);
    _table->setMinimumWidth(180);
    _table->setStyleSheet(
        "QTableWidget{"
        "  background:#0d1117;color:#cdd;gridline-color:#1e2830;"
        "  font-size:11px;font-family:'Consolas','Courier New',monospace;"
        "  alternate-background-color:#121820;"
        "  selection-background-color:#1a4060;}"
        "QHeaderView::section{"
        "  background:#1a2030;color:#8aaccc;border:none;"
        "  padding:4px;font-size:11px;border-bottom:1px solid #2a3848;}");
}

// ─── Slots ────────────────────────────────────────────────────────────────────

void OrbitalScanWidget::onModeChanged()
{
    const bool spherical = _sphericalRadio->isChecked();
    _modeStack->setCurrentIndex(spherical ? 0 : 1);
    updatePreviewLabel();
    emit scanModeChanged(spherical ? "Spherical" : "Random 6-DOF");
}

void OrbitalScanWidget::onGenerate()
{
    if (_sphericalRadio->isChecked()) {
        OrbitalParams p;
        p.radius       = _radiusSpin->value();
        p.halfPitchDeg = _halfPitchSpin->value();
        p.halfYawDeg   = _halfYawSpin->value();
        p.halfRollDeg  = _halfRollSpin->value();
        p.yawStepDeg   = _yawStepSizeSpin->value();
        p.pitchStepDeg = _pitchStepSizeSpin->value();
        p.nFrames      = _nFrames;
        p.settleMs     = _settleMs;
        _waypoints = OrbitalTrajectory::generate(p);

        emit logMessage(
            QString("Spherical trajectory: %1 waypoints "
                    "(±yaw %2° Δ%3°, R=%4 mm)")
                .arg(_waypoints.size())
                .arg(p.halfYawDeg, 0,'f',0).arg(p.yawStepDeg, 0,'f',0)
                .arg(p.radius,     0,'f',0));
    } else {
        RandomWindowParams p;
        p.nPoses    = _nPosesSpin->value();
        p.seed      = _seedSpin->value();
        p.winX      = _winXSpin->value();
        p.winY      = _winYSpin->value();
        p.winZ      = _winZSpin->value();
        p.xyzStep   = _xyzStepSpin->value();
        p.winRoll   = _winRollSpin->value();
        p.winPitch  = _winPitchSpin->value();
        p.winYaw    = _winYawSpin->value();
        p.rotStep   = _rotStepSpin->value();
        p.nFrames   = _nFrames;
        p.settleMs  = _settleMs;
        _waypoints = OrbitalTrajectory::generateRandom(p);

        emit logMessage(
            QString("Random 6-DOF: %1/%2 poses (seed=%3) "
                    "XYZ ±[%4,%5,%6] mm @%7mm  RPY ±[%8,%9,%10]° @%11°")
                .arg(_waypoints.size() - 1).arg(p.nPoses).arg(p.seed)
                .arg(p.winX,0,'f',0).arg(p.winY,0,'f',0).arg(p.winZ,0,'f',0)
                .arg(p.xyzStep)
                .arg(p.winRoll,0,'f',0).arg(p.winPitch,0,'f',0).arg(p.winYaw,0,'f',0)
                .arg(p.rotStep));
    }

    refreshTable();
    _loadScanBtn->setEnabled(!_waypoints.isEmpty());
    updatePreviewLabel();
}

void OrbitalScanWidget::onLoadToScan()
{
    if (_waypoints.isEmpty()) return;
    emit waypointsReady(_waypoints);
    emit logMessage(
        QString("Loaded %1 waypoints into scan sequence").arg(_waypoints.size()));
}

void OrbitalScanWidget::onTableCellClicked(int row, int /*col*/)
{
    if (row < 0 || row >= _waypoints.size()) return;
    const auto &wp = _waypoints[row];
    emit logMessage(
        QString("WP %1: pitch=%2° yaw=%3° | "
                "dx=%4 dy=%5 dz=%6 | dR=%7 dP=%8 dY=%9")
            .arg(row + 1)
            .arg(wp.pitchDeg, 0,'f',1).arg(wp.yawDeg, 0,'f',1)
            .arg(wp.dx,       0,'f',1).arg(wp.dy,      0,'f',1).arg(wp.dz, 0,'f',1)
            .arg(wp.droll,    0,'f',2).arg(wp.dpitch,  0,'f',2).arg(wp.dyaw, 0,'f',2));
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

void OrbitalScanWidget::refreshTable()
{
    _table->setRowCount(0);
    _table->setRowCount(_waypoints.size());

    const bool isSpherical = _sphericalRadio->isChecked();

    for (int i = 0; i < _waypoints.size(); ++i) {
        const auto &wp = _waypoints[i];
        const double dist = (i == 0)
                                ? 0.0
                                : std::sqrt(wp.dx*wp.dx + wp.dy*wp.dy + wp.dz*wp.dz);

        auto mkItem = [](const QString &text, const QString &color = "") {
            auto *item = new QTableWidgetItem(text);
            item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            if (!color.isEmpty()) item->setForeground(QColor(color));
            return item;
        };

        QString rowColor;
        const int band = isSpherical ? wp.row : (i / 5);
        switch (band % 5) {
        case 0: rowColor = "#7ec8e3"; break;
        case 1: rowColor = "#b5ead7"; break;
        case 2: rowColor = "#ffb7b2"; break;
        case 3: rowColor = "#ffd6a5"; break;
        default: rowColor = "#c9b1ff"; break;
        }

        if (i == 0) rowColor = "#f0e68c";

        _table->setItem(i, 0, mkItem(i == 0 ? "Home" : QString::number(i), rowColor));
        _table->setItem(i, 1, mkItem(QString::number(wp.pitchDeg, 'f', 1) + "°", i == 0 ? "" : rowColor));
        _table->setItem(i, 2, mkItem(QString::number(wp.yawDeg,   'f', 1) + "°", i == 0 ? "" : rowColor));
        _table->setItem(i, 3, mkItem(QString::number(wp.dx,       'f', 1)));
        _table->setItem(i, 4, mkItem(QString::number(wp.dy,       'f', 1)));
        _table->setItem(i, 5, mkItem(QString::number(wp.dz,       'f', 1)));
        _table->setItem(i, 6, mkItem(QString::number(wp.droll,    'f', 1)));
        _table->setItem(i, 7, mkItem(QString::number(wp.dpitch,   'f', 1)));
        _table->setItem(i, 8, mkItem(QString::number(wp.dyaw,     'f', 1)));

        QString distStr = (i == 0) ? "—" : QString::number(dist, 'f', 1);
        _table->setItem(i, 9, mkItem(distStr, dist > 250.0 ? "#ffb7b2" : ""));
    }
}

void OrbitalScanWidget::updatePreviewLabel()
{
    if (_sphericalRadio && _sphericalRadio->isChecked()) {
        double halfYaw = _halfYawSpin     ? _halfYawSpin->value()     : 45.0;
        double yawStep = _yawStepSizeSpin ? _yawStepSizeSpin->value() : 15.0;
        int nSteps = (halfYaw < 1e-9 || yawStep < 1e-9)
                         ? 0 : qMax(1, qRound(halfYaw / yawStep));
        int total  = 1 + 2 * nSteps;
        int frames = total * _nFrames;
        _statsLabel->setText(
            QString("→ %1 offsets + Home = %2 pts × %3 f = %4 total")
                .arg(2*nSteps).arg(total)
                .arg(_nFrames)
                .arg(frames));
    } else {
        int nPoses = _nPosesSpin ? _nPosesSpin->value() : 20;
        int frames = _nFrames;
        _statsLabel->setText(
            QString("→ %1 random poses + Home = %2 pts × %3 f = %4 total")
                .arg(nPoses).arg(nPoses+1).arg(frames).arg((nPoses+1)*frames));
    }
}