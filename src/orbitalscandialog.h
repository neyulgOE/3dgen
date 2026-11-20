#ifndef ORBITALSCANDIALOG_H
#define ORBITALSCANDIALOG_H

#include "orbitaltrajectory.h"
#include "types.h"

#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QTableWidget>
#include <QWidget>

class QRadioButton;

// ─────────────────────────────────────────────────────────────────────────────
// OrbitalScanWidget — Measure tab, hai chế độ:
//
//  • Spherical  — quỹ đạo cầu phẳng xOz (chế độ cũ)
//  • Random     — 6-DOF random trong cửa sổ ± per DOF (grid-quantised)
//
// Cả hai chế độ dùng chung một bảng 10 cột thống nhất.
// Timing (nFrames, settleMs) nhận từ MainWindow qua setTimingParams().
// ─────────────────────────────────────────────────────────────────────────────

class OrbitalScanWidget : public QWidget
{
    Q_OBJECT

public:
    explicit OrbitalScanWidget(QWidget *parent = nullptr);

    const QVector<OrbitalWaypoint> &waypoints() const { return _waypoints; }

    // Gọi từ MainWindow mỗi lần user thay đổi timing
    void setTimingParams(int nFrames, int settleMs);

signals:
    void waypointsReady(QVector<OrbitalWaypoint> wps);
    void logMessage(const QString &msg);
    void scanModeChanged(const QString &modeName);

private slots:
    void onGenerate();
    void onLoadToScan();
    void onTableCellClicked(int row, int col);
    void onModeChanged();

private:
    void buildModeSelector();
    void buildSphericalPanel();
    void buildRandomPanel();
    void buildTablePanel();
    void refreshTable();
    void updatePreviewLabel();

    // ── Mode selector ─────────────────────────────────────────────────────────
    QRadioButton   *_sphericalRadio = nullptr;
    QRadioButton   *_randomRadio    = nullptr;
    QStackedWidget *_modeStack      = nullptr;

    // ── Spherical params ──────────────────────────────────────────────────────
    QWidget        *_sphericalPanel    = nullptr;
    QDoubleSpinBox *_radiusSpin        = nullptr;
    QDoubleSpinBox *_halfPitchSpin     = nullptr;
    QDoubleSpinBox *_halfYawSpin       = nullptr;
    QDoubleSpinBox *_halfRollSpin      = nullptr;
    QDoubleSpinBox *_pitchStepSizeSpin = nullptr;
    QDoubleSpinBox *_yawStepSizeSpin   = nullptr;

    // ── Random Window params ──────────────────────────────────────────────────
    QWidget        *_randomPanel   = nullptr;
    QSpinBox       *_nPosesSpin    = nullptr;
    QSpinBox       *_seedSpin      = nullptr;

    // Translation window ±
    QDoubleSpinBox *_winXSpin      = nullptr;
    QDoubleSpinBox *_winYSpin      = nullptr;
    QDoubleSpinBox *_winZSpin      = nullptr;
    QSpinBox       *_xyzStepSpin   = nullptr;   // mm

    // Rotation window ±
    QDoubleSpinBox *_winRollSpin   = nullptr;
    QDoubleSpinBox *_winPitchSpin  = nullptr;
    QDoubleSpinBox *_winYawSpin    = nullptr;
    QSpinBox       *_rotStepSpin   = nullptr;   // deg

    // ── Timing — nhận từ MainWindow, không spinbox riêng ─────────────────────
    int _nFrames  = 5;
    int _settleMs = 400;

    // ── Buttons + stats ───────────────────────────────────────────────────────
    QPushButton *_generateBtn = nullptr;
    QPushButton *_loadScanBtn = nullptr;
    QLabel      *_statsLabel  = nullptr;

    // ── Table ─────────────────────────────────────────────────────────────────
    QTableWidget *_table = nullptr;

    // ── Data ──────────────────────────────────────────────────────────────────
    QVector<OrbitalWaypoint> _waypoints;
};

#endif // ORBITALSCANDIALOG_H