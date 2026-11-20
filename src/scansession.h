#ifndef SCANSESSION_H
#define SCANSESSION_H

#include "types.h"

#include <QList>
#include <QObject>

// ─────────────────────────────────────────────────────────────────────────────
// ScanSession — state machine cho stop-and-scan sequence.
//
// Flow per waypoint (đã fix: settle timer chỉ start SAU khi robot xác nhận
// đã đến nơi, không chạy song song với blocking robot move):
//
//   advanceWaypoint()
//     → emit requestMove(wp)
//     → _waitingMove = true            ← block ở đây
//
//   notifyMoveDone()                   ← AppController gọi khi robot xong
//     → QTimer::singleShot(settleMs)
//       → onSettled()
//         → emit requestSnapshot()
//         → _waitingSnapshot = true
//
//   notifySnapshotDone()               ← AppController gọi sau snapshot
//     → emit requestFetch() × nFrames
//
//   notifyFrameCaptured()              ← AppController gọi mỗi khi nhận cloud
//     → khi đủ nFrames → advanceWaypoint()
// ─────────────────────────────────────────────────────────────────────────────

class ScanSession : public QObject
{
    Q_OBJECT

public:
    explicit ScanSession(QObject *parent = nullptr);

    // ── Waypoint management ───────────────────────────────────────────────────
    void addWaypoint   (const ScanWaypoint &wp);
    void clearWaypoints();
    int  waypointCount () const { return _waypoints.size(); }

    // ── Session control ───────────────────────────────────────────────────────
    void start(int nFrames, int settleMs, int intervalMs);
    void stop ();

    bool isRunning() const { return _running; }

    // ── Callbacks từ AppController ────────────────────────────────────────────

    /// Gọi sau khi robot đã thực sự đến vị trí waypoint (Home + Offset xong).
    /// CHỈ LÚC ĐÓ settle timer mới bắt đầu đếm.
    void notifyMoveDone();

    /// Gọi sau khi AppController đã snapshot pose thành công.
    void notifySnapshotDone();

    /// Gọi sau mỗi lần camera emit pointCloudReady trong lúc scan.
    void notifyFrameCaptured();

signals:
    void requestMove(double dx, double dy, double dz,
                     double droll, double dpitch, double dyaw);
    void requestSnapshot();
    void requestFetch();
    void captureStarted(int orbitalWpIndex);
    void progressUpdated(int current, int total, const QString &status);
    void finished();
    void logMessage(const QString &msg);

private slots:
    void onSettled    ();
    void scheduleFetch();

private:
    void advanceWaypoint();

    QList<ScanWaypoint> _waypoints;

    bool _running           = false;
    int  _currentIdx        = -1;
    int  _framesPerWaypoint = 5;
    int  _settleMs          = 400;
    int  _intervalMs        = 150;
    int  _framesRemaining   = 0;
    int  _currentWpFrames   = 0;
    bool _waitingMove       = false;   // ← MỚI: đang chờ robot di chuyển xong
    bool _waitingSnapshot   = false;
};

#endif // SCANSESSION_H