#include "scansession.h"
#include <QTimer>

ScanSession::ScanSession(QObject *parent)
    : QObject(parent)
{
}

// ─── Waypoint management ──────────────────────────────────────────────────────

void ScanSession::addWaypoint(const ScanWaypoint &wp)
{
    _waypoints.append(wp);
}

void ScanSession::clearWaypoints()
{
    _waypoints.clear();
    _running         = false;
    _currentIdx      = -1;
    _waitingMove     = false;
    _waitingSnapshot = false;
}

// ─── Session control ──────────────────────────────────────────────────────────

void ScanSession::start(int nFrames, int settleMs, int intervalMs)
{
    if (_waypoints.isEmpty()) return;

    _framesPerWaypoint = nFrames;
    _settleMs          = settleMs;
    _intervalMs        = intervalMs;
    _running           = true;
    _currentIdx        = 0;
    _waitingMove       = false;
    _waitingSnapshot   = false;

    emit logMessage(
        QString("ScanSession start — %1 waypoints, %2 frames each, settle %3 ms")
            .arg(_waypoints.size()).arg(nFrames).arg(settleMs));

    advanceWaypoint();
}

void ScanSession::stop()
{
    if (!_running) return;
    _running         = false;
    _currentIdx      = -1;
    _framesRemaining = 0;
    _waitingMove     = false;
    _waitingSnapshot = false;
    emit progressUpdated(0, _waypoints.size(), "Stopped by user");
    emit logMessage("ScanSession stopped");
}

// ─── Internal state machine ───────────────────────────────────────────────────

void ScanSession::advanceWaypoint()
{
    if (!_running) return;

    if (_currentIdx >= _waypoints.size()) {
        _running    = false;
        _currentIdx = -1;
        emit logMessage("ScanSession complete");
        emit progressUpdated(_waypoints.size(), _waypoints.size(), "Complete");
        emit finished();
        return;
    }

    const ScanWaypoint &wp = _waypoints[_currentIdx];
    const QString status   = QString("Moving to waypoint %1 / %2 …")
                               .arg(_currentIdx + 1)
                               .arg(_waypoints.size());
    emit progressUpdated(_currentIdx, _waypoints.size(), status);
    emit logMessage(status);

    // Yêu cầu AppController di chuyển robot.
    // KHÔNG set timer ở đây — settle timer chỉ bắt đầu sau khi
    // AppController gọi notifyMoveDone() xác nhận robot đã đến nơi.
    _waitingMove = true;
    emit requestMove(wp.dx, wp.dy, wp.dz, wp.droll, wp.dpitch, wp.dyaw);

    // AppController::onScanRequestMove() chạy trên main thread nhưng
    // block _robotThread cho đến khi robot xong, sau đó gọi notifyMoveDone().
}

// ─── notifyMoveDone ───────────────────────────────────────────────────────────
// AppController gọi hàm này sau khi robotDoHome() + robotDoOffset() đều xong.
// Lúc này robot đã thực sự dừng tại waypoint — bắt đầu settle timer.

void ScanSession::notifyMoveDone()
{
    if (!_running || !_waitingMove) return;
    _waitingMove = false;

    emit logMessage(
        QString("Move done for waypoint %1 — settling %2 ms …")
            .arg(_currentIdx + 1)
            .arg(_settleMs));

    // Settle delay: chờ rung cơ học tắt dần trước khi snapshot/chụp.
    // Nếu _settleMs == 0, onSettled() vẫn được gọi qua singleShot(0)
    // để giữ flow không đồng bộ nhất quán.
    QTimer::singleShot(_settleMs, this, &ScanSession::onSettled);
}

// ─── onSettled ────────────────────────────────────────────────────────────────

void ScanSession::onSettled()
{
    if (!_running) return;

    emit logMessage(
        QString("Waypoint %1 settled — requesting pose snapshot")
            .arg(_currentIdx + 1));

    _waitingSnapshot = true;
    emit requestSnapshot();
    // AppController gọi notifySnapshotDone() sau khi snapshot xong.
}

// ─── notifySnapshotDone ───────────────────────────────────────────────────────

void ScanSession::notifySnapshotDone()
{
    if (!_running || !_waitingSnapshot) return;
    _waitingSnapshot = false;

    const ScanWaypoint &wp = _waypoints[_currentIdx];

    _currentWpFrames = (wp.nFrames > 0) ? wp.nFrames : _framesPerWaypoint;
    _framesRemaining = _currentWpFrames;

    if (wp.nFrames == 0) {
        // Home intermediate — không chụp, advance ngay.
        emit logMessage(
            QString("Waypoint %1 (Home intermediate) — skip capture, advancing")
                .arg(_currentIdx + 1));
        ++_currentIdx;
        QTimer::singleShot(0, this, &ScanSession::advanceWaypoint);
        return;
    }

    emit captureStarted(wp.orbitalWpIndex);

    emit logMessage(
        QString("Snapshot done — capturing %1 frames at waypoint %2 (orbital idx %3)")
            .arg(_framesRemaining).arg(_currentIdx + 1).arg(wp.orbitalWpIndex));

    scheduleFetch();
}

// ─── scheduleFetch ────────────────────────────────────────────────────────────

void ScanSession::scheduleFetch()
{
    if (!_running) return;

    if (_framesRemaining <= 0) {
        ++_currentIdx;
        advanceWaypoint();
        return;
    }

    emit requestFetch();
    // notifyFrameCaptured() được gọi bởi AppController khi nhận cloud.
}

// ─── notifyFrameCaptured ─────────────────────────────────────────────────────

void ScanSession::notifyFrameCaptured()
{
    if (!_running) return;

    --_framesRemaining;

    const QString status =
        QString("Waypoint %1/%2 — frame %3/%4")
            .arg(_currentIdx + 1).arg(_waypoints.size())
            .arg(_currentWpFrames - _framesRemaining)
            .arg(_currentWpFrames);
    emit progressUpdated(_currentIdx, _waypoints.size(), status);

    if (_framesRemaining > 0)
        QTimer::singleShot(_intervalMs, this, &ScanSession::scheduleFetch);
    else {
        ++_currentIdx;
        QTimer::singleShot(0, this, &ScanSession::advanceWaypoint);
    }
}