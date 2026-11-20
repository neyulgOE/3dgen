#include "cameraworker.h"
#include <QTimer>
#include <QThread>

CameraWorker::CameraWorker()
    : ICameraSource(nullptr)
{}

CameraWorker::~CameraWorker()
{
    if (_tmv) {
        _tmv->disconnectHost();
        delete _tmv;
        _tmv = nullptr;
    }
}

void CameraWorker::init()
{
    Q_ASSERT(_tmv == nullptr);
    _tmv = new TMVisionaryClient(this);

    connect(_tmv, &TMVisionaryClient::logMessage,
            this, &CameraWorker::logMessage);
    connect(_tmv, &TMVisionaryClient::connectionChanged,
            this, &CameraWorker::connectionChanged);

    emit logMessage("CameraWorker ready on thread " +
                    QString::number(reinterpret_cast<quintptr>(QThread::currentThread()), 16));
}

// ─── Connection ───────────────────────────────────────────────────────────────

void CameraWorker::connectHost(const QString &ip, quint16 port)
{
    if (!_tmv) { emit errorOccurred("Worker not initialised"); return; }
    QString err;
    if (!_tmv->connectHost(ip, port, err))
        emit errorOccurred(err);
}

void CameraWorker::disconnectHost()
{
    if (_tmv) _tmv->disconnectHost();
}

bool CameraWorker::isConnected() const
{
    return _tmv && _tmv->isConnected();
}

// ─── fetchCount ───────────────────────────────────────────────────────────────

void CameraWorker::fetchCount(int n, int intervalMs)
{
    if (n <= 0) { emit fetchCountFinished(); return; }
    _fetchCountRemaining = n;
    _fetchCountInterval  = qMax(50, intervalMs);
    QTimer::singleShot(0, this, &CameraWorker::doFetchCount);
}

void CameraWorker::stopFetch()
{
    _fetchCountRemaining = 0;
}

// ─── doFetchCount ─────────────────────────────────────────────────────────────

void CameraWorker::doFetchCount()
{
    if (_fetchCountRemaining <= 0) {
        emit fetchCountFinished();
        return;
    }

    if (!_tmv || !_tmv->isConnected()) {
        emit errorOccurred("fetchCount: connection lost");
        emit fetchCountFinished();
        return;
    }

    RGBDFrame f;
    QString err;
    _tmv->requestRGBDFrame(f, err);

    if (f.depth.valid && !_cloudPending.exchange(true)) {
        emit rgbdFrameReady(f);
    }
    if (!err.isEmpty()) emit errorOccurred("RGBD Error: " + err);

    --_fetchCountRemaining;
    if (_fetchCountRemaining > 0)
        QTimer::singleShot(_fetchCountInterval, this, &CameraWorker::doFetchCount);
    else
        emit fetchCountFinished();
}