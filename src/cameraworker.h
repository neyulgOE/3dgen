#ifndef CAMERAWORKER_H
#define CAMERAWORKER_H

#include "icamerasource.h"
#include "tmvisionaryclient.h"
#include "tmvisionaryclient.h"
#include "types.h"

#include <QMutex>
#include <atomic>

// ─────────────────────────────────────────────────────────────────────────────
// CameraWorker — ICameraSource implementation cho TMVisionary SICK ToF camera.
//
// Chỉ expose fetchCount (dùng trong scan sequence).
// Không còn stream, RGB/Depth image, blend, depth colormap.
// ─────────────────────────────────────────────────────────────────────────────

class CameraWorker : public ICameraSource
{
    Q_OBJECT

public:
    explicit CameraWorker();
    ~CameraWorker() override;

    bool isConnected() const override;

    std::atomic<bool> _cloudPending {false};

public slots:
    void init          () override;
    void connectHost   (const QString &ip, quint16 port) override;
    void disconnectHost() override;
    void fetchCount    (int n, int intervalMs) override;
    void stopFetch() override;

private slots:
    void doFetchCount();

private:
    TMVisionaryClient *_tmv = nullptr;

    int _fetchCountRemaining = 0;
    int _fetchCountInterval  = 100;
};

#endif // CAMERAWORKER_H