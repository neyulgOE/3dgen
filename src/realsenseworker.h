#ifndef REALSENSEWORKER_H
#define REALSENSEWORKER_H

#include "icamerasource.h"
#include <librealsense2/rs.hpp>
#include <QVector>
#include <QMutex>

class RealSenseWorker : public ICameraSource
{
    Q_OBJECT

public:
    explicit RealSenseWorker(QObject *parent = nullptr);
    ~RealSenseWorker() override;

    bool isConnected() const override;

public slots:
    void init() override;
    void connectHost(const QString &ip, quint16 port) override;
    void disconnectHost() override;
    void fetchCount(int n, int intervalMs) override;
    void fetchContinuous(int intervalMs);
    void stopFetch() override;

private:
    rs2::pipeline _pipeline;
    rs2::config _cfg;
    rs2::align _align_to;
    rs2::syncer _syncer;
    
    QVector<IMUData> _imuBuffer;
    QMutex _imuMutex;
    
    bool _connected = false;
    bool _fetching = false;

    int _remainingFetches = 0;
    int _fetchIntervalMs = 0;

private slots:
    void processNextFrame();
};

#endif // REALSENSEWORKER_H
