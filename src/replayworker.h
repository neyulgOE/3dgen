#ifndef REPLAYWORKER_H
#define REPLAYWORKER_H

#include "icamerasource.h"
#include <QTimer>
#include <QDir>
#include <QStringList>

class ReplayWorker : public ICameraSource
{
    Q_OBJECT
public:
    explicit ReplayWorker(QObject *parent = nullptr);
    ~ReplayWorker() override;

    bool isConnected() const override;

public slots:
    void init() override;
    void connectHost(const QString &ip, quint16 port) override; // ip = dir path
    void disconnectHost() override;

    void fetchCount(int n, int intervalMs) override;
    void fetchContinuous(int intervalMs) override;
    void stopFetch() override;

private slots:
    void onTimeout();

private:
    bool _connected = false;
    bool _fetching = false;
    QTimer _timer;
    QDir _recordDir;
    QStringList _rgbFiles;
    int _currentIndex = 0;
    int _remainingFetches = 0;
};

#endif // REPLAYWORKER_H
