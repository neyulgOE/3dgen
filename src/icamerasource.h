#ifndef ICAMERASOURCE_H
#define ICAMERASOURCE_H

#include "types.h"
#include <QObject>

// ─────────────────────────────────────────────────────────────────────────────
// ICameraSource — abstract interface cho mọi nguồn camera depth.
//
// Chỉ expose: connect/disconnect, fetchCount (dùng trong scan sequence).
// Trả về RGBDFrame, việc deproject sẽ do tầng trên (ScanController) lo.
// ─────────────────────────────────────────────────────────────────────────────

class ICameraSource : public QObject
{
    Q_OBJECT

public:
    explicit ICameraSource(QObject *parent = nullptr) : QObject(parent) {}
    ~ICameraSource() override = default;

    virtual bool isConnected() const = 0;

public slots:
    virtual void init          () = 0;
    virtual void connectHost   (const QString &ip, quint16 port) = 0;
    virtual void disconnectHost() = 0;

    // Chụp n frame tuần tự, cách nhau intervalMs.
    // Emit pointCloudReady mỗi frame; emit fetchCountFinished sau frame cuối.
    virtual void fetchCount  (int n, int intervalMs) = 0;
    virtual void fetchContinuous(int intervalMs) { Q_UNUSED(intervalMs); }

    virtual void stopFetch() = 0;

signals:
    void rgbdFrameReady     (RGBDFrame frame);
    void imuDataReady       (QVector<IMUData> imuFrames);
    void fetchCountFinished ();

    void connectionChanged  (bool connected);
    void logMessage         (const QString &msg);
    void errorOccurred      (const QString &err);
};

#endif // ICAMERASOURCE_H