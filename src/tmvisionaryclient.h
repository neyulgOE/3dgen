#ifndef TMVISIONARYCLIENT_H
#define TMVISIONARYCLIENT_H

#include "netprotocol.h"
#include "types.h"
#include "api1202parser.h"
#include <QObject>
#include <QStringList>

// ─── Confirmed protocol type codes ───────────────────────────────────────────
//   1200 = camera_depth  (z16, uint16 LE, scale=0.001 m/unit → ×1000 = mm)
//   1201 = camera_rgb bgr8 (CONFIRMED)
//   1202 = aruco_detections (JSON)
namespace TMVCmd {
constexpr uint16_t DepthFrame  = 1200;
constexpr uint16_t RGBFrame    = 1201;
constexpr uint16_t ArUco1202   = 1202;
constexpr uint16_t Safety3D    = 1000;
}

class TMVisionaryClient : public QObject
{
    Q_OBJECT
public:
    explicit TMVisionaryClient(QObject *parent = nullptr);

    bool connectHost(const QString &ip, quint16 port, QString &err);
    void disconnectHost();
    bool isConnected() const;

    bool probeServer(QString &err);

    bool requestRGBFrame  (RGBFrame   &out, QString &err);
    bool requestDepthFrame(DepthFrame &out, QString &err);
    bool requestRGBDFrame (RGBDFrame  &out, QString &err);
    bool requestAruco1202 (MarkerFrame1202 &frame, QString &err);

    bool requestDeviceInfo(QString &infoJson, QString &err);

signals:
    void connectionChanged(bool connected);
    void logMessage(const QString &msg);

private:
    // ── THAY ĐỔI: thêm QStringList* logs để collect log thay vì emit trực tiếp.
    // Khi gọi từ QtConcurrent thread (requestRGBDFrame), truyền &logs rồi emit
    // sau khi join — tránh data race trên emit.
    // Khi gọi từ worker thread bình thường, truyền nullptr → emit ngay như cũ.
    bool parseFramePayload(const QByteArray &payload,
                           RGBFrame   *rgb,
                           DepthFrame *depth,
                           const QString &context,
                           QString &err,
                           QStringList *logs = nullptr);

    // Helper: emit hoặc append tùy logs có nullptr không
    void log(const QString &msg, QStringList *logs);

    NetProtocol _netRgb;
    NetProtocol _netDepth;

    QString _lastIp;
    quint16 _lastPort = 0;
};

#endif // TMVISIONARYCLIENT_H