#ifndef NETPROTOCOL_H
#define NETPROTOCOL_H

#include <QObject>
#include <QTcpSocket>
#include <QByteArray>
#include <QMutex>
#include <stdint.h>

#pragma pack(push, 1)
struct NetProtocolHeader {
    uint8_t  m_header;
    uint8_t  m_version;
    uint16_t m_number;
    uint32_t m_length;
    uint16_t m_type;
    uint8_t  m_reserved[6];
};
#pragma pack(pop)

class NetProtocol : public QObject
{
    Q_OBJECT
public:
    explicit NetProtocol(QObject *parent = nullptr);
    ~NetProtocol();

    bool connectHost(const QString &ip, quint16 port, int timeoutMs = 5000);
    void disconnectHost();
    bool isConnected() const;

    bool request2response(const QByteArray &requestBody, uint16_t type, QByteArray &outPayload,
                          int headerTimeoutMs = 8000, int bodyTimeoutMs = 12000,
                          QString *err = nullptr);

signals:
    void disconnected();

private slots:
    void slotReadyRead();
    void slotDisconnected();

private:
    int writeAll(const QByteArray &data);
    bool readExact(QByteArray &out, int needBytes, int timeoutMs);
    bool readHeader(QByteArray &head, int timeoutMs, QString *err);
    int unpackHead(const QByteArray &head, QString *err) const;
    int request(const QByteArray &body, uint16_t type);
    QByteArray response(int headerTimeoutMs, int bodyTimeoutMs, QString *err);
    void clearReadBuffer();

private:
    QTcpSocket *_socket = nullptr;
    QByteArray _readBuffer;
    QMutex _mutex;
    QMutex _syncMutex;
};

#endif