#include "netprotocol.h"
#include <QElapsedTimer>
#include <QMutexLocker>
#include <QCoreApplication>

NetProtocol::NetProtocol(QObject *parent)
    : QObject(parent), _socket(new QTcpSocket(this))
{
    connect(_socket, &QTcpSocket::readyRead, this, &NetProtocol::slotReadyRead);
    connect(_socket, &QTcpSocket::disconnected, this, &NetProtocol::slotDisconnected);
}

NetProtocol::~NetProtocol()
{
    disconnectHost();
}

bool NetProtocol::connectHost(const QString &ip, quint16 port, int timeoutMs)
{
    if (_socket->state() == QAbstractSocket::ConnectedState) {
        _socket->disconnectFromHost();
        _socket->abort();
    }
    _socket->connectToHost(ip, port, QTcpSocket::ReadWrite, QTcpSocket::IPv4Protocol);
    return _socket->waitForConnected(timeoutMs);
}

void NetProtocol::disconnectHost()
{
    if (_socket->isOpen()) {
        _socket->close();
    }
    _socket->abort();
    QMutexLocker lock(&_mutex);
    _readBuffer.clear();
}

void NetProtocol::clearReadBuffer()
{
    QMutexLocker lock(&_mutex);
    if (_socket) {
        // Hoàn toàn xả sạch OS buffer
        while (_socket->waitForReadyRead(10)) {
            _socket->readAll();
        }
        _socket->readAll(); 
    }
    _readBuffer.clear();
}

bool NetProtocol::isConnected() const
{
    return _socket->state() == QAbstractSocket::ConnectedState;
}

void NetProtocol::slotReadyRead()
{
    QMutexLocker lock(&_mutex);
    _readBuffer += _socket->readAll();
}

void NetProtocol::slotDisconnected()
{
    emit disconnected();
}

int NetProtocol::writeAll(const QByteArray &data)
{
    if (_socket->state() != QAbstractSocket::ConnectedState) {
        return -1;
    }
    const qint64 written = _socket->write(data);
    if (written <= 0) {
        return static_cast<int>(written);
    }
    _socket->flush();
    return static_cast<int>(written);
}

int NetProtocol::request(const QByteArray &body, uint16_t type)
{
    NetProtocolHeader header{};
    header.m_header = 0x5A;
    header.m_version = 0x01;
    header.m_number = 0;
    header.m_length = static_cast<uint32_t>(body.size());
    header.m_type = type;

    QByteArray packet;
    packet.reserve(static_cast<int>(sizeof(NetProtocolHeader)) + body.size());
    packet.append(reinterpret_cast<const char *>(&header), static_cast<int>(sizeof(NetProtocolHeader)));
    packet.append(body);
    return writeAll(packet);
}

int NetProtocol::unpackHead(const QByteArray &head, QString *err) const
{
    if (head.size() < static_cast<int>(sizeof(NetProtocolHeader))) {
        if (err) *err = "Header too short";
        return -1;
    }
    const auto *h = reinterpret_cast<const NetProtocolHeader *>(head.constData());
    if (h->m_header != 0x5A) {
        if (err) *err = "Invalid header magic";
        return -1;
    }
    if (h->m_version != 0x01) {
        if (err) *err = "Invalid protocol version";
        return -1;
    }
    return static_cast<int>(h->m_length);
}

bool NetProtocol::readExact(QByteArray &out, int needBytes, int timeoutMs)
{
    out.clear();
    if (needBytes <= 0) {
        return true;
    }
    if (_socket->state() != QAbstractSocket::ConnectedState) {
        return false;
    }

    QElapsedTimer timer;
    timer.start();
    while (out.size() < needBytes) {
        {
            QMutexLocker lock(&_mutex);
            const int take = qMin(needBytes - out.size(), _readBuffer.size());
            if (take > 0) {
                out += _readBuffer.left(take);
                _readBuffer.remove(0, take);
            }
        }
        if (out.size() >= needBytes) {
            break;
        }
        const int remain = timeoutMs - static_cast<int>(timer.elapsed());
        if (remain <= 0) {
            return false;
        }
        _socket->waitForReadyRead(qMin(remain, 50));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    }
    return true;
}

bool NetProtocol::readHeader(QByteArray &head, int timeoutMs, QString *err)
{
    head.clear();
    if (_socket->state() != QAbstractSocket::ConnectedState) {
        if (err) *err = "Socket not connected";
        return false;
    }

    const int headerSize = static_cast<int>(sizeof(NetProtocolHeader));
    QElapsedTimer timer;
    timer.start();

    while (timer.elapsed() < timeoutMs) {
        {
            QMutexLocker lock(&_mutex);
            const int magicIdx = _readBuffer.indexOf(char(0x5A));
            if (magicIdx > 0) {
                _readBuffer.remove(0, magicIdx);
            } else if (magicIdx < 0 && _readBuffer.size() > headerSize * 4) {
                _readBuffer.clear();
            }
            if (_readBuffer.size() >= headerSize) {
                head = _readBuffer.left(headerSize);
                QString localErr;
                if (unpackHead(head, &localErr) >= 0) {
                    _readBuffer.remove(0, headerSize);
                    return true;
                }
                _readBuffer.remove(0, 1);
                head.clear();
            }
        }

        const int remain = timeoutMs - static_cast<int>(timer.elapsed());
        if (remain <= 0) break;
        _socket->waitForReadyRead(qMin(remain, 50));
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
    }

    if (err) *err = "Read header timeout";
    return false;
}

QByteArray NetProtocol::response(int headerTimeoutMs, int bodyTimeoutMs, QString *err)
{
    QByteArray head;
    if (!readHeader(head, headerTimeoutMs, err)) {
        return {};
    }
    const int len = unpackHead(head, err);
    if (len < 0) {
        return {};
    }
    QByteArray body;
    if (len > 0 && !readExact(body, len, bodyTimeoutMs)) {
        if (err) *err = QString("Read body timeout, need=%1 got=%2").arg(len).arg(body.size());
        return {};
    }
    return body;
}

bool NetProtocol::request2response(const QByteArray &requestBody, uint16_t type, QByteArray &outPayload,
                                   int headerTimeoutMs, int bodyTimeoutMs, QString *err)
{
    QMutexLocker syncLock(&_syncMutex);
    outPayload.clear();
    clearReadBuffer();
    const int written = request(requestBody, type);
    if (written <= 0) {
        if (err) *err = "Write request failed";
        return false;
    }
    outPayload = response(headerTimeoutMs, bodyTimeoutMs, err);
    return err ? err->isEmpty() : !outPayload.isEmpty();
}