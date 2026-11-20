#include "tmvisionaryclient.h"
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QImage>
#include <QtEndian>
#include <limits>
#include <cstring>

TMVisionaryClient::TMVisionaryClient(QObject *parent)
    : QObject(parent), _netRgb(this), _netDepth(this)
{
}

// ─── Helper: emit hoặc collect log ────────────────────────────────────────────
// Khi logs == nullptr  → emit ngay (single-thread path)
// Khi logs != nullptr  → append vào list, caller emit sau khi join (parallel path)
void TMVisionaryClient::log(const QString &msg, QStringList *logs)
{
    if (logs)
        logs->append(msg);
    else
        emit logMessage(msg);
}

// ─── Connection ───────────────────────────────────────────────────────────────

bool TMVisionaryClient::connectHost(const QString &ip, quint16 port, QString &err)
{
    err.clear();
    _lastIp   = ip;
    _lastPort = port;

    // Sequential — QTcpSocket không thread-safe, không dùng QtConcurrent.
    // Connect lần lượt; timeout 5s mỗi socket, tổng tối đa 10s.
    bool okRgb   = _netRgb.connectHost(ip, port, 5000);
    bool okDepth = _netDepth.connectHost(ip, port, 5000);

    if (!okRgb || !okDepth) {
        if (okRgb)   _netRgb.disconnectHost();
        if (okDepth) _netDepth.disconnectHost();
        err = QString("Failed to connect TMVisionary at %1:%2 (RGB=%3, Depth=%4)")
                  .arg(ip).arg(port)
                  .arg(okRgb   ? "OK" : "FAIL")
                  .arg(okDepth ? "OK" : "FAIL");
        return false;
    }

    emit logMessage(QString("TMVisionary connected (2 sockets): %1:%2").arg(ip).arg(port));
    emit connectionChanged(true);
    return true;
}

void TMVisionaryClient::disconnectHost()
{
    _netRgb.disconnectHost();
    _netDepth.disconnectHost();
    emit connectionChanged(false);
    emit logMessage("TMVisionary disconnected");
}

bool TMVisionaryClient::isConnected() const
{
    return _netRgb.isConnected() && _netDepth.isConnected();
}

// ─── Shared frame payload parser ─────────────────────────────────────────────
//
// Format: [uint32 LE: json_len] [JSON bytes] [binary pixel data]
//
// logs == nullptr  → emit logMessage trực tiếp (single-thread, safe)
// logs != nullptr  → append vào list thay vì emit (parallel-thread, no race)

bool TMVisionaryClient::parseFramePayload(const QByteArray &payload,
                                          RGBFrame   *rgb,
                                          DepthFrame *depth,
                                          const QString &ctx,
                                          QString &err,
                                          QStringList *logs)
{
    if (payload.size() < 8) {
        err = QString("%1: payload too short (%2 B)").arg(ctx).arg(payload.size());
        return false;
    }

    const uchar *raw = reinterpret_cast<const uchar*>(payload.constData());

    quint32 jsonLen = qFromLittleEndian<quint32>(raw);
    if (jsonLen == 0 || jsonLen > 65536 || payload.size() < 4 + (int)jsonLen) {
        err = QString("%1: invalid json_len=%2 (payload %3 B)")
        .arg(ctx).arg(jsonLen).arg(payload.size());
        return false;
    }

    const QByteArray jsonBytes = payload.mid(4, jsonLen);
    QJsonParseError jsonErr;
    const QJsonDocument doc = QJsonDocument::fromJson(jsonBytes, &jsonErr);
    if (jsonErr.error != QJsonParseError::NoError || !doc.isObject()) {
        err = QString("%1: JSON error: %2").arg(ctx).arg(jsonErr.errorString());
        return false;
    }
    const QJsonObject meta = doc.object();

    if (meta.contains("error")) {
        err = ctx + " server error: " + meta["error"].toString();
        return false;
    }

    const int     width  = meta["width"].toInt();
    const int     height = meta["height"].toInt();
    const quint64 ts     = static_cast<quint64>(
        meta.value("pub_nsec").toVariant().toULongLong());
    const QString fmtStr = meta.value("payload_format").toString();
    const QString codec  = meta.value("codec").toString();
    const QString type   = meta.value("type").toString();
    const double  dscale = meta.value("depth_scale").toDouble(0.001);

    log(QString("%1: type='%2' fmt='%3' codec='%4' %5×%6 scale=%7")
            .arg(ctx).arg(type).arg(fmtStr).arg(codec)
            .arg(width).arg(height).arg(dscale), logs);

    if (width <= 0 || height <= 0) {
        err = QString("%1: invalid dims %2×%3").arg(ctx).arg(width).arg(height);
        return false;
    }

    const QByteArray tail = payload.mid(4 + jsonLen);
    bool isDepthPayload = (fmtStr == "z16" || type.contains("depth", Qt::CaseInsensitive));

    // ── Depth path ────────────────────────────────────────────────────────────
    if (depth && isDepthPayload) {
        const int pixels = width * height;
        const int needed = pixels * 2;
        if (tail.size() < needed) {
            err = QString("%1: z16 tail too short (%2 B, need %3)")
            .arg(ctx).arg(tail.size()).arg(needed);
            return false;
        }
        depth->width        = width;
        depth->height       = height;
        depth->timestamp_ns = ts;
        depth->z_mm_data.resize(pixels);
        depth->z_min =  1e9f;
        depth->z_max = -1e9f;

        const float mmPerUnit = static_cast<float>(dscale * 1000.0);
        const uchar *src = reinterpret_cast<const uchar*>(tail.constData());
        for (int i = 0; i < pixels; ++i) {
            quint16 v = qFromLittleEndian<quint16>(src + i * 2);
            if (v == 0) {
                depth->z_mm_data[i] = std::numeric_limits<float>::quiet_NaN();
            } else {
                float z = v * mmPerUnit;
                depth->z_mm_data[i] = z;
                if (z < depth->z_min) depth->z_min = z;
                if (z > depth->z_max) depth->z_max = z;
            }
        }
        depth->valid = true;
        log(QString("%1: Depth OK %2×%3 Z=[%4..%5 mm]")
                .arg(ctx).arg(width).arg(height)
                .arg(depth->z_min, 0, 'f', 0)
                .arg(depth->z_max, 0, 'f', 0), logs);
        return true;
    }

    // ── RGB path ──────────────────────────────────────────────────────────────
    if (rgb && !isDepthPayload) {
        rgb->width        = width;
        rgb->height       = height;
        rgb->timestamp_ns = ts;

        bool useJpeg = (codec == "jpeg" || fmtStr == "jpeg");
        if (!useJpeg && tail.size() > 3) {
            const uchar *t = reinterpret_cast<const uchar*>(tail.constData());
            if (t[0] == 0xFF && t[1] == 0xD8 && t[2] == 0xFF) useJpeg = true;
        }

        if (useJpeg) {
            QImage img;
            if (!img.loadFromData(tail, "JPEG")) {
                err = ctx + ": JPEG decode failed";
                return false;
            }
            img = img.convertToFormat(QImage::Format_RGB888);
            rgb->width  = img.width();
            rgb->height = img.height();
            rgb->data.resize(rgb->width * rgb->height * 3);
            for (int y = 0; y < img.height(); ++y)
                memcpy(rgb->data.data() + y * img.width() * 3,
                       img.constScanLine(y), img.width() * 3);
            rgb->valid = true;
            log(QString("%1: RGB JPEG OK %2×%3").arg(ctx).arg(rgb->width).arg(rgb->height), logs);
            return true;
        }

        const int expected = width * height * 3;
        if (tail.size() < expected) {
            err = QString("%1: RGB8 tail too short (%2 B, need %3)")
            .arg(ctx).arg(tail.size()).arg(expected);
            return false;
        }
        if (fmtStr == "bgr8") {
            rgb->data.resize(expected);
            const uchar *src = reinterpret_cast<const uchar*>(tail.constData());
            uchar *dst = reinterpret_cast<uchar*>(rgb->data.data());
            for (int i = 0; i < width * height; ++i) {
                dst[i*3+0] = src[i*3+2];
                dst[i*3+1] = src[i*3+1];
                dst[i*3+2] = src[i*3+0];
            }
        } else {
            rgb->data = tail.left(expected);
        }
        rgb->valid = true;
        log(QString("%1: RGB OK %2×%3").arg(ctx).arg(width).arg(height), logs);
        return true;
    }

    // ── Type mismatch ─────────────────────────────────────────────────────────
    if (depth && !isDepthPayload) {
        err = QString("%1: got '%2' but expected depth — wrong type code?").arg(ctx).arg(type);
        return false;
    }
    if (rgb && isDepthPayload) {
        err = QString("%1: got depth '%2' but expected RGB — wrong type code?").arg(ctx).arg(type);
        return false;
    }

    err = ctx + ": no output target";
    return false;
}

// ─── ArUco frame ──────────────────────────────────────────────────────────────

bool TMVisionaryClient::requestAruco1202(MarkerFrame1202 &frame, QString &err)
{
    QByteArray payload;
    if (!_netRgb.request2response({}, TMVCmd::ArUco1202, payload, 3000, 8000, &err)) {
        return false;
    }

    // payload = 4 bytes length + JSON
    if (payload.size() < 4) {
        err = "ArUco payload too short";
        return false;
    }

    const quint32 jsonLen = qFromLittleEndian<quint32>(payload.constData());
    if (payload.size() < static_cast<int>(4 + jsonLen)) {
        err = "ArUco payload incomplete";
        return false;
    }

    const QByteArray jsonBytes = payload.mid(4, jsonLen);
    return Api1202Parser::parse(jsonBytes, frame, err);
}

// ─── Depth frame ──────────────────────────────────────────────────────────────

bool TMVisionaryClient::requestDepthFrame(DepthFrame &out, QString &err)
{
    out = {};
    QByteArray payload;
    if (!_netDepth.request2response({}, TMVCmd::DepthFrame, payload, 3000, 20000, &err)) {
        emit logMessage(QString("Depth(1200) request failed: %1").arg(err));
        return false;
    }
    // logs=nullptr → emit trực tiếp (gọi từ worker thread, single path)
    return parseFramePayload(payload, nullptr, &out, "Depth1200", err, nullptr);
}

// ─── RGB frame ────────────────────────────────────────────────────────────────

bool TMVisionaryClient::requestRGBFrame(RGBFrame &out, QString &err)
{
    out = {};
    QByteArray payload;
    if (!_netRgb.request2response({}, TMVCmd::RGBFrame, payload, 3000, 20000, &err)) {
        emit logMessage(QString("RGB(1201) request failed (%1 B): HEX=%2")
                            .arg(payload.size())
                            .arg(QString::fromLatin1(payload.left(32).toHex(' '))));
        return false;
    }
    emit logMessage(QString("RGB(1201): %1 B, HEX[0..16]: %2")
                        .arg(payload.size())
                        .arg(QString::fromLatin1(payload.left(16).toHex(' '))));
    return parseFramePayload(payload, &out, nullptr, "RGB1201", err, nullptr);
}

// ─── RGBD — parallel fetch, thread-safe logging ───────────────────────────────
//
// BUG CŨ: parseFramePayload() gọi emit logMessage() trực tiếp từ bên trong
//         QtConcurrent lambda → 2 thread cùng emit cùng lúc → data race/crash.
//
// FIX:    Truyền &logs (QStringList) vào parseFramePayload thay vì nullptr.
//         Parser chỉ append vào list, không emit gì cả.
//         Sau khi fRgb.result() + fDepth.result() join xong (single thread),
//         mình mới emit từng dòng log → hoàn toàn thread-safe.

bool TMVisionaryClient::requestRGBDFrame(RGBDFrame &out, QString &err)
{
    out = {};

    // QtConcurrent::run bị loại bỏ hoàn toàn.
    //
    // Lý do: QTcpSocket KHÔNG thread-safe. _netRgb/_netDepth được tạo trên
    // worker thread; lambda chạy trên Qt global thread-pool → QSocketNotifier
    // bị access từ sai thread → warning spam → UB → crash.
    //
    // Sequential trên cùng worker thread là cách đúng duy nhất với QTcpSocket.

    // ── RGB ───────────────────────────────────────────────────────────────────
    {
        QByteArray payload;
        QString rgbErr;
        bool ok = _netRgb.request2response({}, TMVCmd::RGBFrame, payload, 3000, 20000, &rgbErr);
        if (ok)
            ok = parseFramePayload(payload, &out.rgb, nullptr, "RGB1201", rgbErr, nullptr);
        if (!ok)
            emit logMessage("RGBD: RGB failed — " + rgbErr);
    }

    // ── Depth ─────────────────────────────────────────────────────────────────
    {
        QByteArray payload;
        QString depthErr;
        bool ok = _netDepth.request2response({}, TMVCmd::DepthFrame, payload, 3000, 20000, &depthErr);
        if (ok)
            ok = parseFramePayload(payload, nullptr, &out.depth, "Depth1200", depthErr, nullptr);
        if (!ok) {
            emit logMessage("RGBD: Depth failed — " + depthErr);
            err = depthErr;
            out.valid = false;
            return false;
        }
    }

    out.valid = true;
    return true;
}

// ─── Protocol Probe ──────────────────────────────────────────────────────────

bool TMVisionaryClient::probeServer(QString &err)
{
    Q_UNUSED(err)
    auto probe = [&](uint16_t type, const QString &label) {
        QByteArray p; QString e;
        _netDepth.request2response({}, type, p, 3000, 20000, &e);
        emit logMessage(QString("=== type %1 (%2) -> %3 bytes ===")
                            .arg(type).arg(label).arg(p.size()));
        emit logMessage(QString("  HEX[0..16]: %1")
                            .arg(QString::fromLatin1(p.left(16).toHex(' '))));
        if (p.size() > 4) {
            quint32 jlen = qFromLittleEndian<quint32>(
                reinterpret_cast<const uchar*>(p.constData()));
            if (jlen > 0 && jlen < 10000 && p.size() >= 4 + (int)jlen) {
                emit logMessage(QString("  JSON header: %1")
                                    .arg(QString::fromUtf8(p.mid(4, jlen))));
                emit logMessage(QString("  Binary tail: %1 bytes").arg(p.size() - 4 - jlen));
            } else {
                emit logMessage(QString("  TXT: %1").arg(QString::fromUtf8(p.left(300))));
            }
        }
    };

    {
        QByteArray p; QString e;
        _netDepth.request2response({}, TMVCmd::ArUco1202, p, 3000, 8000, &e);
        emit logMessage(QString("=== 1202 ArUco (%1 B) ===").arg(p.size()));
        emit logMessage(QString::fromUtf8(p.left(300)));
    }

    probe(1200, "camera_depth CONFIRMED");
    probe(1201, "RGB? unknown");
    probe(1203, "unknown");
    probe(1204, "unknown");
    probe(1000, "Safety3D");
    probe(1001, "Safety3D-summary");

    return true;
}

// ─── ArUco ───────────────────────────────────────────────────────────────────

// ─── Device info ─────────────────────────────────────────────────────────────

bool TMVisionaryClient::requestDeviceInfo(QString &infoJson, QString &err)
{
    QByteArray payload;
    if (!_netDepth.request2response({}, TMVCmd::Safety3D, payload, 3000, 5000, &err)) {
        emit logMessage("Safety3D raw: " + QString::fromUtf8(payload));
        return false;
    }
    infoJson = QString::fromUtf8(payload);
    return true;
}