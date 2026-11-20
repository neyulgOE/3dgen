#include "api1202parser.h"
#include <QDebug>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace {
static bool parseUv(const QJsonValue &value, int &u, int &v)
{
    if (!value.isArray()) return false;
    const QJsonArray a = value.toArray();
    if (a.size() != 2 || !a[0].isDouble() || !a[1].isDouble()) return false;
    u = a[0].toInt();
    v = a[1].toInt();
    return true;
}
}

bool Api1202Parser::parse(const QByteArray &jsonBytes, MarkerFrame1202 &outFrame, QString &err)
{
    outFrame = {};
    err.clear();

    QJsonParseError parseErr;
    const QJsonDocument doc = QJsonDocument::fromJson(jsonBytes, &parseErr);
    if (parseErr.error != QJsonParseError::NoError) {
        err = QString("JSON parse error: %1 at offset %2").arg(parseErr.errorString()).arg(parseErr.offset);
        return false;
    }
    if (!doc.isObject()) {
        err = "Payload is not a JSON object";
        return false;
    }

    const QJsonObject root = doc.object();
    outFrame.type = root.value("type").toString();
    outFrame.pub_nsec = root.value("pub_nsec").toVariant().toULongLong();
    outFrame.data_nsec = root.value("data_nsec").toVariant().toULongLong();
    outFrame.width = root.value("width").toInt();
    outFrame.height = root.value("height").toInt();
    outFrame.found = root.value("found").toBool();
    outFrame.marker_count = root.value("marker_count").toInt();

    if (!root.contains("markers") || !root.value("markers").isArray()) {
        if (root.contains("error")) {
            err = root.value("error").toString();
        } else {
            err = "Missing markers array";
        }
        return false;
    }

    const QJsonArray markers = root.value("markers").toArray();
    outFrame.markers.reserve(markers.size());
    for (const QJsonValue &mv : markers) {
        if (!mv.isObject()) {
            continue;
        }
        const QJsonObject m = mv.toObject();

        // ── RAW DEBUG — in nguyên JSON object của marker này TRƯỚC khi đọc
        // bất kỳ field nào. Mục đích: xác nhận chính xác server có gửi
        // has_rpy/has_normal/has_approach_theta/roll/pitch/yaw/... hay
        // không, và đúng tên field hay không — tránh suy đoán qua giá trị
        // default sau khi .toBool(false)/.toDouble(0.0) đã "che" mất sự
        // thật. Xóa dòng này sau khi debug xong.
        qDebug().noquote() << "[RAW MARKER]"
                           << QString::fromUtf8(QJsonDocument(m).toJson(QJsonDocument::Compact));

        MarkerDetection d;
        d.id = m.value("id").toInt(-1);
        d.has_xyz = m.value("has_xyz").toBool(false);

        parseUv(m.value("center"), d.center_u, d.center_v);
        parseUv(m.value("c0"), d.c0_u, d.c0_v);
        parseUv(m.value("c1"), d.c1_u, d.c1_v);
        parseUv(m.value("c2"), d.c2_u, d.c2_v);
        parseUv(m.value("c3"), d.c3_u, d.c3_v);

        const QJsonValue xyzVal = m.value("xyz");
        if (d.has_xyz && xyzVal.isArray()) {
            const QJsonArray xyz = xyzVal.toArray();
            if (xyz.size() >= 3) {
                d.x = xyz[0].toDouble();
                d.y = xyz[1].toDouble();
                d.z = xyz[2].toDouble();
            }
        }

        // ── Per-marker pose/orientation (ArucoMarker proto fields 20-36) ──
        // Server chỉ gửi các field này khi giá trị khác default (proto3 JSON
        // bỏ field == default), nên has_rpy/has_normal/has_approach_theta có
        // thể tuyệt đối không xuất hiện trong payload với marker xa/nghiêng —
        // đó không phải lỗi parse, .toBool(false)/.toDouble(0.0) phản ánh
        // đúng "server không tính được giá trị này".
        d.has_rpy = m.value("has_rpy").toBool(false);
        if (d.has_rpy) {
            d.roll  = m.value("roll").toDouble(0.0);
            d.pitch = m.value("pitch").toDouble(0.0);
            d.yaw   = m.value("yaw").toDouble(0.0);
        }

        d.rx = m.value("rx").toDouble(0.0);
        d.ry = m.value("ry").toDouble(0.0);
        d.rz = m.value("rz").toDouble(0.0);

        d.has_pnp_t = m.value("has_pnp_t").toBool(false);
        if (d.has_pnp_t) {
            d.tx = m.value("tx").toDouble(0.0);
            d.ty = m.value("ty").toDouble(0.0);
            d.tz = m.value("tz").toDouble(0.0);
        }

        d.has_normal = m.value("has_normal").toBool(false);
        if (d.has_normal) {
            d.nx = m.value("nx").toDouble(0.0);
            d.ny = m.value("ny").toDouble(0.0);
            d.nz = m.value("nz").toDouble(0.0);
        }

        d.has_approach_theta = m.value("has_approach_theta").toBool(false);
        if (d.has_approach_theta) {
            d.approach_theta = m.value("approach_theta").toDouble(0.0);
        }

        outFrame.markers.push_back(d);
    }

    // ── Orientation cấp FRAME (không phải per-marker) ───────────────────
    // Khớp với Message_ArucoDetections trong proto: has_yaw/yaw_deg tính từ
    // centre-line 2 marker; has_roll_pitch/roll_deg/pitch_deg tính từ plane
    // normal trung bình. Web tính sẵn, app chỉ đọc lại — KHÔNG tự tính.
    outFrame.has_yaw  = root.value("has_yaw").toBool(false);
    outFrame.yaw_deg  = root.value("yaw_deg").toDouble(0.0);
    outFrame.has_roll_pitch = root.value("has_roll_pitch").toBool(false);
    outFrame.roll_deg  = root.value("roll_deg").toDouble(0.0);
    outFrame.pitch_deg = root.value("pitch_deg").toDouble(0.0);

    return true;
}