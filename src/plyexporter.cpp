#include "plyexporter.h"

#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <algorithm>

bool PlyExporter::exportToFile(const QString              &path,
                               const std::vector<float>   &xyz,
                               const std::vector<uint8_t> &rgb,
                               int                         count,
                               QString                    &errMsg)
{
    if (count <= 0) {
        errMsg = "No points to export.";
        return false;
    }
    if ((int)xyz.size() < count * 3 || (int)rgb.size() < count * 3) {
        errMsg = "Data inconsistency: buffer size mismatch.";
        return false;
    }

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly)) {
        errMsg = "Cannot open file for writing:\n" + path;
        return false;
    }

    // ── PLY header ─────────────────────────────────────────────────────────
    QTextStream hdr(&f);
    hdr << "ply\nformat binary_little_endian 1.0\n"
        << "comment TMVisionary Camera Viewer\n"
        << "comment coordinate_system: waypoint_0_camera_frame\n"
        << "comment units: millimeters\n"
        << "element vertex " << count << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";
    hdr.flush();

    // ── Binary data — batched để tránh nhiều system call nhỏ ──────────────
#pragma pack(push, 1)
    struct Vertex { float x, y, z; uint8_t r, g, b; };
#pragma pack(pop)
    const int batch = 8192;
    std::vector<Vertex> buf;
    buf.reserve(batch);
    int written = 0;
    while (written < count) {
        buf.clear();
        const int take = std::min(batch, count - written);
        for (int i = 0; i < take; ++i) {
            const int vi = written + i;
            buf.push_back({xyz[vi*3], xyz[vi*3+1], xyz[vi*3+2],
                           rgb[vi*3], rgb[vi*3+1], rgb[vi*3+2]});
        }
        f.write(reinterpret_cast<const char*>(buf.data()),
                qint64(buf.size() * sizeof(Vertex)));
        written += take;
    }
    f.close();

    if (f.error() != QFile::NoError) {
        errMsg = "Write error: " + f.errorString();
        return false;
    }
    return true;
}