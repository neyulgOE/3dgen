#ifndef PLYEXPORTER_H
#define PLYEXPORTER_H

#include <QString>
#include <cstdint>
#include <vector>

// ─── PlyExporter ──────────────────────────────────────────────────────────────
// Stateless utility: ghi point cloud ra file PLY binary little-endian.
// Tách hoàn toàn khỏi UI — MainWindow chỉ gọi exportToFile() và show msgbox.

class PlyExporter
{
public:
    // Trả về true nếu thành công.
    // errMsg được set khi trả về false.
    static bool exportToFile(const QString              &path,
                             const std::vector<float>   &xyz,
                             const std::vector<uint8_t> &rgb,
                             int                         count,
                             QString                    &errMsg);
};

#endif // PLYEXPORTER_H