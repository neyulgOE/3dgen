#ifndef EXPORTER_H
#define EXPORTER_H

#include <QString>
#include "object3d.h"

// ─────────────────────────────────────────────────────────────────────────────
// Exporter — Độc lập module xuất dữ liệu 3D ra file.
//
// Tránh việc UI hay ScanEngine phải tự viết code ghi ổ cứng.
// Chỉ nhận đầu vào là Object3D.
// ─────────────────────────────────────────────────────────────────────────────

class Exporter
{
public:
    static bool exportPLY(const QString &filename, const Object3D &obj, QString &errMs);
    
    // Future:
    // static bool exportOBJ(const QString &filename, const Object3D &obj, QString &errMs);
    // static bool exportSTL(const QString &filename, const Object3D &obj, QString &errMs);
};

#endif // EXPORTER_H
