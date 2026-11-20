#ifndef OBJECT3D_H
#define OBJECT3D_H

#include "types.h"

// ─────────────────────────────────────────────────────────────────────────────
// Object3D — Khối dữ liệu chuẩn hóa đầu ra của ScanEngine.
// 
// Tách biệt so với PointCloud thô. Object3D có thể chứa PointCloud đã được 
// làm sạch, Mesh, Normal, và các Metadata khác (ví dụ: Timestamp, Transform cuối).
// Phù hợp với kiến trúc SDK: Camera -> ScanEngine -> Object3D -> Renderer/Exporter.
// ─────────────────────────────────────────────────────────────────────────────

struct Object3D {
    PointCloud pointCloud;
    
    // Future expansion:
    // std::vector<float> normals;
    // std::vector<int> meshTriangles;
    // QJsonObject metadata;
    
    bool isEmpty() const {
        return pointCloud.count == 0;
    }
};

#endif // OBJECT3D_H
