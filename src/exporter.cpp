#include "exporter.h"
#include "plyexporter.h" // Sử dụng lại lõi PlyExporter cũ

bool Exporter::exportPLY(const QString &filename, const Object3D &obj, QString &errMs)
{
    if (obj.isEmpty()) {
        errMs = "Object3D is empty";
        return false;
    }

    return PlyExporter::exportToFile(
        filename, 
        obj.pointCloud.xyz, 
        obj.pointCloud.rgb, 
        obj.pointCloud.count, 
        errMs
    );
}
