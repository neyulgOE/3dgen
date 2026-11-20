#include "mainwindow.h"
#include <QApplication>
#include "types.h"
#include <QVector>
#include <QMatrix4x4>

int main(int argc, char *argv[])
{
    qRegisterMetaType<RGBDFrame>("RGBDFrame");
    qRegisterMetaType<QVector<IMUData>>("QVector<IMUData>");
    qRegisterMetaType<TrackingFrame>("TrackingFrame");
    qRegisterMetaType<GeometryFrame>("GeometryFrame");
    qRegisterMetaType<QVector<QMatrix4x4>>("QVector<QMatrix4x4>");

    QApplication app(argc, argv);
    app.setApplicationName("TMVisionary Camera Viewer");
    app.setOrganizationName("TMV");

    MainWindow w;
    w.show();
    return app.exec();
}