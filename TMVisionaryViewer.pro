QT += core gui widgets network charts opengl

CONFIG += c++17
TEMPLATE = app
TARGET = TMVisionaryViewer

# --- Intel RealSense SDK 2.0 ---
win32 {
    INCLUDEPATH += "$$quote(C:/Users/ADMIN/OneDrive/Documents/RealSense SDK 2.0/include)"
    LIBS += -L"$$quote(C:/Users/ADMIN/OneDrive/Documents/RealSense SDK 2.0/lib/x64)" -lrealsense2
}
unix {
    LIBS += -lrealsense2
    CONFIG += link_pkgconfig
    PKGCONFIG += opencv4
}

# --- OpenCV (Cho VIO Tracking - Cần cài đặt) ---
# INCLUDEPATH += C:/Users/ADMIN/opencv/include
# LIBS += -LC:/Users/ADMIN/opencv/x64/vc17/lib \
#         -lopencv_core480 -lopencv_rgbd480 -lopencv_calib3d480 -lopencv_imgproc480


SOURCES += \
    src/api1202parser.cpp \
    src/appcontroller.cpp \
    src/cameraworker.cpp \
    src/cloudaccumulator.cpp \
    src/depthprojector.cpp \
    src/coordinatetransformer.cpp \
    src/exporter.cpp \
    src/fastsaminference.cpp \
    src/keyframeextractor.cpp \
    src/main.cpp \
    src/mainwindow.cpp \
    src/netprotocol.cpp \
    src/orbitalscandialog.cpp \
    src/orbitaltrajectory.cpp \
    src/plyexporter.cpp \
    src/pointcloudwidget.cpp \
    src/realsenseworker.cpp \
    src/replayworker.cpp \
    src/scanengine.cpp \
    src/scansession.cpp \
    src/settingsdialog.cpp \
    src/tmvisionaryclient.cpp \
    src/meshviewerwidget.cpp \
    src/vio_tracker.cpp \
    src/voxelmap.cpp

HEADERS += \
    src/api1202parser.h \
    src/appcontroller.h \
    src/cameraworker.h \
    src/cloudaccumulator.h \
    src/depthprojector.h \
    src/coordinatetransformer.h \
    src/exporter.h \
    src/fastsaminference.h \
    src/icamerasource.h \
    src/ifusionalgorithm.h \
    src/keyframeextractor.h \
    src/mainwindow.h \
    src/netprotocol.h \
    src/orbitalscandialog.h \
    src/orbitaltrajectory.h \
    src/plyexporter.h \
    src/pointcloudwidget.h \
    src/realsenseworker.h \
    src/replayworker.h \
    src/scanengine.h \
    src/scansession.h \
    src/settingsdialog.h \
    src/tmvisionaryclient.h \
    src/meshviewerwidget.h \
    src/types.h \
    src/object3d.h \
    src/vio_tracker.h \
    src/voxelmap.h


win32:   LIBS += -lopengl32
unix:    LIBS += -lGL

win32:CONFIG(release, debug|release): DESTDIR_WIN = $$OUT_PWD/release
win32:CONFIG(debug, debug|release): DESTDIR_WIN = $$OUT_PWD/debug