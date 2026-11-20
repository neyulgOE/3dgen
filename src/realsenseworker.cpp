#include "realsenseworker.h"
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMutexLocker>
#include <QCoreApplication>
RealSenseWorker::RealSenseWorker(QObject *parent)
    : ICameraSource(parent), _align_to(RS2_STREAM_COLOR)
{
}

RealSenseWorker::~RealSenseWorker()
{
    disconnectHost();
}

bool RealSenseWorker::isConnected() const
{
    return _connected;
}

void RealSenseWorker::init()
{
    try {
        _cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 30);
        _cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    } catch (const rs2::error & e) {
        emit errorOccurred(QString("RealSense Init Error: %1").arg(e.what()));
    }
}

void RealSenseWorker::connectHost(const QString &ip, quint16 port)
{
    Q_UNUSED(ip);
    Q_UNUSED(port);

    if (_connected) return;

    try {
        rs2::pipeline_profile profile = _pipeline.start(_cfg);
        _connected = true;

        _depthScaleMm = 1.0f;
        try {
            auto sensor = profile.get_device().first<rs2::depth_sensor>();
            _depthScaleMm = sensor.get_depth_scale() * 1000.0f;
        } catch (...) {}

        try {
            auto colorStream = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
            rs2_intrinsics colorIntr = colorStream.get_intrinsics();
            emit intrinsicsReady(colorIntr.width, colorIntr.height,
                                 colorIntr.fx, colorIntr.fy,
                                 colorIntr.ppx, colorIntr.ppy);
            emit logMessage(QString("RealSense Color-Aligned Intrinsics: %1x%2 fx=%3 fy=%4 cx=%5 cy=%6")
                                .arg(colorIntr.width).arg(colorIntr.height)
                                .arg(colorIntr.fx, 0, 'f', 2).arg(colorIntr.fy, 0, 'f', 2)
                                .arg(colorIntr.ppx, 0, 'f', 2).arg(colorIntr.ppy, 0, 'f', 2));
        } catch (const rs2::error &intrErr) {
            emit logMessage(QString("RealSense Intrinsics Warning: %1").arg(intrErr.what()));
        }

        emit connectionChanged(true);
        emit logMessage("RealSense connected successfully.");
    } catch (const rs2::error & e) {
        emit errorOccurred(QString("RealSense Connect Error: %1").arg(e.what()));
    }
}

void RealSenseWorker::disconnectHost()
{
    if (!_connected) return;

    _fetching = false;
    try {
        _pipeline.stop();
    } catch (...) {}
    
    _connected = false;
    emit connectionChanged(false);
    emit logMessage("RealSense disconnected.");
}

void RealSenseWorker::fetchCount(int n, int intervalMs)
{
    if (!_connected) {
        emit errorOccurred("Cannot fetch, RealSense not connected.");
        return;
    }

    if (_fetching) return;
    _fetching = true;
    _remainingFetches = n;
    _fetchIntervalMs = intervalMs;

    {
        QMutexLocker locker(&_imuMutex);
        _imuBuffer.clear();
    }

    QMetaObject::invokeMethod(this, "processNextFrame", Qt::QueuedConnection);
}

void RealSenseWorker::fetchContinuous(int intervalMs)
{
    if (!_connected) {
        emit errorOccurred("Cannot fetch, RealSense not connected.");
        return;
    }

    if (_fetching) return;
    _fetching = true;
    _remainingFetches = -1; // -1 indicates continuous
    _fetchIntervalMs = intervalMs;

    {
        QMutexLocker locker(&_imuMutex);
        _imuBuffer.clear();
    }

    QMetaObject::invokeMethod(this, "processNextFrame", Qt::QueuedConnection);
}

void RealSenseWorker::processNextFrame()
{
    if (!_fetching || (_remainingFetches <= 0 && _remainingFetches != -1) || !_connected) {
        _fetching = false;
        emit fetchCountFinished();
        return;
    }

    const float depth_scale_mm = _depthScaleMm;

    try {
        rs2::frameset frames;
        bool gotAny = false;
        
        while (_pipeline.poll_for_frames(&frames)) {
            gotAny = true;
            
            // Extract IMU
            for (auto f : frames) {
                if (auto mf = f.as<rs2::motion_frame>()) {
                    IMUData imu;
                    imu.timestamp_ns = static_cast<quint64>(mf.get_timestamp() * 1e6);
                    if (mf.get_profile().stream_type() == RS2_STREAM_ACCEL) imu.type = "accel";
                    else if (mf.get_profile().stream_type() == RS2_STREAM_GYRO) imu.type = "gyro";
                    auto motion_data = mf.get_motion_data();
                    imu.x = motion_data.x; imu.y = motion_data.y; imu.z = motion_data.z;
                    QMutexLocker locker(&_imuMutex);
                    _imuBuffer.append(imu);
                }
            }
            
            // Extract Color & Depth
            rs2::video_frame color_frame = frames.get_color_frame();
            rs2::depth_frame depth_frame = frames.get_depth_frame();
            
            if (color_frame && depth_frame) {
                // We have a synced RGBD frame!
                rs2::frameset aligned_frames = _align_to.process(frames);
                color_frame = aligned_frames.get_color_frame();
                depth_frame = aligned_frames.get_depth_frame();
                
                if (color_frame && depth_frame) {
                    QVector<IMUData> imuFrames;
                    {
                        QMutexLocker locker(&_imuMutex);
                        imuFrames = _imuBuffer;
                        _imuBuffer.clear();
                    }
                    if (!imuFrames.isEmpty()) {
                        emit imuDataReady(imuFrames);
                    }
                    
                    RGBDFrame outFrame;
                    
                    // Populate RGB
                    outFrame.rgb.width = color_frame.get_width();
                    outFrame.rgb.height = color_frame.get_height();
                    outFrame.rgb.data = QByteArray(reinterpret_cast<const char*>(color_frame.get_data()), 
                                                   outFrame.rgb.width * outFrame.rgb.height * 3);
                    outFrame.rgb.timestamp_ns = color_frame.get_timestamp() * 1e6; // ms to ns
                    outFrame.rgb.valid = true;

                    // Populate Depth
                    outFrame.depth.width = depth_frame.get_width();
                    outFrame.depth.height = depth_frame.get_height();
                    int num_pixels = outFrame.depth.width * outFrame.depth.height;
                    outFrame.depth.z_mm_data.resize(num_pixels);

                    const uint16_t* depth_data = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
                    
                    for (int i = 0; i < num_pixels; ++i) {
                        outFrame.depth.z_mm_data[i] = static_cast<float>(depth_data[i]) * depth_scale_mm;
                    }
                    
                    outFrame.depth.timestamp_ns = depth_frame.get_timestamp() * 1e6;
                    outFrame.depth.valid = true;
                    outFrame.valid = true;

                    emit rgbdFrameReady(outFrame);
                    if (_remainingFetches > 0) {
                        _remainingFetches--;
                    }
                }
            }
        }
        
        if (!gotAny) {
            // No frames ready yet. Re-queue if fetching (15ms avoids 200Hz spin).
            if (_fetching && (_remainingFetches > 0 || _remainingFetches == -1)) {
                QTimer::singleShot(15, this, &RealSenseWorker::processNextFrame);
            }
            return;
        }
    } catch (const rs2::error & e) {
        if (!_fetching) {
            emit fetchCountFinished();
            return;
        }
        emit errorOccurred(QString("RealSense Fetch Error: %1").arg(e.what()));
        _fetching = false;
        emit fetchCountFinished();
        return;
    }

    if (_fetching && (_remainingFetches > 0 || _remainingFetches == -1)) {
        const int nextMs = (_fetchIntervalMs > 0) ? _fetchIntervalMs : 15;
        QTimer::singleShot(nextMs, this, &RealSenseWorker::processNextFrame);
    } else {
        _fetching = false;
        emit fetchCountFinished();
    }
}

void RealSenseWorker::stopFetch()
{
    _fetching = false;
}
