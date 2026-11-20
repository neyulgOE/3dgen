#ifndef VIO_TRACKER_H
#define VIO_TRACKER_H

#include <QObject>
#include <QMatrix4x4>
#include <QVector>
#include <QThread>
#include "types.h"

#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/rgbd.hpp>

class VioTracker : public QObject
{
    Q_OBJECT
public:
    explicit VioTracker(QObject *parent = nullptr);
    ~VioTracker();

public slots:
    void startTracking();
    void stopTracking();
    QMatrix4x4 getCurrentPose() const { return _currentPose; }
    
    // Nhận dữ liệu từ RealSenseWorker
    void onRgbdFrameReady(const RGBDFrame &frame);
    void onImuDataReady(const QVector<IMUData> &imuData);

signals:
    // Gửi pose và frame đã track thành công ra ngoài
    void trackingFrameReady(const TrackingFrame &frame);
    void logMessage(const QString &msg);

private:
    bool _isTracking = false;
    QMatrix4x4 _currentPose;
    
    // OpenCV Odometry state
    cv::Ptr<cv::rgbd::Odometry> _odometry;
    cv::Mat _prevGray;
    cv::Mat _prevDepth;
    cv::Mat _cameraMatrix;
    cv::Mat _currentRt;
    
    bool _hasPrevFrame = false;
    
    // IMU pre-integration
    cv::Vec3d _accumulatedGyro{0.0, 0.0, 0.0};
    quint64 _lastImuTime = 0;
    std::mutex _imuMutex;
};

#endif // VIO_TRACKER_H
