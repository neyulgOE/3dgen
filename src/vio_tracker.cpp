#include "vio_tracker.h"
#include <QDebug>
#include <QThread>

VioTracker::VioTracker(QObject *parent)
    : QObject(parent)
{
    _currentPose.setToIdentity();
    _currentRt = cv::Mat::eye(4, 4, CV_64F);

    CameraIntrinsics intrinsics;
    _cameraMatrix = (cv::Mat_<double>(3, 3) << intrinsics.fx, 0, intrinsics.cx,
                                               0, intrinsics.fy, intrinsics.cy,
                                               0, 0, 1);

    _odometry = cv::rgbd::Odometry::create("RgbdOdometry");
    if (_odometry) {
        _odometry->setCameraMatrix(_cameraMatrix);
    }
}

VioTracker::~VioTracker()
{
}

void VioTracker::startTracking()
{
    if (_isTracking) return;
    
    _isTracking = true;
    _currentPose.setToIdentity();
    _currentRt = cv::Mat::eye(4, 4, CV_64F);
    _hasPrevFrame = false;
    
    emit logMessage("VIO Tracking started. (OpenCV RgbdOdometry)");
}

void VioTracker::stopTracking()
{
    _isTracking = false;
    emit logMessage("VIO Tracking stopped.");
}

void VioTracker::onImuDataReady(const QVector<IMUData> &imuData)
{
    if (!_isTracking) return;
    
    std::lock_guard<std::mutex> lock(_imuMutex);
    for (const IMUData &data : imuData) {
        if (data.type == "gyro") {
            if (_lastImuTime > 0 && data.timestamp_ns > _lastImuTime) {
                double dt = (data.timestamp_ns - _lastImuTime) / 1e9;
                if (dt > 0 && dt < 0.1) {
                    _accumulatedGyro[0] += data.x * dt;
                    _accumulatedGyro[1] += data.y * dt;
                    _accumulatedGyro[2] += data.z * dt;
                }
            }
            _lastImuTime = data.timestamp_ns;
        }
    }
}

void VioTracker::onRgbdFrameReady(const RGBDFrame &frame)
{
    if (!_isTracking) return;
    
    if (frame.rgb.data.isEmpty() || frame.depth.z_mm_data.isEmpty()) {
        return;
    }

    // Convert RGB to Gray cv::Mat
    cv::Mat colorMat(frame.rgb.height, frame.rgb.width, CV_8UC3, (void*)frame.rgb.data.constData());
    cv::Mat gray;
    if (!colorMat.empty()) {
        cv::cvtColor(colorMat, gray, cv::COLOR_RGB2GRAY);
    }

    // Convert Depth (float mm) to meters
    cv::Mat depthRaw(frame.depth.height, frame.depth.width, CV_32FC1, (void*)frame.depth.z_mm_data.constData());
    cv::Mat depth;
    depthRaw.convertTo(depth, CV_32FC1, 0.001); // mm to meters

    if (!_hasPrevFrame) {
        _prevGray = gray.clone();
        _prevDepth = depth.clone();
        _hasPrevFrame = true;
        return;
    }

    if (_odometry && !_prevGray.empty() && !_prevDepth.empty() && !gray.empty() && !depth.empty()) {
        cv::Mat Rt;
        
        cv::Vec3d gyroCopy;
        {
            std::lock_guard<std::mutex> lock(_imuMutex);
            gyroCopy = _accumulatedGyro;
            _accumulatedGyro = cv::Vec3d(0, 0, 0); // Reset for next frame
        }
        
        // OpenCV initRt: Transform from prev to current.
        // Gyro gives camera rotation. Point transform is inverse of camera rotation.
        // So R_points = R_cam^T. And R_cam is Rodrigues(gyro). R_cam^T = Rodrigues(-gyro).
        cv::Mat R_init;
        cv::Rodrigues(-gyroCopy, R_init);
        cv::Mat initRt = cv::Mat::eye(4, 4, CV_64F);
        R_init.copyTo(initRt(cv::Rect(0, 0, 3, 3)));

        bool isOk = _odometry->compute(_prevGray, _prevDepth, cv::Mat(), gray, depth, cv::Mat(), Rt, initRt);
        
        if (isOk && !Rt.empty()) {
            Rt.convertTo(Rt, CV_64F);
            _currentRt = _currentRt * Rt.inv(); // accumulate pose

            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    _currentPose(r, c) = _currentRt.at<double>(r, c);
                }
            }
        }
    }

    _prevGray = gray.clone();
    _prevDepth = depth.clone();
    
    TrackingFrame tkFrame;
    tkFrame.rgbd = frame;
    tkFrame.pose = _currentPose;
    tkFrame.timestamp_ns = frame.depth.timestamp_ns;
    tkFrame.valid = true;
    
    emit trackingFrameReady(tkFrame);
}
