#pragma once

#include <QObject>
#include <QString>
#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

struct RecorderIntrinsics {
    int width = 640;
    int height = 480;
    double fx = 615.0;
    double fy = 615.0;
    double cx = 320.0;
    double cy = 240.0;
    double depthScale = 1000.0; // 1000.0 -> 1 unit = 1mm
};

class DatasetRecorder : public QObject {
    Q_OBJECT
public:
    explicit DatasetRecorder(QObject* parent = nullptr);
    ~DatasetRecorder() override;

    // Tạo thư mục rgb/, depth_raw/, ghi ngay intrinsics.json và mở luồng ghi đĩa
    bool start(const QString& sessionDir,
               const RecorderIntrinsics& intrinsics,
               int minIntervalMs = 33,
               int maxQueueSize = 60);

    // Đẩy frame từ CameraWorker vào hàng đợi (non-blocking, an toàn luồng)
    // isRgb = true nếu color là RGB (RealSense RS2_FORMAT_RGB8), false nếu đã là BGR
    void enqueueFrame(const cv::Mat& color,
                      const cv::Mat& depthU16,
                      const cv::Mat& pose4x4 = cv::Mat(),
                      bool isRgb = true);

    // Chờ xả hết hàng đợi, ghi poses.json và trả về số frame thực tế đã lưu
    int stop();

    bool isRecording() const { return _recording.load(std::memory_order_acquire); }
    int savedCount() const { return _savedCount.load(std::memory_order_acquire); }
    QString currentDir() const { return _sessionDir; }

signals:
    void sig_log(const QString& msg);
    void sig_recordStats(int savedFrames, double fps, float centerDepthMm);
    void sig_recordFinished(const QString& sessionDir, int totalFrames);
    void sig_recordError(const QString& errorMsg);

private:
    struct FramePacket {
        int index = 0;
        cv::Mat color;
        cv::Mat depthU16;
        cv::Mat pose4x4;
        bool isRgb = true;
    };

    void writerLoop();
    bool writeIntrinsicsJson(const QString& dir, const RecorderIntrinsics& intr);
    bool writePosesJson(const QString& dir);
    static float computeCenterMedianDepthMm(const cv::Mat& depthU16, double depthScale);

    QString _sessionDir;
    RecorderIntrinsics _intrinsics;
    int _minIntervalMs = 33;
    int _maxQueueSize = 60;

    std::atomic<bool> _recording{false};
    std::atomic<bool> _stopRequested{false};
    std::atomic<int> _savedCount{0};
    int _nextFrameIndex = 0;
    int _droppedCount = 0;

    std::chrono::steady_clock::time_point _startTime;
    std::chrono::steady_clock::time_point _lastEnqueueTime;

    std::mutex _queueMtx;
    std::condition_variable _queueCv;
    std::deque<FramePacket> _queue;
    std::thread _writerThread;

    std::mutex _posesMtx;
    std::vector<std::pair<int, cv::Mat>> _recordedPoses;
};
