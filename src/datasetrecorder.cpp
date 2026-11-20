#include "datasetrecorder.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>

DatasetRecorder::DatasetRecorder(QObject* parent)
    : QObject(parent)
{
}

DatasetRecorder::~DatasetRecorder()
{
    if (_recording.load() || _writerThread.joinable()) {
        stop();
    }
}

bool DatasetRecorder::start(const QString& sessionDir,
                            const RecorderIntrinsics& intrinsics,
                            int minIntervalMs,
                            int maxQueueSize)
{
    if (_recording.load()) {
        emit sig_recordError("Đang ghi dữ liệu, hãy dừng phiên hiện tại trước.");
        return false;
    }

    QDir dir(sessionDir);
    if (!dir.mkpath("rgb") || !dir.mkpath("depth_raw")) {
        emit sig_recordError("Không thể tạo thư mục rgb/ và depth_raw/ tại: " + sessionDir);
        return false;
    }

    if (!writeIntrinsicsJson(sessionDir, intrinsics)) {
        emit sig_recordError("Không thể ghi intrinsics.json vào: " + sessionDir);
        return false;
    }

    _sessionDir = sessionDir;
    _intrinsics = intrinsics;
    _minIntervalMs = std::max(0, minIntervalMs);
    _maxQueueSize = std::max(8, maxQueueSize);
    _savedCount.store(0);
    _nextFrameIndex = 0;
    _droppedCount = 0;
    _stopRequested.store(false);

    {
        std::lock_guard<std::mutex> lk(_queueMtx);
        _queue.clear();
    }
    {
        std::lock_guard<std::mutex> lk(_posesMtx);
        _recordedPoses.clear();
    }

    _startTime = std::chrono::steady_clock::now();
    _lastEnqueueTime = _startTime - std::chrono::milliseconds(_minIntervalMs + 1);
    _recording.store(true, std::memory_order_release);

    _writerThread = std::thread(&DatasetRecorder::writerLoop, this);
    emit sig_log(QString("🔴 DatasetRecorder đã sẵn sàng: %1 (fx=%2, fy=%3)")
                     .arg(sessionDir)
                     .arg(intrinsics.fx, 0, 'f', 2)
                     .arg(intrinsics.fy, 0, 'f', 2));
    return true;
}

void DatasetRecorder::enqueueFrame(const cv::Mat& color,
                                   const cv::Mat& depthU16,
                                   const cv::Mat& pose4x4,
                                   bool isRgb)
{
    if (!_recording.load(std::memory_order_acquire)) {
        return;
    }
    if (color.empty() || depthU16.empty() || depthU16.type() != CV_16UC1) {
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - _lastEnqueueTime).count();
    if (_minIntervalMs > 0 && elapsedMs < _minIntervalMs) {
        return;
    }
    _lastEnqueueTime = now;

    // Tính nhanh cự ly trung vị vùng tâm ảnh để cảnh báo MinZ (<250mm) ngay trên UI
    const float centerMm = computeCenterMedianDepthMm(depthU16, _intrinsics.depthScale);

    FramePacket pkt;
    pkt.color = color.clone();
    pkt.depthU16 = depthU16.clone();
    pkt.pose4x4 = (pose4x4.rows == 4 && pose4x4.cols == 4)
                      ? pose4x4.clone()
                      : cv::Mat::eye(4, 4, CV_64F);
    pkt.isRgb = isRgb;

    {
        std::lock_guard<std::mutex> lk(_queueMtx);
        if (static_cast<int>(_queue.size()) >= _maxQueueSize) {
            ++_droppedCount;
            return;
        }
        pkt.index = _nextFrameIndex++;
        _queue.push_back(std::move(pkt));
    }
    _queueCv.notify_one();

    const double totalSec = std::max(0.05, std::chrono::duration<double>(now - _startTime).count());
    const int saved = _savedCount.load(std::memory_order_relaxed);
    emit sig_recordStats(saved, static_cast<double>(saved) / totalSec, centerMm);
}

int DatasetRecorder::stop()
{
    if (!_recording.exchange(false)) {
        if (_writerThread.joinable()) {
            _stopRequested.store(true);
            _queueCv.notify_all();
            _writerThread.join();
        }
        return _savedCount.load();
    }

    _stopRequested.store(true);
    _queueCv.notify_all();
    if (_writerThread.joinable()) {
        _writerThread.join();
    }

    const int total = _savedCount.load();
    if (total == 0) {
        emit sig_recordError(
            "❌ Phiên quay lưu được 0 frames! Kiểm tra lại kết nối RealSense hoặc luồng ảnh RGB-D.");
        return 0;
    }

    writePosesJson(_sessionDir);

    QString summary = QString("✅ Đã lưu hoàn tất %1 frames vào %2").arg(total).arg(_sessionDir);
    if (_droppedCount > 0) {
        summary += QString(" (bỏ qua %1 frames do nghẽn ổ cứng)").arg(_droppedCount);
    }
    emit sig_log(summary);
    emit sig_recordFinished(_sessionDir, total);
    return total;
}

void DatasetRecorder::writerLoop()
{
    const std::string rgbBase = (_sessionDir + "/rgb").toStdString();
    const std::string depthBase = (_sessionDir + "/depth_raw").toStdString();
    const std::vector<int> fastPngParams = {cv::IMWRITE_PNG_COMPRESSION, 1};

    while (true) {
        FramePacket pkt;
        {
            std::unique_lock<std::mutex> lk(_queueMtx);
            _queueCv.wait(lk, [this]() {
                return !_queue.empty() || _stopRequested.load();
            });
            if (_queue.empty() && _stopRequested.load()) {
                break;
            }
            pkt = std::move(_queue.front());
            _queue.pop_front();
        }

        char fname[32];
        std::snprintf(fname, sizeof(fname), "/%06d.png", pkt.index);

        cv::Mat bgrToSave;
        if (pkt.isRgb && pkt.color.channels() == 3) {
            cv::cvtColor(pkt.color, bgrToSave, cv::COLOR_RGB2BGR);
        } else {
            bgrToSave = pkt.color;
        }

        const bool okRgb = cv::imwrite(rgbBase + fname, bgrToSave, fastPngParams);
        const bool okDepth = cv::imwrite(depthBase + fname, pkt.depthU16, fastPngParams);

        if (okRgb && okDepth) {
            {
                std::lock_guard<std::mutex> lk(_posesMtx);
                _recordedPoses.emplace_back(pkt.index, pkt.pose4x4);
            }
            _savedCount.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

bool DatasetRecorder::writeIntrinsicsJson(const QString& dir, const RecorderIntrinsics& intr)
{
    QJsonObject obj;
    obj["width"] = intr.width;
    obj["height"] = intr.height;
    obj["fx"] = intr.fx;
    obj["fy"] = intr.fy;
    obj["cx"] = intr.cx;
    obj["cy"] = intr.cy;
    obj["depth_scale"] = intr.depthScale;

    QFile f(dir + "/intrinsics.json");
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return false;
    }
    f.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
    f.close();
    return true;
}

bool DatasetRecorder::writePosesJson(const QString& dir)
{
    QJsonObject root;
    std::lock_guard<std::mutex> lk(_posesMtx);
    for (const auto& kv : _recordedPoses) {
        cv::Mat M;
        kv.second.convertTo(M, CV_64F);
        QJsonArray rows;
        for (int r = 0; r < 4; ++r) {
            QJsonArray row;
            for (int c = 0; c < 4; ++c) {
                row.append(M.at<double>(r, c));
            }
            rows.append(row);
        }
        char key[32];
        std::snprintf(key, sizeof(key), "cam_%03d", kv.first);
        root[QString::fromLatin1(key)] = rows;
    }

    QFile f(dir + "/poses.json");
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        return false;
    }
    f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
    f.close();
    return true;
}

float DatasetRecorder::computeCenterMedianDepthMm(const cv::Mat& depthU16, double depthScale)
{
    const int H = depthU16.rows;
    const int W = depthU16.cols;
    const int r0 = std::max(0, H / 2 - 50);
    const int r1 = std::min(H, H / 2 + 50);
    const int c0 = std::max(0, W / 2 - 50);
    const int c1 = std::min(W, W / 2 + 50);

    std::vector<uint16_t> samples;
    samples.reserve(700);
    for (int r = r0; r < r1; r += 4) {
        const uint16_t* rowPtr = depthU16.ptr<uint16_t>(r);
        for (int c = c0; c < c1; c += 4) {
            const uint16_t v = rowPtr[c];
            if (v > 0) {
                samples.push_back(v);
            }
        }
    }
    if (samples.empty()) {
        return 0.0f;
    }
    const size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    return static_cast<float>(samples[mid]) * static_cast<float>(1000.0 / depthScale);
}
