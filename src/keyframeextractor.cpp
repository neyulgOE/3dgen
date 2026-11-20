#include "keyframeextractor.h"
#include <cmath>

KeyframeExtractor::KeyframeExtractor(QObject *parent)
    : QObject(parent)
{
    reset();
}

void KeyframeExtractor::reset()
{
    _candidateFrames.clear();
    _sharpnessScores.clear();
    _lastKeptPose.setToIdentity();
    _hasFirstPose = false;
}

double KeyframeExtractor::calculateSharpness(const RGBFrame &rgbFrame)
{
    if (rgbFrame.data.isEmpty()) return 0.0;
    
    cv::Mat colorMat(rgbFrame.height, rgbFrame.width, CV_8UC3, (void*)rgbFrame.data.constData());
    cv::Mat gray;
    cv::cvtColor(colorMat, gray, cv::COLOR_RGB2GRAY);
    
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    
    double variance = sigma.val[0] * sigma.val[0];
    return variance;
}

void KeyframeExtractor::addFrame(const TrackingFrame &frame)
{
    if (!_hasFirstPose) {
        _lastKeptPose = frame.pose;
        _hasFirstPose = true;
        _candidateFrames.push_back(frame);
        _sharpnessScores.push_back(calculateSharpness(frame.rgbd.rgb));
        return;
    }
    
    // Calculate distance from _lastKeptPose
    float dx = frame.pose(0, 3) - _lastKeptPose(0, 3);
    float dy = frame.pose(1, 3) - _lastKeptPose(1, 3);
    float dz = frame.pose(2, 3) - _lastKeptPose(2, 3);
    float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
    
    // Calculate rotation angle roughly (using trace of rotation matrix)
    float trace = frame.pose(0,0) * _lastKeptPose(0,0) + frame.pose(0,1) * _lastKeptPose(0,1) + frame.pose(0,2) * _lastKeptPose(0,2) +
                  frame.pose(1,0) * _lastKeptPose(1,0) + frame.pose(1,1) * _lastKeptPose(1,1) + frame.pose(1,2) * _lastKeptPose(1,2) +
                  frame.pose(2,0) * _lastKeptPose(2,0) + frame.pose(2,1) * _lastKeptPose(2,1) + frame.pose(2,2) * _lastKeptPose(2,2);
    // trace = 1 + 2*cos(theta)
    float cosTheta = (trace - 1.0f) / 2.0f;
    cosTheta = std::clamp(cosTheta, -1.0f, 1.0f);
    float angleRad = std::acos(cosTheta);
    float angleDeg = angleRad * 180.0f / M_PI;
    
    // Keep frame if distance > 0.05m (5cm) or angle > 10 degrees
    if (distance > 0.05f || angleDeg > 10.0f) {
        _lastKeptPose = frame.pose;
        _candidateFrames.push_back(frame);
        _sharpnessScores.push_back(calculateSharpness(frame.rgbd.rgb));
        // emit logMessage(QString("Candidate added. Total: %1").arg(_candidateFrames.size()));
    }
}

QVector<TrackingFrame> KeyframeExtractor::extractKeyframes(int k)
{
    QVector<TrackingFrame> result;
    
    if (_candidateFrames.empty()) {
        return result;
    }
    
    if (_candidateFrames.size() <= static_cast<size_t>(k)) {
        for (const auto &f : _candidateFrames) {
            result.append(f);
        }
        return result;
    }
    
    // Extract positions (translation vectors) for KMeans
    cv::Mat positions(_candidateFrames.size(), 3, CV_32F);
    for (size_t i = 0; i < _candidateFrames.size(); ++i) {
        positions.at<float>(i, 0) = _candidateFrames[i].pose(0, 3);
        positions.at<float>(i, 1) = _candidateFrames[i].pose(1, 3);
        positions.at<float>(i, 2) = _candidateFrames[i].pose(2, 3);
    }
    
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(positions, k, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
               
    // For each cluster, find the frame with the highest sharpness
    std::vector<int> bestIndices(k, -1);
    std::vector<double> bestScores(k, -1.0);
    
    for (size_t i = 0; i < _candidateFrames.size(); ++i) {
        int clusterIdx = labels.at<int>(i, 0);
        double score = _sharpnessScores[i];
        
        if (score > bestScores[clusterIdx]) {
            bestScores[clusterIdx] = score;
            bestIndices[clusterIdx] = i;
        }
    }
    
    for (int i = 0; i < k; ++i) {
        if (bestIndices[i] != -1) {
            result.append(_candidateFrames[bestIndices[i]]);
        }
    }
    
    return result;
}
