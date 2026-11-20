#ifndef FASTSAMINFERENCE_H
#define FASTSAMINFERENCE_H

#include <QObject>
#include <QString>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

class FastSamInference : public QObject
{
    Q_OBJECT
public:
    explicit FastSamInference(QObject *parent = nullptr);
    ~FastSamInference();

    bool loadModel(const QString &modelPath);
    cv::Mat removeBackground(const cv::Mat &inputImage);

signals:
    void logMessage(const QString &msg);

private:
    cv::dnn::Net _net;
    bool _isModelLoaded = false;
    
    // Yolo/FastSAM constants
    const int _inputSize = 640;
    const float _confThreshold = 0.25f;
    const float _iouThreshold = 0.45f;
    
    void processOutput(const cv::Mat &inputImage, const std::vector<cv::Mat> &outputs, cv::Mat &outMask);
};

#endif // FASTSAMINFERENCE_H
