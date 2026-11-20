#ifndef KEYFRAMEEXTRACTOR_H
#define KEYFRAMEEXTRACTOR_H

#include <QObject>
#include <QVector>
#include <QMatrix4x4>
#include <vector>

#include "types.h"
#include <opencv2/opencv.hpp>

class KeyframeExtractor : public QObject
{
    Q_OBJECT
public:
    explicit KeyframeExtractor(QObject *parent = nullptr);

public slots:
    void addFrame(const TrackingFrame &frame);
    void reset();
    QVector<TrackingFrame> extractKeyframes(int k = 24);

signals:
    void logMessage(const QString &msg);

private:
    double calculateSharpness(const RGBFrame &rgbFrame);

    std::vector<TrackingFrame> _candidateFrames;
    std::vector<double> _sharpnessScores;
    QMatrix4x4 _lastKeptPose;
    bool _hasFirstPose;
};

#endif // KEYFRAMEEXTRACTOR_H
