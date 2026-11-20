#include "replayworker.h"
#include <QImage>
#include <QDebug>
#include <QDateTime>

ReplayWorker::ReplayWorker(QObject *parent) : ICameraSource(parent), _timer(this)
{
    connect(&_timer, &QTimer::timeout, this, &ReplayWorker::onTimeout);
}

ReplayWorker::~ReplayWorker()
{
    disconnectHost();
}

bool ReplayWorker::isConnected() const
{
    return _connected;
}

void ReplayWorker::init()
{
}

void ReplayWorker::connectHost(const QString &dirPath, quint16 port)
{
    Q_UNUSED(port);
    _recordDir.setPath(dirPath);
    
    QDir rgbDir(dirPath + "/rgb");
    if (!rgbDir.exists()) {
        emit errorOccurred("Replay error: 'rgb' folder not found in " + dirPath);
        return;
    }
    
    _rgbFiles = rgbDir.entryList(QStringList() << "*.png" << "*.jpg", QDir::Files, QDir::Name);
    if (_rgbFiles.isEmpty()) {
        emit errorOccurred("Replay error: No images found in rgb folder.");
        return;
    }
    
    _connected = true;
    _currentIndex = 0;
    emit connectionChanged(true);
    emit logMessage(QString("ReplayWorker connected to %1. Found %2 frames.").arg(dirPath).arg(_rgbFiles.size()));
}

void ReplayWorker::disconnectHost()
{
    QMetaObject::invokeMethod(&_timer, "stop", Qt::QueuedConnection);
    _connected = false;
    _fetching = false;
    emit connectionChanged(false);
    emit logMessage("ReplayWorker disconnected.");
}

void ReplayWorker::fetchCount(int n, int intervalMs)
{
    if (!_connected) return;
    _remainingFetches = n;
    _fetching = true;
    QMetaObject::invokeMethod(&_timer, "start", Qt::QueuedConnection, Q_ARG(int, intervalMs));
}


void ReplayWorker::fetchContinuous(int intervalMs)
{
    if (!_connected) return;
    _remainingFetches = -1;
    _fetching = true;
    QMetaObject::invokeMethod(&_timer, "start", Qt::QueuedConnection, Q_ARG(int, intervalMs > 0 ? intervalMs : 33));
}

void ReplayWorker::stopFetch()
{
    _fetching = false;
    QMetaObject::invokeMethod(&_timer, "stop", Qt::QueuedConnection);
}

void ReplayWorker::onTimeout()
{
    if (!_fetching) {
        _timer.stop();
        return;
    }
    
    if (_currentIndex >= _rgbFiles.size() || (_remainingFetches == 0)) {
        _timer.stop();
        _fetching = false;
        emit fetchCountFinished();
        return;
    }
    
    QString baseName = _rgbFiles[_currentIndex]; // e.g. 000000.png
    QString rgbPath = _recordDir.filePath("rgb/" + baseName);
    QString depthPath = _recordDir.filePath("depth_raw/" + baseName);
    
    QImage rgbImg(rgbPath);
    QImage depthImg(depthPath);
    
    if (!rgbImg.isNull() && !depthImg.isNull()) {
        rgbImg = rgbImg.convertToFormat(QImage::Format_RGB888);
        
        RGBDFrame frame;
        frame.valid = true;
        
        // RGB
        frame.rgb.valid = true;
        frame.rgb.width = rgbImg.width();
        frame.rgb.height = rgbImg.height();
        frame.rgb.timestamp_ns = QDateTime::currentMSecsSinceEpoch() * 1000000ULL;
        
        int rgbBytes = rgbImg.width() * rgbImg.height() * 3;
        frame.rgb.data = QByteArray(reinterpret_cast<const char*>(rgbImg.constBits()), rgbBytes);
        
        // Depth (16-bit)
        frame.depth.valid = true;
        frame.depth.width = depthImg.width();
        frame.depth.height = depthImg.height();
        frame.depth.timestamp_ns = frame.rgb.timestamp_ns;
        
        int numPixels = depthImg.width() * depthImg.height();
        frame.depth.z_mm_data.resize(numPixels);
        
        const uint16_t* pDepth = reinterpret_cast<const uint16_t*>(depthImg.constBits());
        for (int i = 0; i < numPixels; ++i) {
            frame.depth.z_mm_data[i] = static_cast<float>(pDepth[i]);
        }
        
        emit rgbdFrameReady(frame);
    } else {
        emit logMessage("Replay warning: Failed to load frame " + baseName);
    }
    
    _currentIndex++;
    if (_remainingFetches > 0) {
        _remainingFetches--;
    }
}
