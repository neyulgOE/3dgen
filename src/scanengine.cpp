#include "scanengine.h"
#include "cloudaccumulator.h" // Plugin mặc định hiện tại

ScanEngine::ScanEngine(QObject *parent) : QObject(parent)
{
    // Cắm plugin CloudAccumulator vào hệ thống (tắt mặc định, chỉ bật khi người dùng tích Accumulate)
    _algorithm = new CloudAccumulator(this);
    _algorithm->setEnabled(false);

    // Gắn các signal từ lõi thuật toán ra interface mới
    connect(_algorithm, &IFusionAlgorithm::deltaReady,
            this, &ScanEngine::onAccumulatorDeltaReady);
            
    connect(_algorithm, &IFusionAlgorithm::fullRebuildReady,
            this, &ScanEngine::onAccumulatorFullRebuild);
            
    connect(_algorithm, &IFusionAlgorithm::statsUpdated,
            this, &ScanEngine::onAccumulatorStatsUpdated);
}

void ScanEngine::addFrame(const PointCloud &cloud, const QMatrix4x4 &pose)
{
    // Đẩy vào algorithm
    if (_algorithm) _algorithm->addCloud(cloud, pose);
}

void ScanEngine::clear()
{
    if (_algorithm) _algorithm->clear();
    _currentObject = Object3D();
    emit objectUpdated(_currentObject);
}

Object3D ScanEngine::getObject3D() const
{
    return _currentObject;
}

void ScanEngine::setEnabled(bool on)
{
    if (_algorithm) _algorithm->setEnabled(on);
}

void ScanEngine::setVoxelSize(float mm)
{
    if (_algorithm) _algorithm->setVoxelSize(mm);
}

void ScanEngine::setScanMask(bool on, float cx, float cy, float cz, float r)
{
    if (_algorithm) _algorithm->setScanMask(on, cx, cy, cz, r);
}

void ScanEngine::setIcpEnabled(bool on)
{
    if (_algorithm) _algorithm->setIcpEnabled(on);
}

void ScanEngine::loadRawCloud(const std::vector<float> &xyz, const std::vector<uint8_t> &rgb, int count)
{
    if (_algorithm) _algorithm->loadRawCloud(xyz, rgb, count);
}

// ─── Private Slots ────────────────────────────────────────────────────────────

void ScanEngine::onAccumulatorDeltaReady(const PointCloud &delta)
{
    // Delta (incremental) -> không thay đổi toàn bộ Object3D nhưng
    // cần update renderer ngay lập tức.
    emit partialUpdateReady(delta);
    
    // Tạm thời, vì CloudAccumulator không có hàm trả về full point cloud
    // một cách dễ dàng ngoài lúc rebuild, ta có thể lưu tạm hoặc yêu cầu 
    // full rebuild nếu cần gọi getObject3D().
    // Ở bản nâng cấp, ta sẽ giữ PointCloud trong _currentObject.
}

void ScanEngine::onAccumulatorFullRebuild(const PointCloud &full)
{
    _currentObject.pointCloud = full;
    emit objectUpdated(_currentObject);
}

void ScanEngine::onAccumulatorStatsUpdated(int voxelCount, int pointCount)
{
    emit statsUpdated(voxelCount, pointCount);
}
