#ifndef VOXELMAP_H
#define VOXELMAP_H

#include "types.h"
#include <unordered_map>

// ─── VoxelMap ─────────────────────────────────────────────────────────────────
// Mỗi voxel 15mm lưu EMA của XYZ và RGB, cùng số lần hit.
// Chỉ render voxel có hit >= MIN_HITS để lọc nhiễu 1 frame.
// Color consistency check reject điểm ghost/nhiễu viền sau warm-up.
// Khi vượt MAX_VOXELS, xóa các voxel hit thấp nhất (LFU eviction).

class VoxelMap {
public:
    static constexpr int   MIN_HITS   = 1;
    static constexpr int   MAX_VOXELS = 1000000;

    static constexpr uint32_t WARM_HITS   = 5;
    static constexpr float    EMA_ALPHA   = 0.15f;
    static constexpr float    COLOR_GATE2 = 40000.f;
    // Geometry gate: reject điểm cách EMA center > sqrt(GEOM_GATE2) mm sau warm-up.
    // Mặc định 15mm = 1 voxel_size — loại bỏ ghost từ transform lệch giữa pose.
    static constexpr float    GEOM_GATE2  = 225.f;  // 15mm²

    struct Cell {
        float    avgX = 0, avgY = 0, avgZ = 0;
        float    avgR = 0, avgG = 0, avgB = 0;
        uint32_t hits = 0;
    };

    struct Key {
        int32_t ix, iy, iz;
        bool operator==(const Key &o) const {
            return ix == o.ix && iy == o.iy && iz == o.iz;
        }
    };
    struct KeyHash {
        size_t operator()(const Key &k) const {
            size_t h = (size_t)(uint32_t)k.ix;
            h ^= (size_t)(uint32_t)k.iy * 2654435761u;
            h ^= (size_t)(uint32_t)k.iz * 2246822519u;
            return h;
        }
    };

    using Map = std::unordered_map<Key, Cell, KeyHash>;

    // Voxel size — chỉ có tác dụng sau clear()
    float voxelSize() const { return _voxelSize; }
    void  setVoxelSize(float mm) { _voxelSize = (mm > 0.5f ? mm : 0.5f); }

    // Thêm cloud mới. Trả về delta — chỉ những voxel vừa đạt MIN_HITS lần đầu
    PointCloud addCloud(const PointCloud &cloud);

    void clear();
    int  size()     const { return (int)_map.size(); }
    int  rendered() const { return _renderedCount; }

    PointCloud toPointCloud() const;

    // ICP Alignment (Short-Term & Working Memory matching)
    // Refines the 'transform' matrix by matching the incoming cloud against the existing VoxelMap.
    bool computeICP(const PointCloud &cloud, QMatrix4x4 &transform, int maxIterations = 5) const;

    bool needsFullRebuild() const { return _dirtyFull; }
    void clearRebuildFlag()       { _dirtyFull = false; }

private:
    void evictIfNeeded();

    Map   _map;
    int   _renderedCount = 0;
    bool  _dirtyFull     = false;
    float _voxelSize     = 15.f;
};

#endif // VOXELMAP_H