#include "voxelmap.h"
#include <algorithm>
#include <cmath>
#include <QVector3D>
#include <QVector4D>
#include <QMatrix4x4>

// ─── addCloud ─────────────────────────────────────────────────────────────────

PointCloud VoxelMap::addCloud(const PointCloud &cloud)
{
    // Delta: voxels that just reached MIN_HITS for the first time this call
    std::vector<float>   dXYZ;
    std::vector<uint8_t> dRGB;
    int                  dCount = 0;

    const float *xyz = cloud.xyz.data();
    const uint8_t *rgb = cloud.rgb.data();

    for (int i = 0; i < cloud.count; ++i) {
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];

        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            continue;

        Key key {
            (int32_t)std::floor(x / _voxelSize),
            (int32_t)std::floor(y / _voxelSize),
            (int32_t)std::floor(z / _voxelSize)
        };

        auto it = _map.find(key);
        if (it == _map.end()) {
            // New voxel — initialise
            Cell c;
            c.avgX = x; c.avgY = y; c.avgZ = z;
            c.avgR = rgb[i * 3 + 0];
            c.avgG = rgb[i * 3 + 1];
            c.avgB = rgb[i * 3 + 2];
            c.hits = 1;
            _map[key] = c;

            if (MIN_HITS <= 1) {
                // Immediately visible
                dXYZ.push_back(x); dXYZ.push_back(y); dXYZ.push_back(z);
                dRGB.push_back(rgb[i * 3 + 0]);
                dRGB.push_back(rgb[i * 3 + 1]);
                dRGB.push_back(rgb[i * 3 + 2]);
                ++dCount;
                ++_renderedCount;
            }
        } else {
            Cell &c = it->second;
            uint32_t prevHits = c.hits;

            // Color + geometry consistency gate (skip ghost/edge noise after warm-up)
            if (c.hits >= WARM_HITS) {
                float dr = (float)rgb[i * 3 + 0] - c.avgR;
                float dg = (float)rgb[i * 3 + 1] - c.avgG;
                float db = (float)rgb[i * 3 + 2] - c.avgB;
                if (dr*dr + dg*dg + db*db > COLOR_GATE2)
                    continue;

                // Geometry gate: reject điểm bị transform lệch > sqrt(GEOM_GATE2) mm
                // so với EMA center của voxel. Tránh EMA trượt khỏi surface thực.
                float ex = x - c.avgX, ey = y - c.avgY, ez = z - c.avgZ;
                if (ex*ex + ey*ey + ez*ez > GEOM_GATE2)
                    continue;
            }

            // EMA update
            c.avgX = c.avgX + EMA_ALPHA * (x - c.avgX);
            c.avgY = c.avgY + EMA_ALPHA * (y - c.avgY);
            c.avgZ = c.avgZ + EMA_ALPHA * (z - c.avgZ);
            c.avgR = c.avgR + EMA_ALPHA * ((float)rgb[i * 3 + 0] - c.avgR);
            c.avgG = c.avgG + EMA_ALPHA * ((float)rgb[i * 3 + 1] - c.avgG);
            c.avgB = c.avgB + EMA_ALPHA * ((float)rgb[i * 3 + 2] - c.avgB);
            c.hits = prevHits + 1;

            // Just crossed the MIN_HITS threshold → enter delta
            if (prevHits < (uint32_t)MIN_HITS && c.hits >= (uint32_t)MIN_HITS) {
                dXYZ.push_back(c.avgX); dXYZ.push_back(c.avgY); dXYZ.push_back(c.avgZ);
                dRGB.push_back((uint8_t)c.avgR);
                dRGB.push_back((uint8_t)c.avgG);
                dRGB.push_back((uint8_t)c.avgB);
                ++dCount;
                ++_renderedCount;
            }
        }
    }

    evictIfNeeded();

    PointCloud delta;
    delta.count = dCount;
    delta.xyz   = std::move(dXYZ);
    delta.rgb   = std::move(dRGB);
    return delta;
}

// ─── toPointCloud ─────────────────────────────────────────────────────────────

PointCloud VoxelMap::toPointCloud() const
{
    PointCloud pc;
    pc.xyz.reserve(_map.size() * 3);
    pc.rgb.reserve(_map.size() * 3);

    for (const auto &kv : _map) {
        const Cell &c = kv.second;
        if (c.hits < (uint32_t)MIN_HITS) continue;

        pc.xyz.push_back(c.avgX);
        pc.xyz.push_back(c.avgY);
        pc.xyz.push_back(c.avgZ);
        pc.rgb.push_back((uint8_t)std::max(0.f, std::min(255.f, c.avgR)));
        pc.rgb.push_back((uint8_t)std::max(0.f, std::min(255.f, c.avgG)));
        pc.rgb.push_back((uint8_t)std::max(0.f, std::min(255.f, c.avgB)));
        ++pc.count;
    }

    return pc;
}

// ─── clear ────────────────────────────────────────────────────────────────────

void VoxelMap::clear()
{
    _map.clear();
    _renderedCount = 0;
    _dirtyFull     = false;
}

// ─── evictIfNeeded ────────────────────────────────────────────────────────────
// LFU eviction: khi vượt MAX_VOXELS, xóa 10% voxel có hits thấp nhất.

void VoxelMap::evictIfNeeded()
{
    if ((int)_map.size() <= MAX_VOXELS) return;

    // Collect hit counts
    std::vector<uint32_t> hits;
    hits.reserve(_map.size());
    for (const auto &kv : _map)
        hits.push_back(kv.second.hits);

    // Find threshold: keep top 90%
    size_t cutIdx = hits.size() / 10;          // evict bottom 10%
    std::nth_element(hits.begin(), hits.begin() + cutIdx, hits.end());
    uint32_t threshold = hits[cutIdx];

    int removed = 0;
    int removedVisible = 0;
    for (auto it = _map.begin(); it != _map.end(); ) {
        if (it->second.hits <= threshold) {
            if (it->second.hits >= (uint32_t)MIN_HITS) {
                ++removedVisible;
            }
            it = _map.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }

    _renderedCount = std::max(0, _renderedCount - removedVisible);
    _dirtyFull = true;   // Renderer needs full rebuild after eviction
}

// ─── computeICP ───────────────────────────────────────────────────────────────

bool VoxelMap::computeICP(const PointCloud &cloud, QMatrix4x4 &transform, int maxIterations) const
{
    if (_map.empty() || cloud.count == 0) return false;

    // Sub-sample to speed up (target ~2000 points)
    int stride = std::max(1, cloud.count / 2000);
    const float *xyz = cloud.xyz.data();

    for (int iter = 0; iter < maxIterations; ++iter) {
        struct MatchInfo {
            QVector3D p, q;
            float dist2;
            bool operator<(const MatchInfo &o) const { return dist2 < o.dist2; }
        };
        std::vector<MatchInfo> matches;
        matches.reserve(cloud.count / stride);

        for (int i = 0; i < cloud.count; i += stride) {
            QVector4D p(xyz[i*3], xyz[i*3+1], xyz[i*3+2], 1.0f);
            QVector4D p_t = transform * p;

            int ix = (int32_t)std::floor(p_t.x() / _voxelSize);
            int iy = (int32_t)std::floor(p_t.y() / _voxelSize);
            int iz = (int32_t)std::floor(p_t.z() / _voxelSize);

            QVector3D p_trans(p_t.x(), p_t.y(), p_t.z());
            QVector3D best_q;
            float best_dist2 = 1e9f;
            bool found = false;

            // Search 3x3x3 neighborhood (handles up to 1 voxel size drift perfectly)
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        Key nkey{ix + dx, iy + dy, iz + dz};
                        auto it = _map.find(nkey);
                        if (it != _map.end() && it->second.hits >= MIN_HITS) {
                            QVector3D q(it->second.avgX, it->second.avgY, it->second.avgZ);
                            float d2 = (p_trans - q).lengthSquared();
                            if (d2 < best_dist2) {
                                best_dist2 = d2;
                                best_q = q;
                                found = true;
                            }
                        }
                    }
                }
            }

            // Reject matches that are too far (e.g. > 20mm distance)
            if (found && best_dist2 < 400.0f) { 
                matches.push_back({p_trans, best_q, best_dist2});
            }
        }

        if (matches.size() < 10) break;

        // Trimmed ICP: Keep only the best 80% matches to reject outliers
        std::sort(matches.begin(), matches.end());
        int keep = std::max(10, (int)(matches.size() * 0.8f));
        if (keep > (int)matches.size()) keep = matches.size();

        double sum_p[3] = {0, 0, 0};
        double sum_q[3] = {0, 0, 0};

        for (int i = 0; i < keep; ++i) {
            sum_p[0] += matches[i].p.x();
            sum_p[1] += matches[i].p.y();
            sum_p[2] += matches[i].p.z();
            sum_q[0] += matches[i].q.x();
            sum_q[1] += matches[i].q.y();
            sum_q[2] += matches[i].q.z();
        }
        int num_matches = keep;

        double cp[3] = {sum_p[0]/num_matches, sum_p[1]/num_matches, sum_p[2]/num_matches};
        double cq[3] = {sum_q[0]/num_matches, sum_q[1]/num_matches, sum_q[2]/num_matches};

        double H[3][3] = {0};
        for (int i = 0; i < keep; ++i) {
            double px = matches[i].p.x() - cp[0], py = matches[i].p.y() - cp[1], pz = matches[i].p.z() - cp[2];
            double qx = matches[i].q.x() - cq[0], qy = matches[i].q.y() - cq[1], qz = matches[i].q.z() - cq[2];
            H[0][0] += px*qx; H[0][1] += px*qy; H[0][2] += px*qz;
            H[1][0] += py*qx; H[1][1] += py*qy; H[1][2] += py*qz;
            H[2][0] += pz*qx; H[2][1] += pz*qy; H[2][2] += pz*qz;
        }

        // Horn's method: 4x4 symmetric matrix N
        double N[4][4];
        N[0][0] =  H[0][0] + H[1][1] + H[2][2];
        N[0][1] =  H[1][2] - H[2][1];
        N[0][2] =  H[2][0] - H[0][2];
        N[0][3] =  H[0][1] - H[1][0];

        N[1][0] = N[0][1];
        N[1][1] =  H[0][0] - H[1][1] - H[2][2];
        N[1][2] =  H[0][1] + H[1][0];
        N[1][3] =  H[2][0] + H[0][2];

        N[2][0] = N[0][2];
        N[2][1] = N[1][2];
        N[2][2] = -H[0][0] + H[1][1] - H[2][2];
        N[2][3] =  H[1][2] + H[2][1];

        N[3][0] = N[0][3];
        N[3][1] = N[1][3];
        N[3][2] = N[2][3];
        N[3][3] = -H[0][0] - H[1][1] + H[2][2];

        // Power iteration with shift to guarantee positive definiteness
        double trace_abs = 0;
        for(int r=0; r<4; ++r) for(int c=0; c<4; ++c) trace_abs += std::abs(N[r][c]);
        double shift = trace_abs + 1.0; 
        for (int i=0; i<4; ++i) N[i][i] += shift;

        double v[4] = {1.0, 0.0, 0.0, 0.0};
        for (int p = 0; p < 20; ++p) {
            double vn[4] = {0,0,0,0};
            for(int r=0; r<4; ++r)
                for(int c=0; c<4; ++c)
                    vn[r] += N[r][c] * v[c];
            double norm = std::sqrt(vn[0]*vn[0] + vn[1]*vn[1] + vn[2]*vn[2] + vn[3]*vn[3]);
            if (norm > 1e-12) {
                v[0] = vn[0]/norm; v[1] = vn[1]/norm; v[2] = vn[2]/norm; v[3] = vn[3]/norm;
            }
        }

        double qw = v[0], qx = v[1], qy = v[2], qz = v[3];
        QMatrix4x4 delta;
        delta.setToIdentity();
        
        delta(0,0) = 1.0 - 2.0*qy*qy - 2.0*qz*qz;
        delta(0,1) = 2.0*qx*qy - 2.0*qz*qw;
        delta(0,2) = 2.0*qx*qz + 2.0*qy*qw;
        
        delta(1,0) = 2.0*qx*qy + 2.0*qz*qw;
        delta(1,1) = 1.0 - 2.0*qx*qx - 2.0*qz*qz;
        delta(1,2) = 2.0*qy*qz - 2.0*qx*qw;
        
        delta(2,0) = 2.0*qx*qz - 2.0*qy*qw;
        delta(2,1) = 2.0*qy*qz + 2.0*qx*qw;
        delta(2,2) = 1.0 - 2.0*qx*qx - 2.0*qy*qy;

        double tx = cq[0] - (delta(0,0)*cp[0] + delta(0,1)*cp[1] + delta(0,2)*cp[2]);
        double ty = cq[1] - (delta(1,0)*cp[0] + delta(1,1)*cp[1] + delta(1,2)*cp[2]);
        double tz = cq[2] - (delta(2,0)*cp[0] + delta(2,1)*cp[1] + delta(2,2)*cp[2]);

        delta(0,3) = tx;
        delta(1,3) = ty;
        delta(2,3) = tz;

        transform = delta * transform;
    }

    return true;
}