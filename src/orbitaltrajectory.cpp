#include "orbitaltrajectory.h"
#include <QtMath>
#include <algorithm>
#include <random>

// ─────────────────────────────────────────────────────────────────────────────
// lookAt — ZYX Euler extraction
// ─────────────────────────────────────────────────────────────────────────────
void OrbitalTrajectory::lookAt(const QVector3D &camPos,
                               const QVector3D &target,
                               double rollDeg,
                               double &outRoll,
                               double &outPitch,
                               double &outYaw)
{
    QVector3D fwd = (target - camPos).normalized();

    QVector3D worldUp(0.f, -1.f, 0.f);
    if (std::abs(QVector3D::dotProduct(fwd, worldUp)) > 0.999f)
        worldUp = QVector3D(1.f, 0.f, 0.f);

    QVector3D right = QVector3D::crossProduct(fwd, worldUp).normalized();

    float rollRad = float(rollDeg * M_PI / 180.0);
    QVector3D rightR = right * std::cos(rollRad) +
                       QVector3D::crossProduct(fwd, right) * std::sin(rollRad);

    QVector3D up = QVector3D::crossProduct(rightR, fwd).normalized();

    float ry = std::asin(qBound(-1.f, float(-fwd.x()), 1.f));
    float rx, rz;
    if (std::abs(std::cos(ry)) > 1e-6f) {
        rx = std::atan2(float(fwd.y()), float(fwd.z()));
        rz = std::atan2(float(up.x()),  float(rightR.x()));
    } else {
        rx = std::atan2(float(-up.z()), float(up.y()));
        rz = 0.f;
    }

    const double rad2deg = 180.0 / M_PI;
    outRoll  = rx * rad2deg;
    outPitch = ry * rad2deg;
    outYaw   = rz * rad2deg;
}

// ─────────────────────────────────────────────────────────────────────────────
// generate — spherical orbit (unchanged)
// ─────────────────────────────────────────────────────────────────────────────
QVector<OrbitalWaypoint> OrbitalTrajectory::generate(const OrbitalParams &p)
{
    QVector<OrbitalWaypoint> result;

    const int nSteps = (p.halfYawDeg < 1e-9 || p.yawStepDeg < 1e-9)
                           ? 0
                           : qMax(1, qRound(p.halfYawDeg / p.yawStepDeg));

    // Waypoint 0: Home
    {
        OrbitalWaypoint wp0;
        wp0.row = 0; wp0.col = 0;
        result.append(wp0);
    }

    if (nSteps == 0)
        return result;

    const QVector3D target(0.f, 0.f, float(p.radius));

    QVector<double> yawAngles;
    for (int i = 1; i <= nSteps; ++i) yawAngles.append( i * p.yawStepDeg);
    for (int i = 1; i <= nSteps; ++i) yawAngles.append(-i * p.yawStepDeg);

    int colIdx = 1;
    for (double aziDeg : std::as_const(yawAngles)) {
        const double aziRad = aziDeg * M_PI / 180.0;

        const double camX = -p.radius * std::sin(aziRad);
        const double camY =  0.0;
        const double camZ =  p.radius * (1.0 - std::cos(aziRad));

        OrbitalWaypoint wp;
        wp.row = 0; wp.col = colIdx++;
        wp.pitchDeg    = 0.0;
        wp.yawDeg      = aziDeg;
        wp.camX = camX; wp.camY = camY; wp.camZ = camZ;
        wp.rollDeg     = p.halfRollDeg;
        wp.pitchCamDeg = 0.0;
        wp.yawCamDeg   = aziDeg;

        // Gán đúng trục vật lý dựa trên calibration:
        // dy = camX, -dz = camY, -dx = camZ
        wp.dx = -camZ; wp.dy = camX; wp.dz = -camY;
        wp.droll = p.halfRollDeg; wp.dpitch = 0.0; wp.dyaw = aziDeg;

        result.append(wp);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// generateRandom — 6-DOF random trong cửa sổ ± (grid-quantised)
//
// Logic lấy từ rebuildOffsetPlan của project khác:
//  • Lưới xyz bước xyzStep mm, góc bước rotStep độ
//  • Loại điểm rỗng (cả 6 thành phần = 0)
//  • Loại trùng lặp bằng key set
//  • Seed = 0 → time-based; khác → deterministic
// ─────────────────────────────────────────────────────────────────────────────
QVector<OrbitalWaypoint> OrbitalTrajectory::generateRandom(const RandomWindowParams &p)
{
    QVector<OrbitalWaypoint> result;

    // ── Waypoint 0: Home ──────────────────────────────────────────────────────
    {
        OrbitalWaypoint wp0;
        wp0.row = 0; wp0.col = 0;
        result.append(wp0);
    }

    if (p.nPoses <= 0)
        return result;

    // ── RNG setup ─────────────────────────────────────────────────────────────
    std::mt19937 rng;
    if (p.seed == 0)
        rng.seed(std::random_device{}());
    else
        rng.seed(static_cast<uint32_t>(p.seed));

    // ── Lưới cho từng DOF ─────────────────────────────────────────────────────
    // Translation: [-winX, +winX] bước xyzStep
    const int xStep  = qMax(1, p.xyzStep);
    const int rStep  = qMax(1, p.rotStep);

    const int bX = static_cast<int>(p.winX);
    const int bY = static_cast<int>(p.winY);
    const int bZ = static_cast<int>(p.winZ);
    const int bR = static_cast<int>(p.winRoll);
    const int bPi = static_cast<int>(p.winPitch);
    const int bW = static_cast<int>(p.winYaw);

    // Tổng số tổ hợp có thể — dùng để cap nPoses
    const long long nX  = bX  > 0 ? (2*bX /xStep + 1) : 1;
    const long long nY  = bY  > 0 ? (2*bY /xStep + 1) : 1;
    const long long nZ  = bZ  > 0 ? (2*bZ /xStep + 1) : 1;
    const long long nR  = bR  > 0 ? (2*bR /rStep  + 1) : 1;
    const long long nP  = bPi > 0 ? (2*bPi/rStep  + 1) : 1;
    const long long nW  = bW  > 0 ? (2*bW /rStep  + 1) : 1;
    const long long maxComb = nX * nY * nZ * nR * nP * nW - 1; // -1 loại điểm 0

    int count = p.nPoses;
    if (maxComb > 0 && count > (int)maxComb)
        count = (int)maxComb;

    // ── Sampling ──────────────────────────────────────────────────────────────
    auto sampleAxis = [&](int bound, int step) -> int {
        if (bound <= 0 || step <= 0) return 0;
        int nSlots = 2 * bound / step + 1;
        int slot   = std::uniform_int_distribution<int>(0, nSlots - 1)(rng);
        return (slot - bound / step) * step;
    };

    QSet<QString> seen;
    int col = 1;
    int tries = 0;
    const int maxTries = count * 200;

    while (col <= count && tries < maxTries) {
        ++tries;

        const int dx     = sampleAxis(bX,  xStep);
        const int dy     = sampleAxis(bY,  xStep);
        const int dz     = sampleAxis(bZ,  xStep);
        const int droll  = sampleAxis(bR,  rStep);
        const int dpitch = sampleAxis(bPi, rStep);
        const int dyaw   = sampleAxis(bW,  rStep);

        // Loại điểm rỗng hoàn toàn
        if (dx == 0 && dy == 0 && dz == 0 &&
            droll == 0 && dpitch == 0 && dyaw == 0)
            continue;

        // Loại trùng lặp
        const QString key = QString("%1_%2_%3_%4_%5_%6")
                                .arg(dx).arg(dy).arg(dz)
                                .arg(droll).arg(dpitch).arg(dyaw);
        if (seen.contains(key))
            continue;
        seen.insert(key);

        OrbitalWaypoint wp;
        wp.row = 0;
        wp.col = col++;

        wp.dx     = dx;
        wp.dy     = dy;
        wp.dz     = dz;
        wp.droll  = droll;
        wp.dpitch = dpitch;
        wp.dyaw   = dyaw;

        // Mirror into display fields
        wp.camX = dx; wp.camY = dy; wp.camZ = dz;
        wp.pitchDeg    = dpitch;
        wp.yawDeg      = dyaw;
        wp.rollDeg     = droll;
        wp.pitchCamDeg = dpitch;
        wp.yawCamDeg   = dyaw;

        result.append(wp);
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
ScanWaypoint OrbitalTrajectory::deltaOffset(const OrbitalWaypoint &from,
                                            const OrbitalWaypoint &to)
{
    ScanWaypoint delta;
    
    // TM Robot applies translation then rotation in the Tool frame.
    // To reach absolute 'to' from absolute 'from', the translation delta
    // must be rotated back by the 'from' rotation, otherwise it drifts.
    
    QMatrix4x4 R;
    R.setToIdentity();
    // Assuming ZYX Euler order for TM Robot
    R.rotate(from.dyaw, 0, 0, 1);
    R.rotate(from.dpitch, 0, 1, 0);
    R.rotate(from.droll, 1, 0, 0);

    QVector3D diff(to.dx - from.dx, to.dy - from.dy, to.dz - from.dz);
    QVector3D diff_rel = R.inverted().mapVector(diff);

    delta.dx     = diff_rel.x();
    delta.dy     = diff_rel.y();
    delta.dz     = diff_rel.z();
    
    delta.droll  = to.droll  - from.droll;
    delta.dpitch = to.dpitch - from.dpitch;
    delta.dyaw   = to.dyaw   - from.dyaw;
    return delta;
}