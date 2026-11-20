#ifndef ORBITALTRAJECTORY_H
#define ORBITALTRAJECTORY_H

#include "types.h"
#include <QVector>
#include <QVector3D>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
// OrbitalTrajectory — hai chế độ sinh waypoint:
//
//  1. Spherical (generate)        — quỹ đạo cầu phẳng xOz, camera nhìn về T
//  2. Random Window (generateRandom) — 6-DOF random trong cửa sổ ± per DOF
// ─────────────────────────────────────────────────────────────────────────────

struct OrbitalParams {
    double radius       = 300.0;
    double halfPitchDeg = 45.0;
    double halfYawDeg   = 45.0;
    double halfRollDeg  =  0.0;
    double yawStepDeg   = 15.0;
    double pitchStepDeg = 15.0;
    int    nFrames      = 5;
    int    settleMs     = 400;
};

// ─────────────────────────────────────────────────────────────────────────────
// RandomWindowParams
// Mỗi DOF sample uniform trong [-window, +window].
// step = độ phân giải lưới (quantise); 0 → continuous float.
// seed = 0 → non-deterministic (time-based).
// ─────────────────────────────────────────────────────────────────────────────
struct RandomWindowParams {
    int    nPoses    = 20;

    // Translation window ±mm  (lưới bước 10 mm)
    double winX = 100.0;
    double winY = 100.0;
    double winZ = 100.0;
    int    xyzStep = 10;   // mm

    // Rotation window ±deg  (lưới bước 5°)
    double winRoll  = 45.0;
    double winPitch = 45.0;
    double winYaw   = 45.0;
    int    rotStep  = 5;   // deg

    int    seed     = 0;   // 0 = time-based

    int    nFrames  = 5;
    int    settleMs = 400;
};

struct OrbitalWaypoint {
    int    row   = 0;
    int    col   = 0;
    double pitchDeg    = 0.0;
    double yawDeg      = 0.0;

    double camX = 0.0, camY = 0.0, camZ = 0.0;
    double rollDeg     = 0.0;
    double pitchCamDeg = 0.0;
    double yawCamDeg   = 0.0;

    double dx = 0.0, dy = 0.0, dz = 0.0;
    double droll = 0.0, dpitch = 0.0, dyaw = 0.0;

    ScanWaypoint toScanWaypoint() const {
        ScanWaypoint wp;
        wp.dx = dx; wp.dy = dy; wp.dz = dz;
        wp.droll = droll; wp.dpitch = dpitch; wp.dyaw = dyaw;
        return wp;
    }
};

class OrbitalTrajectory
{
public:
    static QVector<OrbitalWaypoint> generate      (const OrbitalParams      &p);
    static QVector<OrbitalWaypoint> generateRandom(const RandomWindowParams &p);
    static ScanWaypoint             deltaOffset   (const OrbitalWaypoint    &from,
                                    const OrbitalWaypoint    &to);
private:
    static void lookAt(const QVector3D &camPos, const QVector3D &target,
                       double rollDeg,
                       double &outRoll, double &outPitch, double &outYaw);
};

#endif // ORBITALTRAJECTORY_H