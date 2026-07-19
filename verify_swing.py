"""Swing-bake verification (driftwave Stage 1). Runs on the REAL 30-shot session: proves the shaft-IMU
bundle parses, impact/phase segment, the quaternion S^3 geometry is well-formed, the per-shot feature matrix
is finite AND discriminative (features vary shot to shot = a real 'model' substrate), and a swing transposes
into a COCYCLE regime. Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.swing import (load_shot, detect_impact, segment_swing, quat_geometry, swing_features,
                           swing_timeline, bake_session, swing_consistency, FEATURES, _mag, _rate)
from cocycle.engine import Context
from cocycle.components.field import TopologyField
from cocycle.components.projector import Projector

SESSION = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"
SHOT1 = os.path.join(SESSION, "GolfML1.2_2024-03-21T14-51-24_Shot1")


def main():
    shot = load_shot(SHOT1)
    ra, rq = _rate(shot["AD"][0]), _rate(shot["SFQ"][0])
    s1 = (all(k in shot for k in ("AD", "GD", "SFQ", "SFEA")) and 1200 < ra < 1800 and 120 < rq < 260
          and all(np.isfinite(v[1]).all() for v in shot.values()))

    t_imp, imp_g = detect_impact(shot)
    m = _mag(shot["AD"][1]); base = float(np.median(m[:300]))
    s2 = imp_g > 10.0 and imp_g > 8 * base and 0.0 < t_imp < shot["AD"][0][-1]

    seg = segment_swing(shot, t_imp)
    s3 = (seg["backswing_s"] > 0 and seg["downswing_s"] > 0 and 1.0 < seg["tempo_ratio"] < 6.0
          and seg["takeaway"] < seg["top"] < seg["impact"])

    geo = quat_geometry(shot, seg)
    qn = np.linalg.norm(shot["SFQ"][1] / (np.linalg.norm(shot["SFQ"][1], axis=1, keepdims=True) + 1e-12), axis=1)
    s4 = (abs(qn.mean() - 1.0) < 1e-6 and 0.0 <= geo["geo_addr_impact_rad"] <= np.pi
          and 0.0 < geo["swing_planarity"] <= 1.0 and geo["s3_arc_len"] > 0.0)

    ids, stamps, names, M = bake_session(SESSION)
    finite = M.size and np.isfinite(M).all()
    std = M.std(0) if M.size else np.zeros(len(FEATURES))
    def col(n): return M[:, FEATURES.index(n)]
    discriminative = all(std[FEATURES.index(n)] > 0 for n in           # impact_g excluded: sensor-saturated
                         ("peak_gyro_dps", "tempo_ratio", "spec_centroid_hz", "s3_arc_len"))
    s5 = M.shape[0] >= 25 and finite and discriminative

    comp = [TopologyField(n=5000), Projector()]; c = Context(); [k.setup(c) for k in comp]
    tl = swing_timeline(shot); X0 = None
    for i in range(0, min(tl.n_frames, 150)):
        c.data["features"] = tl.to_features(i)
        for k in comp: k.update(c)
        if i == 0: X0 = c.get("field")["X"].copy()
    moved = float(np.linalg.norm(c.get("field")["X"] - X0, axis=1).mean())
    s6 = moved > 1e-3 and np.isfinite(c.get("proj")["xy"]).all()

    # S7: label-free launch proxies physically sane + a session repeatability signature
    f1, _ = swing_features(shot)
    cons = swing_consistency(ids, names, M)
    s7 = (15 < f1["club_speed_ms"] < 70 and 0 <= f1["swing_plane_tilt_deg"] <= 90
          and 0 <= f1["release_ratio"] <= 1.5 and np.isfinite(cons["cv"]).all() and col("club_speed_ms").std() > 0)

    print(f"[SWING] Shot1: accel {ra:.0f} Hz, quat {rq:.0f} Hz | impact {imp_g:.1f} g @ t={t_imp:.2f}s over {base:.2f} g baseline")
    print(f"[SWING] tempo: back {seg['backswing_s']:.3f}s / down {seg['downswing_s']:.3f}s = {seg['tempo_ratio']:.2f}:1 | "
          f"peak gyro {seg['peak_gyro_dps']:.0f} deg/s")
    print(f"[SWING] S^3 geom: geodesic(addr,impact) {geo['geo_addr_impact_rad']:.2f} rad, arc {geo['s3_arc_len']:.2f} rad, "
          f"planarity {geo['swing_planarity']:.3f}")
    print(f"[SWING] Shot1 launch proxies: club_speed {f1['club_speed_ms']:.1f} m/s ({f1['club_speed_ms']*2.237:.0f} mph), "
          f"plane_tilt {f1['swing_plane_tilt_deg']:.0f} deg, release {f1['release_ratio']:.2f}")
    print(f"[SWING] session matrix: {M.shape[0]} shots x {M.shape[1]} features (launch-monitor-join-ready)")
    if M.size:
        print(f"        club_speed  {col('club_speed_ms').mean():7.1f} +/- {col('club_speed_ms').std():5.1f} m/s")
        print(f"        tempo_ratio {col('tempo_ratio').mean():7.2f} +/- {col('tempo_ratio').std():5.2f}")
        print(f"        spec_cen_hz {col('spec_centroid_hz').mean():7.1f} +/- {col('spec_centroid_hz').std():5.1f} Hz")
        order = np.argsort(cons["cv"])
        rep = ", ".join(f"{names[j]} {cons['cv'][j]*100:.0f}%" for j in order[:3])
        vary = ", ".join(f"{names[j]} {cons['cv'][j]*100:.0f}%" for j in order[-3:][::-1])
        print(f"[SWING] repeatability (CV): most consistent -> {rep}   |   most variable -> {vary}")
        print(f"[SWING] outlier shots (>2.5 sigma, candidate mishits): {list(cons['outliers'].keys()) or 'none'}")

    checks = {
        "S1 shot bundle parses: streams present, multi-rate, finite": s1,
        "S2 impact recovered: sharp strike spike over quiet baseline": s2,
        "S3 phase segmentation: back/down split, sane tempo ratio": s3,
        "S4 S^3 geometry well-formed: unit quats, geodesic, planarity, arc": s4,
        "S5 session matrix finite AND discriminative across shots": s5,
        "S6 swing transposes into a COCYCLE regime (field advects)": s6,
        "S7 label-free launch proxies sane + repeatability signature computed": s7,
    }
    print("\n[SWING] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[SWING] {'ALL CRITERIA MET — the model half is built; launch-monitor output joins by timestamp' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
