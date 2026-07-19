"""Biomech-layer verification — the ASMI phase model / double pendulum, extracted from the shaft IMU and
tried against the FSX ground truth. Gates are instrument correctness (features well-formed on all 30 real
shots); the article's CLAIMS (casting bleeds speed; summation of segmental velocities; sequence quality
shows up at impact) are tested and reported as findings, not gated. Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.swing import bake_session, FEATURES
from cocycle.launch import parse_fsx_pdf, join_session, loo_r2, screen_corr

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"
PDF = os.path.join(SESSION, "FSX_Shotresults.pdf")
BIO = ("release_onset", "lag_retention", "late_speed_ratio", "transition_dwell_s",
       "backswing_smooth", "follow_decel_dps2")


def main():
    ids_x, stamps, fnames, X = bake_session(SESSION)
    ids_y, lnames, Y = parse_fsx_pdf(PDF)
    ids, Xj, Yj = join_session(ids_x, X, ids_y, Y)

    def xcol(n): return Xj[:, fnames.index(n)]
    def ycol(n): return Yj[:, lnames.index(n)]

    b1 = (Xj.shape == (30, len(FEATURES)) and np.isfinite(Xj).all())
    ro, lr, ls = xcol("release_onset"), xcol("lag_retention"), xcol("late_speed_ratio")
    b2 = bool(np.all((ro >= 0) & (ro <= 1)) and np.all(lr >= 0) and np.all(ls > 0))
    b3 = all(xcol(n).std() > 0 for n in BIO)                     # discriminative, not constant

    targets = ("club_speed_mph", "ball_speed_mph", "carry_yds", "efficiency",
               "attack_angle_deg", "dynamic_loft_deg", "face_to_path_deg")
    results = {}
    for tg in targets:
        y = ycol(tg)
        corr = screen_corr(Xj, fnames, y)
        top = [nm for nm, _ in corr[:3]]
        r2, n = loo_r2(Xj[:, [fnames.index(nm) for nm in top]], y)
        results[tg] = (corr, top, r2, n)
    b4 = all(np.isfinite(r2) for _, _, r2, _ in results.values())

    def r_of(feat, tg):
        y = ycol(tg); x = xcol(feat)
        ok = np.isfinite(x) & np.isfinite(y)
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    print(f"[BIOMECH] matrix: {Xj.shape[0]} shots x {Xj.shape[1]} features ({len(BIO)} biomech) vs {len(targets)} FSX targets")
    print(f"[BIOMECH] release_onset {ro.mean():.2f} +/- {ro.std():.2f} of downswing | lag_retention "
          f"{lr.mean():.2f} +/- {lr.std():.2f} | late/early speed {ls.mean():.2f} +/- {ls.std():.2f}")
    print("\n[BIOMECH] the article's claims vs the data (Pearson r, n=30):")
    print(f"    'casting bleeds speed':    release_onset->club_speed r={r_of('release_onset','club_speed_mph'):+.2f}   "
          f"lag_retention->club_speed r={r_of('lag_retention','club_speed_mph'):+.2f}")
    print(f"    'summation of velocities': late_speed_ratio->club_speed r={r_of('late_speed_ratio','club_speed_mph'):+.2f}   "
          f"->efficiency r={r_of('late_speed_ratio','efficiency'):+.2f}")
    print(f"    'sequence shows at impact': release_onset->efficiency r={r_of('release_onset','efficiency'):+.2f}   "
          f"transition_dwell->efficiency r={r_of('transition_dwell_s','efficiency'):+.2f}")
    print("\n[BIOMECH] full screen (top-3 correlates + LOO-CV R^2; * = biomech feature):")
    for tg, (corr, top, r2, n) in results.items():
        cs = ", ".join(f"{'*' if nm in BIO else ''}{nm} {r:+.2f}" for nm, r in corr[:3])
        print(f"    {tg:18s} LOO R^2 = {r2:+.2f}   {cs}")

    checks = {
        "B1 extended matrix well-formed: 30 x 24, all finite": b1,
        "B2 biomech features in valid ranges (onset in [0,1], ratios > 0)": b2,
        "B3 biomech features discriminative across shots": b3,
        "B4 screen computes on all 7 FSX targets": b4,
    }
    print("\n[BIOMECH] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[BIOMECH] {'ALL CRITERIA MET — phase-model features extracted and tried against ground truth' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
