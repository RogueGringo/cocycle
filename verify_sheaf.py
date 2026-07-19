"""Sheaf-layer verification: restriction maps + gluing + the cocycle, tried honestly against FSX truth.
Gates are instrument correctness (sections register onto the shared base, warp is monotone, gluing computes).
The payoff question -- do section-level (curve-shape + cocycle) features recover signal the scalar collapse
lost, and does the cocycle residual localize the mishits -- is reported as findings, with the SAME good-shots
robustness the scalar analysis got. Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.sheaf import build_sheaf, glue, fpca, joint_features
from cocycle.launch import parse_fsx_pdf, join_session, loo_r2, screen_corr

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"
PDF = os.path.join(SESSION, "FSX_Shotresults.pdf")


def loo_good(F, y, mis):
    good = np.array([i not in mis for i in range(len(y))])
    return loo_r2(F[good], y[good])


def main():
    sh = build_sheaf(SESSION, N=256)
    ids, names, F, evr = joint_features(sh, k=4)
    g = glue(sh["omega"])

    # instrument gates
    s1 = sh["omega"].shape == (len(ids), 256) and np.isfinite(sh["omega"]).all() and np.isfinite(sh["energy"]).all()
    s2 = 0.0 < sh["P"][0] < sh["P"][1] < 1.0                          # canonical top before impact, monotone
    s3 = np.isfinite(F).all() and F.shape[0] == len(ids)
    s4 = np.isfinite(g["residual"]).all() and (g["residual"] >= 0).all()

    ids_y, lnames, Y = parse_fsx_pdf(PDF)
    ids_c, Fj, Yj = join_session(ids, F, ids_y, Y)
    # realign glued residual to the joined id order
    idx = {s: k for k, s in enumerate(ids)}
    resid = g["residual"][[idx[s] for s in ids_c]]
    def yc(n): return Yj[:, lnames.index(n)]

    # the cocycle localizes mishits? rank the 2 catastrophic by residual
    bs = yc("ball_speed_mph"); mis_ids = [ids_c[i] for i in range(len(ids_c)) if bs[i] < 0.7 * np.median(bs)]
    order = [ids_c[i] for i in np.argsort(-resid)]
    rank_mis = [order.index(m) + 1 for m in mis_ids]
    mis = {i for i in range(len(ids_c)) if ids_c[i] in mis_ids}

    print(f"[SHEAF] built: {len(ids)} sections restricted onto shared base N=256; canonical top={sh['P'][0]:.2f} impact={sh['P'][1]:.2f}")
    print(f"[SHEAF] fPCA explained-var: omega {np.round(evr['omega_evr'],2)}  energy {np.round(evr['energy_evr'],2)}")
    print(f"[SHEAF] gluing tightness (mean cross-shot std of omega): {sh['omega'].std(0).mean():.0f} deg/s")
    print(f"[SHEAF] COCYCLE residual: catastrophic mishits {mis_ids} rank {rank_mis} of {len(ids_c)} by |residual| "
          f"(1 = largest obstruction)  -> {'localized at the top' if max(rank_mis)<=4 else 'NOT localized'}")

    print("\n[SHEAF] section-level features vs FSX truth (LOO-CV R^2; ALL30 vs GOOD, honest):")
    r2s = {}
    for tg in ("club_speed_mph", "ball_speed_mph", "carry_yds", "offline_yds", "efficiency"):
        y = yc(tg)
        corr = screen_corr(Fj, names, y); top = [n for n, _ in corr[:3]]
        cols = [names.index(n) for n in top]
        rA, _ = loo_r2(Fj[:, cols], y); rG, nG = loo_good(Fj[:, cols], y, mis)
        r2s[tg] = (rA, rG)
        cs = ", ".join(f"{n} {r:+.2f}" for n, r in corr[:3])
        print(f"    {tg:16s} ALL R^2={rA:+.2f}  GOOD{nG} R^2={rG:+.2f}   top: {cs}")

    # |residual| as a standalone mishit/offline score
    roff = float(np.corrcoef(resid, np.abs(yc("offline_yds")))[0, 1])
    print(f"\n[SHEAF] cocycle residual -> |offline| r={roff:+.2f}  (does the gluing obstruction track dispersion?)")

    checks = {
        "SH1 all sections registered onto shared base, finite": bool(s1),
        "SH2 restriction warp monotone (top before impact on base)": bool(s2),
        "SH3 joint section-level feature matrix well-formed": bool(s3),
        "SH4 gluing + cocycle residual computed (nonneg, finite)": bool(s4),
        "SH5 cocycle residual localizes the catastrophic mishits (top-4)": bool(max(rank_mis) <= 4),
    }
    print("\n[SHEAF] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[SHEAF] {'ALL CRITERIA MET -- sheaf built by restriction+gluing; cocycle = the obstruction/signal' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
