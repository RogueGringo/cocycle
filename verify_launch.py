"""Stage-2 verification — close inference:model:output on the REAL session: parse the FSX PRO launch-monitor
PDF (the output), join to the swing-bake feature matrix (the model), validate the unsupervised mishit
detector against ground truth, and run the honest inference screen (correlations + leave-one-out R^2).
Gates are on INSTRUMENT correctness (parse checksum, join integrity, computation validity); the correlation
values themselves are findings and are reported as found. Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.swing import bake_session, swing_consistency, FEATURES
from cocycle.launch import parse_fsx_pdf, join_session, loo_r2, screen_corr, LABELS

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"
PDF = os.path.join(SESSION, "FSX_Shotresults.pdf")


def main():
    ids_y, lnames, Y = parse_fsx_pdf(PDF)                       # checksums itself vs the printed Avg rows
    l1 = len(ids_y) == 30 and Y.shape == (30, len(LABELS))

    ids_x, stamps, fnames, X = bake_session(SESSION)
    ids, Xj, Yj = join_session(ids_x, X, ids_y, Y)
    l2 = len(ids) == 30 and np.isfinite(Xj).all()

    def ycol(n): return Yj[:, lnames.index(n)]
    def xcol(n): return Xj[:, fnames.index(n)]

    # L3: unsupervised mishit detector vs ground truth (catastrophic = ball speed < 70% of session median)
    cons = swing_consistency(ids, fnames, Xj)
    flagged = set(cons["outliers"].keys())
    bs = ycol("ball_speed_mph")
    catastrophic = {ids[i] for i in range(len(ids)) if bs[i] < 0.70 * np.median(bs)}
    l3 = catastrophic <= flagged

    # L4: the inference screen — IMU features -> club speed & ball speed (top-3 screened features, LOO ridge)
    results = {}
    for target in ("club_speed_mph", "ball_speed_mph", "carry_yds", "efficiency"):
        y = ycol(target)
        corr = screen_corr(Xj, fnames, y)
        top = [nm for nm, _ in corr[:3]]
        r2, n = loo_r2(Xj[:, [fnames.index(nm) for nm in top]], y)
        results[target] = (corr[:3], top, r2, n)
    l4 = all(np.isfinite(r2) for _, _, r2, _ in results.values())

    # L5: the club-speed proxy calibration (the headline validation of the label-free path)
    proxy = xcol("club_speed_ms") * 2.237                        # -> mph
    truth = ycol("club_speed_mph")
    r_proxy = float(np.corrcoef(proxy, truth)[0, 1])
    scale = float(np.mean(truth / proxy))
    l5 = np.isfinite(r_proxy) and 0.1 < scale < 2.0

    print(f"[LAUNCH] parsed {len(ids_y)} shots x {Y.shape[1]} labels from FSX PDF (checksum vs printed Avg: OK)")
    print(f"[LAUNCH] joined {len(ids)}/30 shots: X[{Xj.shape[0]}x{Xj.shape[1]}] features -> Y[{Yj.shape[0]}x{Yj.shape[1]}] labels")
    print(f"[LAUNCH] ground truth: ball {np.nanmean(bs):.1f} mph, club {np.nanmean(truth):.1f} mph, "
          f"carry {np.nanmean(ycol('carry_yds')):.1f} yds")
    print(f"\n[LAUNCH] mishit detector: flagged {sorted(flagged)} | catastrophic (truth) {sorted(catastrophic)} "
          f"| caught {len(catastrophic & flagged)}/{len(catastrophic)}")
    print(f"\n[LAUNCH] club-speed proxy vs truth: r = {r_proxy:+.2f}, scale factor = {scale:.2f} "
          f"(proxy {np.mean(proxy):.0f} mph vs truth {np.mean(truth):.0f} mph)")
    print("\n[LAUNCH] inference screen (top IMU correlates + LOO-CV R^2, n=30 -- small-sample, read as signal not certainty):")
    for target, (corr, top, r2, n) in results.items():
        cs = ", ".join(f"{nm} {r:+.2f}" for nm, r in corr)
        print(f"    {target:15s} LOO R^2 = {r2:+.2f} (n={n})   correlates: {cs}")

    checks = {
        "L1 FSX PDF parses: 30 shots x 21 labels, checksummed vs printed Avg": l1,
        "L2 join integrity: 30/30 shots matched, features finite": l2,
        "L3 unsupervised mishit detector catches all catastrophic mishits": l3,
        "L4 inference screen computes (finite LOO R^2 on all targets)": l4,
        "L5 club-speed proxy calibration well-defined": l5,
    }
    print("\n[LAUNCH] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[LAUNCH] {'ALL CRITERIA MET — inference:model:output is CLOSED on the real session' if ok else 'FAILURES ABOVE'}")

    if ok:  # persist the joined table next to the dataset (data stays with data)
        out = os.path.join(SESSION, "session_joined.csv")
        with open(out, "w") as f:
            f.write("shot," + ",".join(fnames) + "," + ",".join(lnames) + "\n")
            for i, s in enumerate(ids):
                f.write(s + "," + ",".join(f"{v:.6g}" for v in Xj[i]) + ","
                        + ",".join(f"{v:.6g}" for v in Yj[i]) + "\n")
        print(f"[LAUNCH] wrote {out} ({len(ids)} rows x {len(fnames)+len(lnames)} cols)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
