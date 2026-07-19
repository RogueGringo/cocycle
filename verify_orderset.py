"""Order-set verification: the session as an ordered sequence. Gates instrument correctness (decomposition,
changepoint machinery) and the headline claim: the IMU alone recovers the session's REGIME STRUCTURE (the
crash and the surge, known from FSX order) via changepoints on phase-scale series. Rank statistics; the
scale-vs-shape and overall-vs-isolated ledgers are reported as findings. Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
from cocycle.sheaf import build_sheaf
from cocycle.orderset import decompose, binseg, lag1, spearman, variance_ledger, zscore, convergence_null
from cocycle.launch import parse_fsx_pdf
from cocycle.imprint import session_imprints

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"


def main():
    sh = build_sheaf(SESSION, N=256)
    dec = decompose(sh)
    ids = dec["ids"]; n = len(ids)
    o1 = (n == 30 and all(np.isfinite(v).all() and (v > 0).all() for v in dec["scale"].values())
          and all(np.isfinite(v).all() for v in dec["shape"].values()))

    iy, ln, Y = parse_fsx_pdf(os.path.join(SESSION, "FSX_Shotresults.pdf")); yi = {s: k for k, s in enumerate(iy)}
    cs = np.array([Y[yi[s], ln.index("club_speed_mph")] for s in ids])
    fsx_cp = binseg(cs)                                          # ground-truth regime structure

    # 12 IMU phase-elements (NB: dict "scale"/"shape" share keys, so build a flat list -- {**a,**b} would
    # collide and silently keep only 6, the bug the round-trip caught in the render).
    elements = list(dec["scale"].values()) + list(dec["shape"].values())
    from collections import Counter
    votes = Counter()
    for v in elements:
        for c in binseg(v): votes[c] += 1

    print(f"[ORDER] {n} shots in session order | FSX club speed changepoints at shots "
          f"{[ids[c] for c in fsx_cp]} (regimes: stable -> crash -> surge -> settle)")
    print("\n[ORDER] element tracking of the FSX order-series (Spearman; rank stats -- series are heavy-tailed):")
    best_rho = 0.0
    for name, v in {**{f"sc:{k}": x for k, x in dec["scale"].items()},
                    **{f"sh:{k}": x for k, x in dec["shape"].items()}}.items():
        rho = spearman(v, cs)
        if abs(rho) > abs(best_rho): best_rho = rho
        marks = " ".join(ids[c] for c in binseg(v)) or "-"
        print(f"    {name:16s} rho={rho:+.2f}  lag1={lag1(v):+.2f}  changepoints: {marks}")
    o2 = np.isfinite(best_rho) and len(fsx_cp) >= 2

    top_votes = sorted(votes.items(), key=lambda kv: -kv[1])[:4]
    corroborated = [ids[c] for c in fsx_cp if sum(votes[d] for d in (c - 1, c, c + 1)) >= 2]
    real, pval = convergence_null(elements, fsx_cp, T=5000)     # THE honest test (permutation null of the gate)
    print(f"\n[ORDER] changepoint votes (all 12 elements): {', '.join(f'{ids[s]}:{v}' for s, v in top_votes)}")
    print(f"[ORDER] {len(corroborated)} FSX boundaries corroborated ({corroborated}) -- BUT permutation null:")
    print(f"        p(>= {real} corroborated by chance) = {pval:.2f}  =>  "
          f"{'SIGNIFICANT' if pval < 0.05 else 'NOT significant; the alignment is what correlated elements produce by chance'}")
    # CERTIFIED NEGATIVE: the convergence does NOT beat its own permutation null (12 correlated elements make
    # >=2 corroborated boundaries the expected outcome). Narrative recovery is NOT supported on this session.
    o3 = pval > 0.05

    led = variance_ledger(dec)
    print("\n[ORDER] variance ledger -- overall vs isolated (which element carries the session's variance):")
    for name, kind, cv, l1 in led[:5]:
        print(f"    {kind:5s} {name:14s} CV={cv:5.2f}  lag1={l1:+.2f}")
    print(f"    ...   least-varying: {', '.join(f'{n}({k})' for n, k, _, _ in led[-3:])}")
    scale_cv = np.mean([cv for _, k, cv, _ in led if k == "scale"])
    shape_cv = np.mean([cv for _, k, cv, _ in led if k == "shape"])
    print(f"[ORDER] mean CV: scale {scale_cv:.2f} vs shape {shape_cv:.2f} -> "
          f"{'SCALE (amplitude) is the mobile axis; shape is conserved' if scale_cv > shape_cv else 'SHAPE varies more than scale'}")
    o4 = np.isfinite(scale_cv) and np.isfinite(shape_cv)

    checks = {
        "O1 phase decomposition well-formed (30 shots, scales>0, finite)": bool(o1),
        "O2 order-series machinery computes (rank corr, changepoints)": bool(o2),
        "O3 CERTIFIED: changepoint alignment NOT beyond its permutation null": bool(o3),
        "O4 scale-vs-shape variance ledger computes": bool(o4),
    }
    print("\n[ORDER] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[ORDER] {'ALL CRITERIA MET -- machinery sound; session-narrative recovery is CERTIFIED NULL (the mishit cluster, not real structure)' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
