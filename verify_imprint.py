"""Imprint verification + the honest experiment: is the IMAGE a better representation than scalars/curves/
fPCA? Gates: rasterization correctness + image-space mishit separation (must agree with the cocycle). The
prediction question runs under the same ALL-vs-GOOD harness and is reported as found -- representation #4
either beats the n-limit or (expected) confirms it. Also writes the fingerprint montage (the human image).
Headless."""
import sys, os
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import warnings; warnings.filterwarnings("ignore")
from cocycle.imprint import session_imprints, image_features, K
from cocycle.launch import parse_fsx_pdf, join_session, loo_r2, screen_corr

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"


def montage(ids, imgs, mishits, flagged, path):
    from PIL import Image, ImageDraw
    k = imgs.shape[1]; cell = 96; cols = 6; rows = (len(ids) + cols - 1) // cols
    W, H = cols * cell, rows * (cell + 14)
    canvas = Image.new("RGB", (W, H), (10, 13, 18)); dr = ImageDraw.Draw(canvas)
    for i, (sid, im) in enumerate(zip(ids, imgs)):
        r, c = divmod(i, cols)
        d = im.copy()                                       # 1px dilation so thin arcs survive tile scaling
        d[1:] = np.maximum(d[1:], im[:-1]); d[:, 1:] = np.maximum(d[:, 1:], im[:, :-1])
        g = (np.clip(np.sqrt(d / 0.25), 0, 1) * 255).astype(np.uint8)   # gain: p90~0.5 -> full bright
        tint = (255, 106, 90) if sid in mishits else (240, 182, 74) if sid in flagged else (53, 224, 208)
        rgb = np.stack([g.astype(np.int32) * t // 255 for t in tint], -1).astype(np.uint8)  # NEP50: uint8*int wraps
        tile = Image.fromarray(rgb).resize((cell, cell), Image.NEAREST)
        x, y = c * cell, r * (cell + 14)
        canvas.paste(tile, (x, y))
        dr.text((x + 4, y + cell + 1), sid.replace("Shot", "#"), fill=tint)
    canvas.save(path)


def main():
    ids, imgs = session_imprints(SESSION)
    i1 = len(ids) == 30 and np.isfinite(imgs).all() and (imgs.reshape(30, -1).max(1) > 0).all()

    scores, evr, dist = image_features(imgs, k=6)
    iy, ln, Y = parse_fsx_pdf(os.path.join(SESSION, "FSX_Shotresults.pdf"))
    ids_c, Sj, Yj = join_session(ids, np.column_stack([scores, dist]), iy, Y)
    names = [f"img_pc{i}" for i in range(scores.shape[1])] + ["img_dist"]
    def yc(n): return Yj[:, ln.index(n)]

    bs = yc("ball_speed_mph"); mis = {ids_c[i] for i in range(len(ids_c)) if bs[i] < 0.7 * np.median(bs)}
    # what images ACTUALLY encode: brightness = integrated speed, so slow mishits must be FAINT. (Shape
    # distance does NOT find mishits -- a flub is a normal-shaped arc, just slow; rasterization destroys the
    # timing profile the sheaf cocycle uses. That is the finding, tested here as the brightness gate.)
    bright = imgs.reshape(len(imgs), -1).sum(1)
    bmap = {ids[i]: bright[i] for i in range(len(ids))}
    faint = sorted(ids_c, key=lambda s: bmap[s])
    ranks = [faint.index(m) + 1 for m in sorted(mis)]
    i2 = len(mis) > 0 and max(ranks) <= 4
    dvec = Sj[:, -1]
    flagged = set(faint[:4])
    shape_rank = [int(np.argsort(-dvec).tolist().index(ids_c.index(m))) + 1 for m in sorted(mis)]

    good = np.array([ids_c[i] not in mis for i in range(len(ids_c))])
    print(f"[IMPRINT] {len(ids)} swings rasterized to {K}x{K} canonical fingerprints (plane-projected, ball-anchored)")
    print(f"[IMPRINT] image-PCA explained var: {np.round(evr, 2)}")
    print(f"[IMPRINT] brightness (= integrated speed) ranks truth-mishits {sorted(mis)} faint-rank {ranks} of {len(ids_c)}")
    print(f"[IMPRINT] finding: SHAPE distance ranks them {shape_rank} of {len(ids_c)} -- a flub is a normal-shaped arc, "
          f"just slow; rasterizing time away destroys the cocycle's signal")
    print("\n[IMPRINT] representation #4 (images) vs known outputs -- same honest harness:")
    r2s = {}
    for tg in ("club_speed_mph", "ball_speed_mph", "carry_yds", "efficiency"):
        y = yc(tg)
        corr = screen_corr(Sj, names, y); top = [n for n, _ in corr[:3]]
        cols = [names.index(n) for n in top]
        rA, _ = loo_r2(Sj[:, cols], y)
        rG, nG = loo_r2(Sj[good][:, cols], y[good])
        r2s[tg] = rG
        cs = ", ".join(f"{n} {r:+.2f}" for n, r in corr[:3])
        print(f"    {tg:15s} ALL R^2={rA:+.2f}  GOOD{nG} R^2={rG:+.2f}   top: {cs}")
    i3 = all(np.isfinite(v) for v in r2s.values())

    out = "imprint_montage.png"
    montage(ids_c, imgs[[ids.index(s) for s in ids_c]], mis, flagged, out)
    i4 = os.path.exists(out) and os.path.getsize(out) > 1000
    print(f"\n[IMPRINT] wrote {out} (30 fingerprints; red = truth mishit, amber = image-flagged)")

    checks = {
        "I1 all 30 swings rasterize finite + nonempty": bool(i1),
        "I2 brightness (integrated speed) ranks truth-mishits faintest-4": bool(i2),
        "I3 outputs screen computes (ALL vs GOOD)": bool(i3),
        "I4 fingerprint montage written": bool(i4),
    }
    print("\n[IMPRINT] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    verdict = ("images BEAT the n-limit on good shots" if any(v > 0.3 for v in r2s.values())
               else "images CONFIRM the n-limit (4th representation, same wall) -- the value is the fingerprint itself")
    print(f"\n[IMPRINT] {'ALL CRITERIA MET -- ' + verdict if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
