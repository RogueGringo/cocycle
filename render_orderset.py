"""Session-narrative render: the 30 shots as an ORDERED story. FSX club speed over shot order with regime
bands (stable -> crash -> surge -> settle) from its own changepoints, overlaid with the IMU's independent
changepoint VOTES -- showing the shaft alone flags the crash (17) and the surge (22). PIL PNG."""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
from cocycle.sheaf import build_sheaf
from cocycle.orderset import decompose, binseg, phase_windows
from cocycle.launch import parse_fsx_pdf

S = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"
W, H = 1040, 600
BG, INK, DIM, CYAN, AMBER, BAD = (10, 13, 18), (231, 237, 243), (139, 151, 165), (53, 224, 208), (240, 182, 74), (255, 106, 90)


def font(sz):
    try: return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        try: return ImageFont.truetype("DejaVuSans-Bold.ttf", sz)
        except Exception: return ImageFont.load_default()


def main():
    sh = build_sheaf(S, N=256); dec = decompose(sh); ids = dec["ids"]
    iy, ln, Y = parse_fsx_pdf(S + r"\FSX_Shotresults.pdf"); yi = {s: k for k, s in enumerate(iy)}
    cs = np.array([Y[yi[s], ln.index("club_speed_mph")] for s in ids])
    bs = np.array([Y[yi[s], ln.index("ball_speed_mph")] for s in ids])
    mis = {i for i in range(len(ids)) if bs[i] < 0.7 * np.median(bs)}
    fcp = binseg(cs)
    votes = Counter()
    for v in list(dec["scale"].values()) + list(dec["shape"].values()):   # flat list: dicts share keys
        for c in binseg(v): votes[c] += 1
    from cocycle.orderset import convergence_null
    _, pval = convergence_null(list(dec["scale"].values()) + list(dec["shape"].values()), fcp, T=2000)

    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    L, R, T, B = 60, W - 24, 70, 430
    n = len(ids)
    def X(i): return L + i / (n - 1) * (R - L)
    ymin, ymax = cs.min() - 6, cs.max() + 6
    def Yv(v): return B - (v - ymin) / (ymax - ymin) * (B - T)

    d.text((L, 22), "SESSION PROFILE — FSX outcomes vs IMU changepoints", font=font(20), fill=INK)
    d.text((L, 48), f"IMU energy shifts land near the crash (17) & surge (22), BUT permutation null p={pval:.2f} "
            f"— alignment is chance, not recovery", font=font(12), fill=DIM)

    # regime bands from FSX changepoints
    bnds = [0] + fcp + [n]; labels = ["stable", "crash / recovery", "surge", "settle"]
    for k in range(len(bnds) - 1):
        a, b = bnds[k], bnds[k + 1]
        shade = (18, 22, 30) if k % 2 == 0 else (24, 20, 26)
        d.rectangle([X(a) - (0 if a == 0 else 0), T, X(min(b, n - 1)), B], fill=shade)
    for k in range(len(bnds) - 1):
        a, b = bnds[k], bnds[k + 1]; lab = labels[k] if k < len(labels) else ""
        d.text(((X(a) + X(min(b, n - 1))) / 2 - len(lab) * 3, T + 6), lab, font=font(12), fill=DIM)
    # FSX changepoint verticals
    for c in fcp:
        d.line([X(c), T, X(c), B], fill=(70, 80, 92), width=1)
    # y grid
    for gv in range(int(ymin // 10 * 10), int(ymax) + 1, 10):
        if ymin <= gv <= ymax:
            d.line([L, Yv(gv), R, Yv(gv)], fill=(30, 36, 44)); d.text((L - 26, Yv(gv) - 6), f"{gv}", font=font(11), fill=DIM)

    # club speed line + points
    d.line([(X(i), Yv(cs[i])) for i in range(n)], fill=CYAN, width=2)
    for i in range(n):
        col = BAD if i in mis else CYAN
        r = 4 if i in mis else 3
        d.ellipse([X(i) - r, Yv(cs[i]) - r, X(i) + r, Yv(cs[i]) + r], fill=col)
    d.text((L, Yv(cs[0]) - 20), "club speed (mph)", font=font(11), fill=CYAN)

    # IMU changepoint votes lane
    VT, VB = B + 34, H - 46
    d.text((L, VT - 18), "IMU changepoint votes (12 independent phase-elements)", font=font(12), fill=DIM)
    vmax = max(votes.values()) if votes else 1
    for i in range(n):
        v = votes.get(i, 0)
        if v == 0: continue
        corrob = any(abs(i - c) <= 1 for c in fcp) and v >= 2
        col = AMBER if corrob else (70, 80, 92)
        hgt = v / vmax * (VB - VT)
        d.rectangle([X(i) - 6, VB - hgt, X(i) + 6, VB], fill=col)
        d.text((X(i) - 6, VB + 2), f"{v}", font=font(11), fill=col)
    # x labels (every shot number)
    for i in range(n):
        d.text((X(i) - 6, H - 24), f"{int(ids[i].replace('Shot',''))}", font=font(9),
                fill=BAD if i in mis else DIM)
    d.text((L, H - 40), "shot #", font=font(11), fill=DIM)

    img.save("render_orderset.png")
    print(f"wrote render_orderset.png | FSX changepoints {[ids[c] for c in fcp]} | "
          f"IMU-corroborated {[ids[c] for c in fcp if sum(votes.get(d_,0) for d_ in (c-1,c,c+1))>=2]}")


if __name__ == "__main__":
    main()
