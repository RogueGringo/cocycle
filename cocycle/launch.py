"""Stage-2 'output' half of inference:model:output — parse a Foresight FSX PRO session-report PDF into
per-shot launch-monitor labels (ball flight + club delivery) and join them to the swing-bake feature matrix.
Join key = shot index: both sources are session-ordered on the same date (IMU zips Shot1..N chronological;
FSX details rows 1..N in session order). Missing values ('-', e.g. undetected loft on a mishit) -> NaN.
The printed Avg rows in the PDF are used as a parse checksum. numpy + pypdf only."""
from __future__ import annotations
import re
import numpy as np

BALL_COLS = ("ball_speed_mph", "launch_dir_deg", "launch_angle_deg", "backspin_rpm", "sidespin_rpm",
             "curve_yds", "carry_yds", "total_yds", "offline_yds", "descent_deg")
CLUB_COLS = ("club_speed_mph", "club_path_deg", "face_to_target_deg", "face_to_path_deg", "attack_angle_deg",
             "dynamic_loft_deg", "efficiency", "lie_angle_deg", "vertical_impact_mm", "club_offline_yds",
             "horizontal_impact_mm")
LABELS = BALL_COLS + CLUB_COLS

_NUM = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


def _tok_val(tok):
    if tok == "-":
        return np.nan
    return float(tok) if _NUM.match(tok) else None


def _parse_rows(text, ncols):
    """Rows look like '  7 116.5 +5.3 ... 45.8' — a shot index then ncols values ('-' = missing)."""
    rows = {}
    for line in text.splitlines():
        toks = line.split()
        if not toks or not toks[0].isdigit():
            continue
        idx = int(toks[0])
        vals = [_tok_val(t) for t in toks[1:]]
        if None in vals or len(vals) != ncols or not (1 <= idx <= 99):
            continue
        rows[idx] = vals
    return rows


def _parse_avg(text, ncols):
    for line in text.splitlines():
        toks = line.split()
        if toks and toks[0].rstrip(".") == "Avg":
            vals = [_tok_val(t) for t in toks[1:]]
            if len(vals) == ncols and None not in vals:
                return np.array(vals, np.float64)
    return None


def parse_fsx_pdf(path):
    """FSX PRO session report -> (shot_ids, label_names, Y[S, L]) with NaN for undetected metrics."""
    from pypdf import PdfReader
    r = PdfReader(path)
    ball, club, ball_avg, club_avg = {}, {}, None, None
    for pg in r.pages:
        text = pg.extract_text() or ""
        if "Details" not in text:
            continue
        is_club = "CLUB" in text and "SPEED" in text and "EFFICIENCY" in text
        rows = _parse_rows(text, len(CLUB_COLS) if is_club else len(BALL_COLS))
        avg = _parse_avg(text, len(CLUB_COLS) if is_club else len(BALL_COLS))
        if is_club:
            club.update(rows); club_avg = avg if avg is not None else club_avg
        else:
            ball.update(rows); ball_avg = avg if avg is not None else ball_avg
    shots = sorted(set(ball) & set(club))
    if not shots:
        raise ValueError(f"{path}: no per-shot detail rows parsed")
    Y = np.array([ball[s] + club[s] for s in shots], np.float64)
    # checksum the parse against the report's own printed averages (ignoring NaN rows, as FSX does).
    # horizontal_impact is checked on |magnitude|: FSX prints its summary Avg with the OPPOSITE sign of the
    # mean of its own detail rows (-14 vs +14.2 here; std matches) -- a sign-convention quirk in the report.
    for avg, cols, off in ((ball_avg, BALL_COLS, 0), (club_avg, CLUB_COLS, len(BALL_COLS))):
        if avg is not None:
            ours = np.nanmean(Y[:, off:off + len(cols)], axis=0)
            bad = [cols[j] for j in range(len(cols)) if np.isfinite(avg[j])
                   and abs((abs(ours[j]) if cols[j] == "horizontal_impact_mm" else ours[j])
                           - (abs(avg[j]) if cols[j] == "horizontal_impact_mm" else avg[j])) > 0.6]
            if bad:
                raise ValueError(f"parse checksum failed vs printed Avg on {bad}")
    return [f"Shot{s}" for s in shots], list(LABELS), Y


def join_session(ids_x, X, ids_y, Y):
    """Inner-join feature matrix X and label matrix Y on shot id; returns (ids, X', Y')."""
    common = [i for i in ids_x if i in set(ids_y)]
    xi = {s: k for k, s in enumerate(ids_x)}; yi = {s: k for k, s in enumerate(ids_y)}
    return common, X[[xi[s] for s in common]], Y[[yi[s] for s in common]]


def loo_r2(X, y, ridge=1e-3):
    """Leave-one-out cross-validated R^2 of a standardized ridge linear model (numpy only). Honest for
    small n: every prediction is made by a model that never saw that shot."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    n = len(y)
    if n < 5:
        return np.nan, ok.sum()
    preds = np.empty(n)
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-12
        Z = (X[m] - mu) / sd
        A = Z.T @ Z + ridge * len(y[m]) * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (y[m] - y[m].mean()))
        preds[i] = y[m].mean() + ((X[i] - mu) / sd) @ w
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - preds) ** 2) / (ss + 1e-12)), n


def screen_corr(X, names_x, y):
    """Pearson r of each finite feature against y (NaN-safe); returns [(name, r)] sorted by |r|."""
    out = []
    for j, nm in enumerate(names_x):
        x = X[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5 or x[ok].std() == 0 or y[ok].std() == 0:
            continue
        r = float(np.corrcoef(x[ok], y[ok])[0, 1])
        out.append((nm, r))
    return sorted(out, key=lambda t: -abs(t[1]))
