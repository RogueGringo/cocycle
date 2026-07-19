# Spec — Multi-Club Capture: the design that can answer the prediction question

status: proposed · date: 2026-07-19 · supersedes the null verdict of the 30-shot single-club session by fixing its ROOT CAUSE.

## Why the first session couldn't answer it (the diagnosis, not a guess)

Six independent representations (scalars → registered curves → fPCA → images → order-set → mechanically-anchored)
plus a physics reconstruction ALL returned the same verdict on `30SwingSession_Individual`: the shaft IMU predicts
**nothing** about launch outcomes among the good shots, while it robustly does mishit detection (the cocycle) and
motion reconstruction. That is not a modelling failure — it is a **data-design** failure with one cause:

- **No outcome variance to model.** One club (5-iron), one consistent player: club-speed CV ≈ 5%. Nearly ALL the
  outcome variance across the session was 2 catastrophic mishits (17, 18). Remove them and the outputs are a tight
  cluster — there is nothing left for any input to predict. Every "signal" we found was that 2-shot cluster in
  disguise (each collapsed on the good-shots split or its permutation null).

The fix is to **manufacture outcome variance deliberately and cleanly**, then re-run the exact same honest harness.

## Objective & success criteria (pre-registered, to avoid the p-hacking the whole thread guarded against)

Answer, with adequate statistical power: *does the shaft IMU predict launch-monitor outcomes (club/ball speed,
carry, launch, spin, dynamic loft) beyond the trivial?* Two acceptable outcomes, both valuable:

- **G-POS**: a held-out predictive model with honest LOO/grouped-CV R² ≥ 0.30 on a target, that **survives the
  good-shots split AND a club-stratified permutation null** (i.e., predicts *within* club, not just "which club").
- **G-NULL**: a *powered* null — if no signal, we can state it at power ≥ 0.8 to detect r = 0.3 (see below), which
  the n=30 session could not (it was powered only for r ≥ 0.5). A powered null is a real scientific result.

The pass condition for the CAPTURE itself is the power + hygiene criteria below; the model result is the experiment.

## Power analysis (grounds the shot counts — computed, not asserted)

n to detect a Pearson r at power 0.8, α=0.05 (Fisher-z): **r=0.5→30, r=0.4→47, r=0.3→85, r=0.2→194**. Regression
rule: n ≥ 10–15 obs per predictor. Our features gave good-shots |r| < 0.2 on the first session, so to *rule in or
out* a real small effect we need **n ≈ 85–120 per condition of interest**, and ≥ 60–90 total for a 6-feature model.

## Design

Three axes of variance, cheapest-first:

1. **Multi-club (between-club range).** 4 clubs spanning the envelope: **Driver, 6-iron, 9-iron, Pitching Wedge**
   (~40 mph of club speed, wide launch/spin/attack range). This gives a large *outcome* range the single 5-iron
   could not — but a model that only says "fast = driver" is trivial, so multi-club is necessary, not sufficient.
2. **Effort blocks (within-club variance — the direct fix for CV≈5%).** Per club, hit deliberate blocks at
   **~70% / ~85% / 100% effort** (call it before each block). This injects controlled club-speed variance *within a
   club*, so within-club prediction becomes testable — attacking the exact cause of the first null.
3. **Players (generalization + between-subject variance), Phase 2.** Start with 1 player (Phase 1); add 2–3 more
   (varied handicap) once Phase 1 clears hygiene, to test cross-subject generalization and widen the space.

**Counts (Phase 1, one player):** 4 clubs × 3 effort blocks × ~12 shots = **~144 shots**. Gives ≈ 36 shots/club
(within-club power for r ≈ 0.45), ~48/effort-condition, and 144 total (powered to r ≈ 0.22 pooled). Phase 2 (4
players) → ~576 shots, powered to r ≈ 0.3 within reasonable strata.

## Capture hygiene (each item traces to a defect we hit)

- **Sensor range / saturation.** The first IMU was ±16 g/axis and **saturated at every impact** (27.7 g = 16√3),
  killing impact_g. Use a **±32 g (or higher) accelerometer** if available, or accept gyro-only for club dynamics
  and record the range in metadata. Gyro must not clip either (peaks ~3000 °/s; use ≥ ±4000 °/s FSR).
- **Fixed, identical shaft mount** (same position/orientation across all shots) so `s_hat` and the lever geometry
  are constant — required for the reconstruction and mechanical-event detection to be comparable.
- **Clean shot correspondence.** First session joined IMU↔FSX by *shot order* (fragile). Add a **hard sync**: a
  deliberate tap on the shaft (a sharp gyro/accel spike) at session start, and log wall-clock per shot on both
  devices, so the join is by timestamp with an order fallback. One mismatched row poisons the labels.
- **Swing-vs-other isolation is already built** (`mechanics.isolate_swing`): 12/30 first-session files contained
  practice-swing/waggle motion. Keep ≥ 3 s of quiet before each swing; the isolator excludes the rest.
- **Metadata per shot:** club, effort block, player, sensor FSR, sync offset, ball model. No PII beyond a player id.

## Analysis plan (pre-registered — run the SAME harness, no new degrees of freedom)

1. Bake every shot through the existing pipeline **unchanged**: `swing.py` (24 features incl. biomech),
   `mechanics.py` (P2/P6/delivery, isolation), `sheaf.py` (registered sections + cocycle), `reconstruct.py`.
2. Join labels via `launch.py` (generalize the parser if the monitor differs from Foresight FSX; the join +
   `loo_r2` + `screen_corr` are monitor-agnostic).
3. **Grouped CV**: leave-one-CLUB-out and leave-one-PLAYER-out (not just LOO shot) — the real generalization test.
4. Every claim passes **all three** guards that killed the false positives here: good-shots split, club-stratified
   permutation null, and held-out grouped CV. Report effect sizes with CIs, not just R².
5. Pre-commit the target list and the 6-feature sets *before* seeing labels (the file is the pre-registration).

## What the existing code does unchanged vs. what to adapt

- **Unchanged:** all of `swing/mechanics/sheaf/reconstruct/imprint/orderset`, the honest-harness verifiers, the
  cocycle quality map, the signal bake. Multi-rate CSV ingest, impact/segmentation, S³ geometry — all club-agnostic.
- **Adapt (small):** `launch.py` parser if the launch monitor isn't FSX PRO (one format adapter); add `club`/
  `effort`/`player` columns to the join; add grouped-CV wrappers (leave-one-club/player-out) beside `loo_r2`.

## Go / no-go gates for the capture

- C1: sync verified (IMU tap spike aligns to LM first-shot timestamp) before collecting the full session.
- C2: no accelerometer saturation on a test strike at max effort (or range logged and gyro-only accepted).
- C3: each shot isolates to a single clean swing (`isolate_swing` clean-flag) on a 10-shot pilot.
- C4: ≥ 140 shots (Phase 1) / ≥ 550 (Phase 2) with < 5% unusable → then the prediction question is finally powered.
