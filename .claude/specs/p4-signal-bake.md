# P4 Spec — offline signal bake → renderer-agnostic elementals timeline

status: approved · date: 2026-07-18 · reframes COCYCLE: signal → elementals → transpose-to-any-regime (audio is one source; MWD transducer files are another).

## Objective
An offline `bake` stage: user params, ~seconds of compute, parses a signal FILE into a dense, **renderer-agnostic**
timeline of "elementals" at a target framerate (default 144 fps, floor 90). Because the timeline is decoupled from
any scene, the SAME bake drives any regime — the rank-7 field, Hopf, LED, binaural — and any SIGNAL — audio, or
raw multi-channel sensor data (MWD drilling transducers).

## The elementals (signal-agnostic per channel)
level (rms, peak) · 7 log band energies · spectral (centroid, rolloff, flux, flatness) · temporal (zcr, onset
strength) · periodicity (dominant frequency → pitch for audio, RPM for a rotating tool). Baked with GLOBAL
per-track normalization (the offline advantage over the real-time EMA extractor).

## Constraints
- Output timeline: `[n_frames, n_elementals]` at the target fps (>=90), + names + fps + duration.
- Renderer-agnostic: `Timeline.to_features(i)` returns the FeatureExtractor-compatible dict, so existing scenes
  consume the bake UNCHANGED (that IS "transpose against any regime").
- Signal-agnostic: `bake_signal(x, sr, ...)` is the core; `bake_audio(path)` and multi-channel `bake_multichannel`
  are wrappers. Window/freq-range scale with sr so low-rate sensor data (MWD) works with the same code.
- numpy only.

## Acceptance criteria (verified headless)
- G1: bake a synthetic audio track → timeline at 144 fps (>=90), correct frame count, all elementals finite/complete.
- G2: correctness — a planted tone's dominant-frequency elemental recovers the tone; onsets fire on transients.
- G3: transpose-to-regime — feed `timeline.to_features(i)` into the GivensProjector/field: it advects (the baked
  score drives a scene with no real-time extractor).
- G4: signal-agnostic — bake a synthetic multi-channel MWD-like signal (rotating tool: RPM tone + noise + shock
  events at low sr); the per-channel dominant-frequency elemental recovers the planted RPM; shocks show as onsets.

## Plan
`cocycle/signal.py`: `Elementals`/`bake_signal(x, sr, fps, nfft, frange)` → `Timeline`; `bake_audio`,
`bake_multichannel`. `verify_bake.py` asserts G1–G4. Real MWD ingest = a thin format adapter (LAS/DLIS/binary →
channel arrays) added when a sample file is provided — the bake core is already format-agnostic.
