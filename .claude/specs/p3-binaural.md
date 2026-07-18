# P3 Spec — Binaural spatializer bound to the geometry

status: approved · date: 2026-07-18 · honest scope: ITD/ILD gives strong left-right orbit; front-back/elevation need HRTF (spaudiopy/sofar) — the follow-on enhancement.

## Objective
Replace the stereo-pan Spatializer stub with a real **binaural** renderer: split the audio into the 7 bands
(same bands as FeatureExtractor), place each band as a spatial voice on a ring whose rotation is driven by the
SAME prosody that spins the visuals, and render each voice with interaural time (ITD) + level (ILD) differences.
As the geometry rotates, the 7 band-voices physically orbit the listener in lockstep — the "brain-tricking" cue.

## Constraints
- Physically-grounded ITD (Woodworth-style, ~0.7 ms max) + equal-power/head-shadow ILD; fractional-delay per ear.
- Bound to the visual drive: the ring spin Φ integrates from the same f0/prosody the projector uses → audio and
  light move together.
- torch-free, numpy only. Real-time block API for the live path + an offline buffer renderer for verification.

## Acceptance criteria (verified headless / numpy)
- F1: ITD/ILD correctness — source hard-right ⇒ LEFT ear delayed AND right ear louder; hard-left ⇒ mirror; center ⇒ symmetric.
- F2: coherent orbit — over a full spin, the running L−R balance oscillates through both sides (multiple zero-crossings).
- F3: valid finite stereo output; render a listenable binaural WAV (orbiting moan) as the artifact.

## Plan
`cocycle/binaural.py`: `itd_ild(az, sr)`; `binauralize(audio, sr, spin_fn)` (STFT band-split → per-band azimuth
= 2πb/7 + Φ(t) → ITD fractional-delay + ILD gains → overlap-add L/R). `verify_spatial.py` asserts F1–F3 and
writes `orbit_binaural.wav`. A `Binaural` component wraps the block API for the live engine (follow-on wiring).
Enhancement path (named, not built): HRTF binaural via `sofar` SOFA HRIRs / HOA via `spaudiopy` for elevation + externalization.
