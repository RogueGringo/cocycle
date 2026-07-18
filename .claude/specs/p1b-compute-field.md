# P1b Spec — Compute-shader particle field (millions of particles)

status: approved · date: 2026-07-18 · verified: moderngl compute PASS on RTX 5070 (GL 4.3)

## Objective
Move `TopologyField` off numpy (~60k CPU particles) onto a moderngl **compute shader** so the RTX 5070
advects **millions** of 7-D particles per frame. The solenoidal Fourier flow + audio-band gains port to GLSL;
the audio-driven Givens rotation continues to drive the *projection* (P1), the compute pass drives the *field*.

## Constraints
- **GL 4.3+ compute** (verified available). Particles in an SSBO (`bind_to_storage_buffer`), advected in place.
- Same physics as the numpy `TopologyField`: divergence-free Fourier modes (A_m ⟂ K_m), per-band gains,
  per-axis breathing, onset kick, soft radial spring — so the look is continuous with P0/P1.
- **Dependency isolation:** compute path is moderngl-only (no torch needed for the field).

## Acceptance criteria (verified headless, standalone context)
- C1: compute field runs at **N = 1,000,000** without error on the 5070.
- C2: particles **advect** (mean per-step displacement > 0), stay **finite**, and stay **bounded** (‖x‖ near the shell, |x|<6).
- C3: audio-reactive — displacement responds to the band gains (loud ≠ quiet).
- C4: real-time headroom — ≥ 60 field-updates/sec at 1M particles (report measured rate).

## Plan
`cocycle/compute_field.py`: a `ComputeField` holding the SSBO + compute program + a modes-SSBO; `step(bands,
rms, flux, dt)` sets uniforms and dispatches `ceil(N/64)` groups; `read()` returns numpy for verification/LED.
Render integration (projector reading the SSBO directly, or a vertex shader consuming it) is the follow-on
ticket — this slice proves the compute physics at scale. Self-test in `__main__` asserts C1–C4.
