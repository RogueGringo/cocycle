# Context: COCYCLE — audio-reactive rank-7 field engine

## Core principle
An audio signal rendered as a **signal on a rank-7 topology field**, via a lean component SDK. The engine
descends from a *closed* research program (`../ig-primon-t1/`, `../deltasheaf-v02/`, `../procrustes-gate/`)
whose one durable structural fact drives the design: **isolated geometry is task-inert; the exploitable object
is the orthogonal transformation itself.** So the projector core is a Givens-parameterized **orthogonal**
transform (`cocycle/givens.py`) — orthogonal by construction, differentiable, never re-orthogonalized.
Do NOT reach for compressive/bounded reparameterizations (tanh-squash, learned projections onto a smaller
space) in the transform path: compression starves gradient resolution — that was empirically falsified.

## Architecture (do not restructure without cause)
`engine.py`: `Component` / `Context` (per-frame blackboard) / `Composition` (ordered component list, one shared
Context). Pipeline of subderived components, each writing `ctx.data[self.name]`:
`AudioSource → FeatureExtractor(prosody) → TopologyField(rank-7 particles) → Projector(7D→2D, 21 planes) →
LEDMapper + Spatializer`.

## Essential commands (use these exactly)
| Operation | Command |
| :-- | :-- |
| **Verify the whole pipeline (the instrument)** | `.venv/Scripts/python.exe run_headless.py` |
| **Live viewer (on the rig only)** | `.venv/Scripts/python.exe run_live.py` (`--audio mic\|file --file x.wav`) |
| **Givens rotor self-test** | `python -m cocycle.givens` (needs torch env) |
| **Deps** | `.venv/Scripts/python.exe -m pip install -r requirements.txt` |

## Non-obvious ground truth (read before editing)
- **`run_headless.py` is the instrument, not a demo.** It asserts every stage is live/finite/changing and
  exits non-zero on failure. It also does an **offscreen** GL render smoke-test (standalone context → FBO →
  pixel readback) because the live window **cannot be seen from an agent/build environment**. Never "verify"
  by launching `run_live.py` here — validate headless; the window is for the human on the rig.
- **The core is numpy on purpose** (field/projector/features) so the math is testable without a GL context.
  moderngl is the renderer. **torch is a separate, optional dependency** used only by `cocycle/givens.py`
  (the differentiable/GPU-rotor path) — the `.venv` is numpy-lean and does NOT carry torch; the cu128 torch
  build lives in `../ig-primon-t1/.venv` and is protected — do not modify it.
- **`GivensRotor` gotcha:** a norm-preserving loss (`‖R·x‖²`) has **zero gradient** w.r.t. the angles — that
  is rotation-invariance, not a bug. Train it with an angle-*dependent* objective (a linear functional of R,
  a target-matching loss). It builds each plane rotation additively from constant basis buffers (no in-place
  writes) so autograd stays clean.
- **7-D field → C(7,2) = 21 rotation planes.** The Projector integrates 21 audio-driven angles; the 4th
  rotated coordinate is hue (palette turns as the object rotates through w). The star polytope (the 5-cell)
  is the Čech complex from the research: 5 vertices, 10 edges, 10 faces.
- **The field is numpy (~60k particles).** Millions-of-particles GPU-shader advection (transform feedback)
  is P1, behind the same `TopologyField` interface — swap the impl, not the contract.

## Status & roadmap
- **P0 vertical slice: DONE, verified on the RTX 5070** (all stages pass headless; offscreen GL renders).
- **P1** — port `TopologyField` to a GPU shader (millions of particles); make `Projector` use `GivensRotor`
  on-GPU; optional card-2 live embedding so the 7-D coordinates are real representation data.
- **P2** — Raspberry-Pi runtime + real LED driver (`rpi-rgb-led-matrix` / WS2812 SPI / DMX) + HRTF/ambisonic
  spatializer. The Pi runs the same Composition at LED resolution (low pixel count → real-time).
- **P3** — authoring editor: live-edit Compositions on the rig, package → deploy to Pi.
- **P4 offline signal bake: DONE, verified (verify_bake.py G1–G5).** `cocycle/signal.py` bakes ANY time-series
  file → a renderer-agnostic `Timeline` of 16 elementals at 144 fps (floor 90), global per-track normalization.
  Signal-agnostic: `bake_signal(x, sr)` core; audio (pitch) and multi-channel MWD/vibration (RPM) prove out on
  the same code — dominant-freq via FFT-autocorr + YIN CMND (valley-bottom + parabolic; naive argmax is
  lag-biased, do not use it). `Timeline.to_features(i)` feeds existing scenes unchanged (= transpose to any
  regime). Super-user control panel is GENERATED from the `BakeParams` schema (`cocycle/panel.py` →
  `cocycle_panel.html`, gitignored); its config round-trips via `python -m cocycle.signal --config x.json`.
  Real MWD ingest = a thin LAS/DLIS/binary → channel-array adapter, added when a sample file lands.

## Spec-Driven Development
Work in phases: **Spec → Plan → Tasks → Implement → Verify.** Specs are immutable decision logs in
`.claude/specs/`; plans in `.claude/plans/`. Do not mutate source before the spec for a feature is approved.
Every Implement step must end with `run_headless.py` green.

## Style & commits
- No inline comments explaining basic Python. Comments carry the *why* / the non-obvious, matching existing density.
- Conventional commits (e.g. `feat(projector): drive GivensRotor from prosody`). Commit/push only on explicit request.
