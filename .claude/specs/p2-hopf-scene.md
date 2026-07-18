# P2 Spec — Hopf fibration scene

status: approved · date: 2026-07-18 · note: a NEW visual object (S³/4-D), sibling to the rank-7 field — not an upgrade to it.

## Objective
A second scene: the Hopf fibration S³→S². Base points on S² each lift to a great circle in S³; rotate S³ by
an audio-driven SO(4) (6-plane Givens) rotation, stereographically project S³→R³, perspective R³→2D, and draw
each fiber as a glowing closed loop — the interlocking, never-intersecting circles.

## Constraints
- Correctness: the generated per-fiber point sets must be genuine Hopf fibers — the Hopf map is CONSTANT along
  each fiber (that's the geometric acceptance test, not eyeballing pixels).
- Efficiency: upload the S³ fiber points ONCE; the vertex shader rotates (R[16] uniform), stereographic-projects,
  and perspective-projects each frame — only the 16-float rotation changes per frame.
- torch-free (numpy SO(4) Givens); GL 3.3 (attributes, no SSBO needed).

## Acceptance criteria (verified headless / numpy)
- E1: renders offscreen non-black with many lit line pixels (the circles draw).
- E2: **fibers are genuine Hopf fibers** — for every fiber, the Hopf map of its points is constant
  (max per-fiber S²-image variance < 1e-4); every point lies on S³ (‖·‖≈1).
- E3: audio-reactive — the SO(4) rotation advects; loud vs quiet frames differ.

## Plan
`cocycle/hopf.py`: `hopf_fibers(B, M)` (Fibonacci-sphere base points → z1,z2 preimage → e^{iθ} sweep → S³
points [B,M,4] + per-fiber hue); `HopfRenderer` (upload once; vertex shader rotate→stereographic→perspective;
draw B LINE_LOOPs); `ProsodyGivens4` (numpy 6-plane SO(4) driven by prosody). `verify_hopf.py` asserts E1–E3;
`run_hopf.py` the live viewer.
