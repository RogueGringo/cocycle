# P1 Spec — GPU-Accelerated Givens Projector via Prosody

status: approved · owner: research→engineering handoff · date: 2026-07-18

## Objective
Replace the static numpy `_rot` in `Projector` with the differentiable, orthogonal-by-construction
`GivensRotor` (`cocycle/givens.py`), driven live by prosody tensors, device-agnostic (GPU when available).

## Architectural constraints
- **Strict orthogonality by construction.** Givens parameterization only — no matrix inversion, no Cayley
  transform, no Stiefel retraction. Orthogonality must hold for *any* angle values, never re-projected.
- **Parallel structure.** The C(7,2)=21 planes are organized into commutative rounds (a 1-factorization of
  K₇: 7 rounds of 3 disjoint planes) so the operation is parallel-friendly; the per-frame cost is dominated by
  the single batched `X @ Rᵀ` matmul, which runs on the RTX 5070 when torch sees CUDA.
- **Dependency isolation.** torch lives only in the GPU-rotor path (`givens.py`, `projector_gpu.py`). The core
  engine orchestration (`engine.py`, numpy field/features/ledmap) stays torch-free; `import cocycle` must not
  require torch.

## Implementation tasks
1. **Env:** install torch into the cocycle venv (cu128 for the 5070; CPU acceptable for logic verification).
2. **Wire:** `GivensProjector` component importing `GivensRotor`, output-compatible with `Projector`
   (`ctx['proj'] = {xy, near, w, band}`) — a drop-in for the render path.
3. **Signal mapping:** route `FeatureExtractor` prosody (7 bands → the 21 planes; f0 → base spin; centroid →
   w-plane bias; flux → transient jolts; formants → specific planes) to modulate the 21 angle *velocities*;
   integrate continuously (smooth, gradient-ready).
4. **Render integration:** swap `Projector` → `GivensProjector` in the headless/verify path; project via the
   live `GivensRotor` tensor output.

## Acceptance criteria (verified headless)
- A1: composition initializes headless with torch active (device reported: cuda or cpu).
- A2: audio advects the 21 angles in real time — angles change across frames AND respond to the audio
  (silent-vs-loud angular velocity differs).
- A3: **continuous orthogonality** over the live stream — every frame `max|RRᵀ−I| < 1e-6` and `det(R) ∈ [1−1e-4, 1+1e-4]`.
- A4: the full pipeline still yields valid LED output (LEDMapper non-degenerate) with the GivensProjector in place.

## Plan (how)
- `projector_gpu.py`: hold a `GivensRotor(7)` on the chosen device; keep a 21-vector of angles; each frame
  compute per-plane velocity from prosody, integrate, `rotor.set_angles(...)`, `R = rotor.matrix()`; rotate the
  numpy particle batch through torch (`X→device @ Rᵀ`), then the same single-stage perspective as `Projector`
  (DEPTH_W, FOCAL, scale) so the visual is unchanged; return numpy `xy/near/w/band` for LED/render/spatial.
- Verification: `verify_p1.py` runs the composition with `GivensProjector` for ~150 synthetic-audio frames,
  asserting A1–A4 and printing the device + orthogonality residual band. Exits non-zero on any breach.
