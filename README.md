# COCYCLE

An audio-reactive engine that renders a **signal on a rank-7 topology field** — a lean, Unreal-style
component SDK. One `Composition` runs at full fidelity on the authoring rig and, later, at LED
resolution on a Raspberry Pi; only the sink components (LED / spatial audio) change.

The rank-7 object is the exact Čech complex from the DeltaSheaf research thread (5 vertices, 10
edges, 10 faces), which proved *task-inert* for LLM evaluation — here its **21-plane projection
richness** is repurposed as a pure audio-reactive representation space.

## Pipeline (the subderived components)

```
AudioSource → FeatureExtractor → TopologyField → Projector → ┬─ LEDMapper   (light)
 (synth/mic/file) (prosody-aware) (rank-7, N pts) (7D→2D, 21  └─ Spatializer (sound)
                                                    planes)
```

- **AudioSource** — built-in voice-like "moan" synth, or WAV/FLAC via `soundfile`, or live mic via `sounddevice`.
- **FeatureExtractor** — prosody-aware: 7 log bands (→ the 7 axes) plus f0, spectral centroid, flux/onset, rms, rolloff, formant peaks, arousal/valence.
- **TopologyField** — N particles in R⁷ on a manifold shell, advected by a divergence-free solenoidal Fourier flow whose gains are driven by the seven bands. Pure numpy (verifiable); GPU-shader advection to millions is the P1 swap behind the same interface.
- **Projector** — composes the C(7,2)=21 Givens rotations, single-stage perspective 7D→2D; the 4th rotated coord is hue, so the palette turns as the object rotates through w.
- **LEDMapper** — bins the projection onto a physical LED layout (default 32×32 sim); the output grid is exactly what a real driver pushes on the Pi.
- **Spatializer** — the auditory co-projection: per-band sources panned by azimuth (P0 stereo; HRTF/ambisonic at P2).

## Run

```bash
python -m venv .venv && .venv/Scripts/activate        # (Windows) or source .venv/bin/activate
pip install -r requirements.txt

python run_headless.py          # verify the whole pipeline, no window (writes led_frame_*.ppm)
python run_live.py              # real-time viewer on the rig — built-in moan synth
python run_live.py --audio mic  # live input
python run_live.py --audio file --file track.wav
```

`run_headless.py` is the instrument: it asserts every stage is live, finite, and changing, and does
an offscreen GPU render smoke-test. It exits non-zero on any failure.

## Roadmap

- **P0 — vertical slice** ✅ audio → field → projection → simulated LED + spatial, verified on the rig.
- **P1** — GPU-shader advection (transform feedback → millions of particles); card-2 runs a live embedding model so the 7-D coordinates are real representation data; motion-trail compositor.
- **P2** — Raspberry Pi runtime + real LED driver (`rpi-rgb-led-matrix` / WS2812 SPI / DMX) + speaker/HRTF output; the transposition to your physical layout.
- **P3** — authoring editor: live-edit compositions on the rig, package → deploy to Pi.

## Layout

```
cocycle/
  engine.py              # Component / Context / Composition
  components/
    audio.py features.py field.py projector.py ledmap.py spatial.py render_gl.py
run_headless.py          # verification harness
run_live.py              # real-time viewer (moderngl-window)
```
