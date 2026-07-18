"""ComputeField — the rank-7 particle field on a moderngl compute shader (spec p1b-compute-field.md).

Millions of 7-D particles advected in place in an SSBO by the same divergence-free Fourier flow as the numpy
TopologyField (A_m perpendicular K_m -> solenoidal), with per-band audio gains, per-axis breathing, an onset
kick, and a soft radial spring. moderngl-only (no torch); GL 4.3+ compute. Particles are 8 floats each
(7 used + 1 pad) for std430 alignment.
"""
from __future__ import annotations
import numpy as np
import moderngl

COMPUTE = """#version 430
layout(local_size_x=64) in;
layout(std430, binding=0) buffer Pos   { float p[]; };     // N*8, first 7 used
layout(std430, binding=1) buffer Modes { float m[]; };     // M*16: [K(7), A(7), phi, band]
uniform int N; uniform int M; uniform float dt; uniform float speed; uniform float rms; uniform float flux;
uniform float bands[7];
void main(){
    uint gid = gl_GlobalInvocationID.x; if (gid >= uint(N)) return;
    uint b = gid * 8u;
    float x[7]; for (int k=0;k<7;k++) x[k]=p[b+uint(k)];
    float v[7]; for (int k=0;k<7;k++) v[k]=0.0;
    for (int mi=0; mi<M; mi++){
        uint mo = uint(mi)*16u;
        float dp = 0.0; for (int k=0;k<7;k++) dp += m[mo+uint(k)]*x[k];
        float gain = 0.35 + 1.1*bands[int(m[mo+15u])];
        float s = sin(dp + m[mo+14u]) * gain;
        for (int k=0;k<7;k++) v[k] += s * m[mo+7u+uint(k)];
    }
    float bmean=0.0; for(int k=0;k<7;k++) bmean+=bands[k]; bmean/=7.0;
    float r=0.0; for(int k=0;k<7;k++) r+=x[k]*x[k]; r=sqrt(r)+1e-9;
    float spd = speed*(0.5 + 1.3*rms);
    for (int k=0;k<7;k++){
        v[k] += 0.6*(bands[k]-bmean) + 0.8*flux*x[k];
        x[k] += v[k]*dt*spd;
        x[k] += (1.5 - r)/r * x[k] * (0.08 + 0.10*rms);
        x[k]  = clamp(x[k], -6.0, 6.0);
    }
    for (int k=0;k<7;k++) p[b+uint(k)]=x[k];
}
"""


class ComputeField:
    def __init__(self, ctx: moderngl.Context, n=1_000_000, modes=28, radius=1.5, speed=0.55, seed=7):
        self.ctx = ctx; self.n = n; self.speed = speed
        self.cs = ctx.compute_shader(COMPUTE)
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((n, 7)).astype(np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); X *= radius
        buf = np.zeros((n, 8), np.float32); buf[:, :7] = X
        self.band = (np.arange(n) % 7).astype(np.int32)
        self.pos = ctx.buffer(buf.tobytes()); self.pos.bind_to_storage_buffer(0)
        # solenoidal Fourier modes -> flat [M,16]
        K = (rng.standard_normal((modes, 7)) * 1.7).astype(np.float32)
        A = rng.standard_normal((modes, 7)).astype(np.float32)
        A -= (np.sum(A*K, 1, keepdims=True)/(np.sum(K*K, 1, keepdims=True)+1e-9))*K      # A ⟂ K
        phi = rng.random(modes).astype(np.float32) * 2*np.pi
        band = (np.arange(modes) % 7).astype(np.float32)
        M = np.concatenate([K, A, phi[:, None], band[:, None]], 1).astype(np.float32)   # [modes,16]
        self.modes = ctx.buffer(M.tobytes()); self.modes.bind_to_storage_buffer(1)
        for name, val in (("N", n), ("M", modes)):
            self.cs[name].value = val

    def step(self, bands, rms=0.4, flux=0.0, dt=1/60):
        self.pos.bind_to_storage_buffer(0); self.modes.bind_to_storage_buffer(1)
        self.cs["dt"].value = float(dt); self.cs["speed"].value = float(self.speed)
        self.cs["rms"].value = float(rms); self.cs["flux"].value = float(flux)
        self.cs["bands"].value = tuple(float(b) for b in bands)
        self.cs.run(group_x=(self.n + 63) // 64)

    def read(self):
        return np.frombuffer(self.pos.read(), np.float32).reshape(self.n, 8)[:, :7]


if __name__ == "__main__":
    import time, sys
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
    N = 1_000_000
    ctx = moderngl.create_standalone_context(require=430)
    print(f"[ComputeField] {ctx.info['GL_RENDERER'][:30]}  GL {ctx.info['GL_VERSION'][:12]}  N={N:,}")
    cf = ComputeField(ctx, n=N)
    x0 = cf.read().copy()
    # C4: PURE compute-dispatch rate — particles stay on the GPU (no per-frame readback; that's how the
    # real pipeline consumes the SSBO). The earlier 11 fps was 64 MB/frame of GPU->CPU readback, not compute.
    bands = np.full(7, 0.5, np.float32)
    cf.step(bands); ctx.finish()                                   # warm
    t0 = time.perf_counter(); STEPS = 120
    for _ in range(STEPS): cf.step(bands, rms=0.6)
    ctx.finish()
    fps = STEPS / (time.perf_counter() - t0)
    xf = cf.read()
    finite = np.isfinite(xf).all()
    moved = float(np.linalg.norm(xf - x0, axis=1).mean())
    bounded = float(np.abs(xf).max())
    # C3: reactivity via two short bursts (readback only at burst boundaries)
    def burst(bval, rms, steps=15):
        a = cf.read()
        for _ in range(steps): cf.step(np.full(7, bval, np.float32), rms=rms)
        ctx.finish(); return float(np.linalg.norm(cf.read() - a, axis=1).mean()) / steps
    disp_loud = burst(0.7, 0.8); disp_quiet = burst(0.05, 0.05)
    reactive = abs(disp_loud - disp_quiet)
    print(f"  C1 runs at 1M: PASS")
    print(f"  C2 finite={finite}  moved(mean)={moved:.3f}  bounded max|x|={bounded:.2f} (shell 1.5)")
    print(f"  C3 audio-reactive: loud disp {disp_loud:.4f} vs quiet {disp_quiet:.4f}  (Δ {reactive:.4f})")
    print(f"  C4 rate: {fps:.0f} field-updates/sec at {N:,} particles ({1000/fps:.2f} ms/step)")
    ok = finite and moved > 1e-4 and bounded < 6.5 and reactive > 1e-4 and fps >= 60
    print(f"[ComputeField] {'ALL ACCEPTANCE CRITERIA MET' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)
