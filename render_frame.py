"""Offscreen still renders of the COCYCLE scenes (no window). Accumulates motion-trail frames into an FBO
for glow, then saves PNGs. Proof images since the live window can't be shown from the build env."""
import sys, numpy as np
from PIL import Image

W = 1200


def save(fbo, path):
    px = np.frombuffer(fbo.read(components=3), np.uint8).reshape(W, W, 3)[::-1]   # GL origin bottom-left
    Image.fromarray(px).save(path); print(f"  wrote {path}")


def fade_prog(ctx, a):
    import moderngl
    p = ctx.program(vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}",
                    fragment_shader=f"#version 330\nout vec4 c;void main(){{c=vec4(0.02,0.02,0.06,{a});}}")
    vao = ctx.vertex_array(p, [(ctx.buffer(np.array([-1,-1,3,-1,-1,3],np.float32).tobytes()), "2f", "p")])
    return vao


def render_hopf():
    import moderngl
    from cocycle.hopf import HopfRenderer, ProsodyGivens4
    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.framebuffer([ctx.texture((W, W), 3)]); fbo.use(); ctx.clear(0.02, 0.02, 0.06)
    hr = HopfRenderer(ctx, 800, 320); pg = ProsodyGivens4(); fq = fade_prog(ctx, 0.12)
    feats = {"bands7": np.array([.6,.5,.7,.4,.5,.3,.4], np.float32), "f0": 0.45, "flux": 0.0}
    for i in range(90):
        pg.update(feats, 1/60)
        ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        fq.render(mode=moderngl.TRIANGLES, vertices=3)
        hr.draw(pg.R_flat(), scale=0.55, focal=3.0)
    save(fbo, "render_hopf.png")


def render_field():
    import moderngl
    from cocycle.compute_field import ComputeField
    from cocycle.gpu_render import GPURenderer, ProsodyGivens
    ctx = moderngl.create_standalone_context(require=430)
    fbo = ctx.framebuffer([ctx.texture((W, W), 3)]); fbo.use(); ctx.clear(0.02, 0.02, 0.06)
    cf = ComputeField(ctx, n=1_500_000); gr = GPURenderer(ctx); pg = ProsodyGivens(); fq = fade_prog(ctx, 0.14)
    feats = {"bands7": np.array([.6,.4,.6,.5,.4,.5,.3], np.float32), "f0": 0.4, "centroid": 0.5, "flux": 0.0, "rms": 0.6}
    for i in range(80):
        cf.step(feats["bands7"], rms=feats["rms"], flux=0.0); pg.update(feats, 1/60)
        ctx.enable(moderngl.BLEND); ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        fq.render(mode=moderngl.TRIANGLES, vertices=3)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        gr.draw(cf.pos, pg.R_flat(), 1_500_000)
    save(fbo, "render_field.png")


if __name__ == "__main__":
    render_hopf()
    render_field()
    print("done")
