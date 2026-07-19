"""Render the REAL golf swing as geometry: the shaft-orientation quaternion is a point on S^3, so one swing
is a curve on the 3-sphere. Stereographic-project that curve to R^3 and draw it, colored by swing phase
(backswing -> downswing -> impact flash -> follow-through), rotating with additive glow. Offscreen -> PNG,
since the live window can't be shown from the build env. 'spectral geometry is literally everything.'"""
import sys, numpy as np
from PIL import Image
from cocycle.swing import load_shot, detect_impact, segment_swing

W = 1000
SHOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual\GolfML1.2_2024-03-21T14-51-24_Shot1"


def stereo(q):                                   # S^3 (unit 4D) -> R^3, project from the +w pole
    w = np.clip(q[:, 0:1], -0.999, 0.999)
    return q[:, 1:] / (1.0 - w)


def rot(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return (Ry @ Rx).astype("f4")


def main():
    import moderngl
    shot = load_shot(SHOT)
    t_imp, _ = detect_impact(shot); seg = segment_swing(shot, t_imp)
    tq, q = shot["SFQ"]; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    for i in range(1, len(q)):                    # hemisphere-continuous (kill quaternion sign flips)
        if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
    win = (tq >= seg["takeaway"]) & (tq <= seg["follow"]); qs = q[win]; ts = tq[win]
    P = stereo(qs); P = P - np.median(P, 0)
    P = np.clip(P / (np.percentile(np.abs(P), 98) + 1e-9), -1.6, 1.6).astype("f4")
    ph = np.clip((ts - seg["takeaway"]) / (seg["follow"] - seg["takeaway"] + 1e-9), 0, 1)
    col = np.stack([ph, 0.3 + 0.4 * np.sin(ph * np.pi), 1 - ph], 1).astype("f4")   # cool->warm
    ii = int(np.argmin(np.abs(ts - seg["impact"]))); col[max(0, ii - 1):ii + 2] = [1, 1, 1]  # impact flash
    verts = np.hstack([P, col]).astype("f4")

    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.framebuffer([ctx.texture((W, W), 3)]); fbo.use(); ctx.clear(0.02, 0.02, 0.06)
    prog = ctx.program(
        vertex_shader="#version 330\nuniform mat3 R;in vec3 ip;in vec3 ic;out vec3 v;"
        "void main(){vec3 p=R*ip;float z=2.6+p.z;gl_Position=vec4(p.xy*(1.9/z),0,1);v=ic;}",
        fragment_shader="#version 330\nin vec3 v;out vec4 c;void main(){c=vec4(v,0.85);}")
    vbo = ctx.buffer(verts.tobytes()); vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "ip", "ic")])
    fq = ctx.program(vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}",
                     fragment_shader="#version 330\nout vec4 c;void main(){c=vec4(0.02,0.02,0.06,0.10);}")
    fvao = ctx.vertex_array(fq, [(ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes()), "2f", "p")])
    ctx.line_width = 2.0
    for f in range(150):
        prog["R"].write(rot(0.5, f * 0.028).tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA); fvao.render(moderngl.TRIANGLES, vertices=3)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE); vao.render(moderngl.LINE_STRIP)
    px = np.frombuffer(fbo.read(components=3), np.uint8).reshape(W, W, 3)[::-1]
    Image.fromarray(px).save("render_swing.png")
    print(f"wrote render_swing.png  ({len(P)} quaternion samples on S^3; impact flash at phase {ph[ii]:.2f})")


if __name__ == "__main__":
    main()
