"""Render the RECONSTRUCTED clubhead arc through space (anchored at the ball), from the drift-free orientation.
Unlike render_swing.py (the abstract S^3 quaternion path), this is the actual swing motion in 3D. Offscreen
-> PNG. Also prints swing-plane stats and a gate on planarity (a real swing arc is nearly planar)."""
import sys, numpy as np
from PIL import Image
from cocycle.reconstruct import reconstruct_arc, swing_plane, plane_inclination, world_up
from cocycle.swing import load_shot, detect_impact, segment_swing

W = 1000
SHOT = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual\GolfML1.2_2024-03-21T14-51-24_Shot1"


def rot(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    return (np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])).astype("f4")


def main():
    import moderngl
    sh = load_shot(SHOT); ti, _ = detect_impact(sh); seg = segment_swing(sh, ti)
    arc, phase = reconstruct_arc(sh, seg)
    up = world_up(sh); pl = swing_plane(arc); inc = plane_inclination(arc, up)
    print(f"[RECON] arc {len(arc)} pts, anchored at ball |arc[impact]|~{np.linalg.norm(arc[np.argmin(abs(phase-(seg['impact']-seg['takeaway'])/(seg['follow']-seg['takeaway'])))]):.2f} "
          f"| planarity {pl['planarity']:.3f} | plane inclination {inc:.0f} deg from vertical")

    P = arc - arc.mean(0); P = (P / (np.percentile(np.abs(P), 99) + 1e-9)).astype("f4")
    col = np.stack([phase, 0.3 + 0.4 * np.sin(phase * np.pi), 1 - phase], 1).astype("f4")
    ii = int(np.argmin(np.abs(phase - (seg["impact"] - seg["takeaway"]) / (seg["follow"] - seg["takeaway"]))))
    col[max(0, ii - 1):ii + 2] = [1, 1, 1]                                # impact / ball = white
    verts = np.hstack([P, col]).astype("f4")

    ctx = moderngl.create_standalone_context(require=330)
    fbo = ctx.framebuffer([ctx.texture((W, W), 3)]); fbo.use(); ctx.clear(0.02, 0.02, 0.06)
    prog = ctx.program(
        vertex_shader="#version 330\nuniform mat3 R;in vec3 ip;in vec3 ic;out vec3 v;"
        "void main(){vec3 p=R*ip;float z=2.6+p.z;gl_Position=vec4(p.xy*(1.7/z),0,1);v=ic;gl_PointSize=6.0;}",
        fragment_shader="#version 330\nin vec3 v;out vec4 c;void main(){c=vec4(v,0.9);}")
    vbo = ctx.buffer(verts.tobytes()); vao = ctx.vertex_array(prog, [(vbo, "3f 3f", "ip", "ic")])
    ball = ctx.buffer(np.hstack([-P[ii], [1, 1, 1]]).astype("f4").tobytes())
    bvao = ctx.vertex_array(prog, [(ball, "3f 3f", "ip", "ic")])
    fq = ctx.program(vertex_shader="#version 330\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}",
                     fragment_shader="#version 330\nout vec4 c;void main(){c=vec4(0.02,0.02,0.06,0.12);}")
    fvao = ctx.vertex_array(fq, [(ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], "f4").tobytes()), "2f", "p")])
    ctx.line_width = 2.5
    for f in range(150):
        prog["R"].write(rot(0.42, f * 0.026).tobytes())
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA); fvao.render(moderngl.TRIANGLES, vertices=3)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        vao.render(moderngl.LINE_STRIP); bvao.render(moderngl.POINTS)
    px = np.frombuffer(fbo.read(components=3), np.uint8).reshape(W, W, 3)[::-1]
    Image.fromarray(px).save("render_reconstruct.png")
    print("wrote render_reconstruct.png")


if __name__ == "__main__":
    main()
