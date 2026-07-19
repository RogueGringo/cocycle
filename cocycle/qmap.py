"""Generates the COCYCLE Swing Quality Map -- the launch-monitor-free coaching instrument built on the one
result that survived every skeptical pass: the cocycle (gluing obstruction). It shows the golfer's glued
canonical swing with their own per-phase variation band, the per-shot obstruction ranking (who doesn't glue),
and a per-shot x per-phase heatmap of WHERE each swing breaks from the template -- in units of the golfer's
own sigma. Ground-truth mishits (from the FSX PDF, if present) are marked to show the cocycle validating.
Self-contained HTML (canvas-rendered, theme-aware); open locally or publish as an Artifact. numpy only."""
from __future__ import annotations
import json, os, sys
import numpy as np
from .sheaf import build_sheaf, quality_map

SESSION = r"C:\Users\JT-DEV1\Desktop\development\30SwingSession_Individual"


def _ds(a, n):
    g = np.linspace(0, 1, n); x = np.linspace(0, 1, len(a))
    return np.interp(g, x, a)


def _ds_rows(M, n):
    return np.array([_ds(r, n) for r in M])


def build_html(session=SESSION):
    sh = build_sheaf(session, N=256)
    qm = quality_map(sh, "omega")
    ids = qm["ids"]; resid = qm["residual"]; P = qm["P"]
    order = [int(i) for i in np.argsort(-resid)]                 # worst obstruction first

    mishits = set()                                             # ground-truth catastrophic mishits (optional)
    pdf = os.path.join(session, "FSX_Shotresults.pdf")
    if os.path.exists(pdf):
        try:
            from .launch import parse_fsx_pdf, join_session
            iy, ln, Y = parse_fsx_pdf(pdf)
            ic, _, Yj = join_session(ids, np.zeros((len(ids), 1)), iy, Y)
            bs = Yj[:, ln.index("ball_speed_mph")]
            mishits = {ic[i] for i in range(len(ic)) if bs[i] < 0.70 * np.median(bs)}
        except Exception:
            pass
    flagged = {ids[i] for i in order[:4]}                       # cocycle top-4 obstruction

    tempo = P[0] / max(P[1] - P[0], 1e-6)
    data = {
        "ids": ids, "order": order,
        "residual": [float(x) for x in resid],
        "mishits": sorted(mishits), "flagged": sorted(flagged),
        "mean": [float(x) for x in _ds(qm["mean"], 140)],
        "band": [float(x) for x in _ds(qm["tightness"], 140)],
        "heat": [[round(float(v), 3) for v in r] for r in np.clip(_ds_rows(qm["z"][order], 72), 0, 4)],
        "heat_ids": [ids[i] for i in order],
        "worst_phase": [float(qm["worst_phase"][i]) for i in order],
        "P_top": float(P[0]), "P_impact": float(P[1]),
        "n": len(ids), "tempo": float(tempo),
        "tightness_mean": float(qm["tightness"].mean()),
    }
    return _TEMPLATE.replace("/*__DATA__*/", json.dumps(data))


_TEMPLATE = r"""<title>COCYCLE — Swing Quality Map</title>
<style>
:root{--bg:#eef1f4;--surface:#f8fafb;--inset:#e6eaee;--line:#d3d9df;--ink:#12161b;--dim:#5c6773;
  --accent:#0a9c8c;--warn:#b8791a;--bad:#c1443a;--good:#1a8a4a;
  --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;--sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;}
@media(prefers-color-scheme:dark){:root{--bg:#0a0d12;--surface:#12161d;--inset:#0c1015;--line:#232b35;
  --ink:#e7edf3;--dim:#8b97a5;--accent:#35e0d0;--warn:#f0b64a;--bad:#ff6a5a;--good:#4cd07d;}}
:root[data-theme=light]{--bg:#eef1f4;--surface:#f8fafb;--inset:#e6eaee;--line:#d3d9df;--ink:#12161b;--dim:#5c6773;--accent:#0a9c8c;--warn:#b8791a;--bad:#c1443a;--good:#1a8a4a;}
:root[data-theme=dark]{--bg:#0a0d12;--surface:#12161d;--inset:#0c1015;--line:#232b35;--ink:#e7edf3;--dim:#8b97a5;--accent:#35e0d0;--warn:#f0b64a;--bad:#ff6a5a;--good:#4cd07d;}
*{box-sizing:border-box}
.wrap{max-width:1080px;margin:0 auto;padding:22px 20px 60px;color:var(--ink);background:var(--bg);font-family:var(--sans);min-height:100vh}
header.top{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:18px}
.brand{font-weight:700;letter-spacing:.14em;font-size:15px}.brand b{color:var(--accent)}
.sub{color:var(--dim);font-size:12.5px;font-family:var(--mono)}.spacer{flex:1}
.themetgl{font-family:var(--mono);font-size:11.5px;color:var(--dim);background:var(--inset);border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
.stats{display:flex;gap:22px;flex-wrap:wrap;margin:2px 0 20px;font-family:var(--mono)}
.stat .v{font-size:22px;font-weight:600;color:var(--ink)}.stat .v.bad{color:var(--bad)}
.stat .k{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--dim);margin-top:2px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:16px 16px 14px;margin-bottom:16px}
.card h2{margin:0 0 3px;font-size:11px;letter-spacing:.15em;text-transform:uppercase;color:var(--dim);font-weight:600}
.card p.cap{margin:0 0 12px;font-size:12px;color:var(--dim);font-family:var(--mono)}
canvas{width:100%;display:block}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--dim);margin-top:8px}
.legend span{display:flex;align-items:center;gap:6px}.sw{width:11px;height:11px;border-radius:3px}
</style>

<div class="wrap">
  <header class="top">
    <div><div class="brand"><b>COCYCLE</b> · SWING QUALITY MAP</div>
      <div class="sub">the gluing obstruction — a launch-monitor-free consistency map</div></div>
    <div class="spacer"></div><button class="themetgl" id="themebtn">theme</button>
  </header>
  <div class="stats" id="stats"></div>

  <div class="card"><h2>Your canonical swing</h2>
    <p class="cap">glued mean angular velocity across the phase arc, with your own ±1σ variation band</p>
    <canvas id="curve" height="300"></canvas></div>

  <div class="card"><h2>Obstruction ranking — who doesn't glue</h2>
    <p class="cap">each swing's cocycle residual (distance from your glued template). red = launch-monitor-confirmed mishit</p>
    <canvas id="bars" height="470"></canvas></div>

  <div class="card"><h2>Where each swing breaks</h2>
    <p class="cap">per-phase obstruction in units of your own σ; rows sorted worst-first, columns = swing phase</p>
    <canvas id="heat" height="470"></canvas>
    <div class="legend">
      <span><i class="sw" style="background:var(--bad)"></i>ground-truth mishit</span>
      <span><i class="sw" style="background:var(--warn)"></i>cocycle-flagged (top 4)</span>
      <span><i class="sw" style="background:var(--accent)"></i>within template</span></div>
  </div>
</div>

<script>
const D = /*__DATA__*/;
const $ = s=>document.querySelector(s);
function cs(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function flagColor(id){return D.mishits.includes(id)?cs('--bad'):D.flagged.includes(id)?cs('--warn'):cs('--accent');}
function fit(cv){const r=cv.getBoundingClientRect(),d=window.devicePixelRatio||1;cv.width=r.width*d;cv.height=cv.height*d;
  const x=cv.getContext('2d');x.setTransform(d,0,0,d,0,0);return[x,r.width,cv.height/d];}
function heatColor(t){t=Math.max(0,Math.min(1,t/4));const stops=[[10,16,24],[53,120,160],[70,200,140],[240,182,74],[224,70,60]];
  const p=t*(stops.length-1),i=Math.floor(p),f=p-i,a=stops[i],b=stops[Math.min(i+1,stops.length-1)];
  return`rgb(${a.map((v,k)=>Math.round(v+(b[k]-v)*f)).join(',')})`;}

function stats(){
  const nm=D.mishits.length;
  $("#stats").innerHTML=[
    ['n',D.n,'swings'],['tempo',D.tempo.toFixed(2)+':1','backswing / downswing'],
    ['flagged',D.flagged.length,'cocycle obstruction'],['mishits',nm,'launch-confirmed']
  ].map(([k,v,d])=>`<div class="stat"><div class="v${k=='mishits'&&nm?' bad':''}">${v}</div><div class="k">${k} · ${d}</div></div>`).join('');
}
function phaseShade(x,W,H,x0){ // shade backswing / downswing / follow + markers
  const xt=x0+D.P_top*W, xi=x0+D.P_impact*W;
  x.fillStyle=cs('--inset');x.globalAlpha=0.5;x.fillRect(x0,0,D.P_top*W,H);x.globalAlpha=1;
  x.strokeStyle=cs('--line');x.lineWidth=1;
  [[xt,'top'],[xi,'impact']].forEach(([xx,lb])=>{x.beginPath();x.moveTo(xx,6);x.lineTo(xx,H-16);x.stroke();
    x.fillStyle=cs('--dim');x.font='11px '+cs('--mono');x.fillText(lb,xx+4,14);});
  x.fillStyle=cs('--dim');x.font='11px '+cs('--mono');
  x.fillText('backswing',x0+4,H-4);x.fillText('downswing',xt+4,H-4);x.fillText('follow',xi+4,H-4);
}
function drawCurve(){
  const[x,W,H]=fit($("#curve")),pad=8,x0=34,w=W-x0-10,h=H-28;
  const m=D.mean,b=D.band,n=m.length,lo=[],hi=[];for(let i=0;i<n;i++){lo.push(m[i]-b[i]);hi.push(m[i]+b[i]);}
  const all=lo.concat(hi),mn=Math.min(...all),mx=Math.max(...all),Y=v=>pad+(1-(v-mn)/(mx-mn+1e-9))*h;
  const X=i=>x0+i/(n-1)*w;
  phaseShade(x,w,H,x0);
  x.beginPath();for(let i=0;i<n;i++)x[i?'lineTo':'moveTo'](X(i),Y(hi[i]));
  for(let i=n-1;i>=0;i--)x.lineTo(X(i),Y(lo[i]));x.closePath();x.fillStyle=cs('--accent');x.globalAlpha=0.16;x.fill();x.globalAlpha=1;
  x.beginPath();for(let i=0;i<n;i++)x[i?'lineTo':'moveTo'](X(i),Y(m[i]));x.strokeStyle=cs('--accent');x.lineWidth=2;x.stroke();
  x.strokeStyle=cs('--line');x.beginPath();x.moveTo(x0,Y(0));x.lineTo(x0+w,Y(0));x.stroke();
  x.fillStyle=cs('--dim');x.font='10px '+cs('--mono');x.fillText('ω',6,pad+8);x.fillText('0',6,Y(0)+3);
}
function drawBars(){
  const[x,W,H]=fit($("#bars")),x0=52,w=W-x0-46,n=D.order.length,bh=(H-10)/n-2;
  const mx=Math.max(...D.residual);
  D.order.forEach((oi,r)=>{const id=D.ids[oi],v=D.residual[oi],y=6+r*(H-10)/n;
    x.fillStyle=flagColor(id);x.fillRect(x0,y,v/mx*w,bh);
    x.fillStyle=cs('--dim');x.font='10px '+cs('--mono');x.textAlign='right';x.fillText(id.replace('Shot','#'),x0-6,y+bh-1);
    x.textAlign='left';x.fillStyle=cs('--ink');x.fillText(v.toFixed(2),x0+v/mx*w+5,y+bh-1);});
  x.textAlign='left';
}
function drawHeat(){
  const[x,W,H]=fit($("#heat")),x0=52,w=W-x0-10,rows=D.heat.length,cols=D.heat[0].length,bh=(H-24)/rows,bw=w/cols;
  D.heat.forEach((row,r)=>{const id=D.heat_ids[r],y=6+r*bh;
    for(let c=0;c<cols;c++){x.fillStyle=heatColor(row[c]);x.fillRect(x0+c*bw,y,bw+0.5,bh-0.5);}
    x.fillStyle=flagColor(id);x.font='10px '+cs('--mono');x.textAlign='right';x.fillText(id.replace('Shot','#'),x0-6,y+bh-2);});
  x.textAlign='left';
  const xt=x0+D.P_top*w,xi=x0+D.P_impact*w,H2=6+rows*bh;
  x.strokeStyle=cs('--ink');x.globalAlpha=0.5;x.lineWidth=1;
  [xt,xi].forEach(xx=>{x.beginPath();x.moveTo(xx,6);x.lineTo(xx,H2);x.stroke();});x.globalAlpha=1;
  x.fillStyle=cs('--dim');x.font='11px '+cs('--mono');
  x.fillText('backswing',x0+4,H2+14);x.fillText('downswing',xt+4,H2+14);x.fillText('follow',xi+4,H2+14);
}
function drawAll(){stats();drawCurve();drawBars();drawHeat();}
$("#themebtn").onclick=()=>{const r=document.documentElement,cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  r.setAttribute('data-theme',cur==='dark'?'light':'dark');drawAll();};
window.addEventListener('resize',drawAll);drawAll();
</script>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cocycle_qmap.html"
    open(out, "w", encoding="utf-8").write(build_html())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
