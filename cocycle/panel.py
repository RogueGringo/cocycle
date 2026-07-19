"""Generates the COCYCLE super-user control panel (spec p4). The panel is BUILT from the introspectable
BakeParams schema (params_schema) + ELEMENTALS, so it never drifts from the engine: add a field to BakeParams
and a control appears. A super-user independently selects elementals, operates every param/hyperparam, picks a
regime, and the panel emits a config that round-trips through `python -m cocycle.signal --config ...`.
Output is a self-contained HTML fragment (no external assets) — open locally or publish as an Artifact."""
from __future__ import annotations
import json, sys
from .signal import params_schema, ELEMENTALS

ELEMENTAL_GROUPS = {
    "level": ["rms", "peak"], "bands (rank-7)": [f"band{b}" for b in range(7)],
    "spectral": ["centroid", "rolloff", "flux", "flatness"], "temporal": ["zcr"],
    "periodicity": ["domfreq"], "onset": ["onset"],
}
REGIMES = [("field", "rank-7 field"), ("hopf", "Hopf S3"), ("binaural", "binaural orbit"), ("led", "LED matrix")]
PRESETS = {
    "audio - music": {"signal": {"type": "audio"}, "params": {"fps": 144, "nfft": 2048, "n_bands": 7,
        "fmin": 50, "fmax": 0, "dom_fmin": 80, "dom_fmax": 1200, "norm_lo": 2, "norm_hi": 98}},
    "voice": {"signal": {"type": "audio"}, "params": {"fps": 144, "nfft": 2048, "n_bands": 7,
        "fmin": 60, "fmax": 8000, "dom_fmin": 60, "dom_fmax": 400, "norm_lo": 2, "norm_hi": 98}},
    "MWD - drilling": {"signal": {"type": "multichannel"}, "params": {"fps": 144, "nfft": 2048, "n_bands": 7,
        "fmin": 1, "fmax": 200, "dom_fmin": 1, "dom_fmax": 50, "norm_lo": 2, "norm_hi": 98}},
    "vibration - machine": {"signal": {"type": "multichannel"}, "params": {"fps": 144, "nfft": 4096, "n_bands": 7,
        "fmin": 2, "fmax": 500, "dom_fmin": 5, "dom_fmax": 500, "norm_lo": 1, "norm_hi": 99}},
}


def build_html():
    data = json.dumps({"params": params_schema(), "groups": ELEMENTAL_GROUPS,
                       "regimes": REGIMES, "presets": PRESETS, "all": list(ELEMENTALS)})
    return _TEMPLATE.replace("/*__DATA__*/", data)


_TEMPLATE = r"""<title>COCYCLE — bake control</title>
<style>
:root{
  --bg:#eef1f4; --surface:#f8fafb; --inset:#e6eaee; --line:#d3d9df; --ink:#12161b; --dim:#5c6773;
  --accent:#0a9c8c; --param:#0a9c8c; --hyper:#b8791a; --good:#1a8a4a; --shadow:rgba(20,30,40,.10);
  --mono:ui-monospace,"Cascadia Code","SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0a0d12; --surface:#12161d; --inset:#0c1015; --line:#232b35; --ink:#e7edf3; --dim:#8b97a5;
  --accent:#35e0d0; --param:#35e0d0; --hyper:#f0b64a; --good:#4cd07d; --shadow:rgba(0,0,0,.4);
}}
:root[data-theme=light]{
  --bg:#eef1f4; --surface:#f8fafb; --inset:#e6eaee; --line:#d3d9df; --ink:#12161b; --dim:#5c6773;
  --accent:#0a9c8c; --param:#0a9c8c; --hyper:#b8791a; --good:#1a8a4a; --shadow:rgba(20,30,40,.10);
}
:root[data-theme=dark]{
  --bg:#0a0d12; --surface:#12161d; --inset:#0c1015; --line:#232b35; --ink:#e7edf3; --dim:#8b97a5;
  --accent:#35e0d0; --param:#35e0d0; --hyper:#f0b64a; --good:#4cd07d; --shadow:rgba(0,0,0,.4);
}
*{box-sizing:border-box}
.wrap{max-width:1180px;margin:0 auto;padding:22px 20px 60px;color:var(--ink);background:var(--bg);
  font-family:var(--sans);min-height:100vh}
header.top{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;border-bottom:1px solid var(--line);
  padding-bottom:14px;margin-bottom:20px}
.brand{font-weight:700;letter-spacing:.14em;font-size:15px}
.brand b{color:var(--accent)}
.sub{color:var(--dim);font-size:12.5px;font-family:var(--mono)}
.spacer{flex:1}
.themetgl{font-family:var(--mono);font-size:11.5px;color:var(--dim);background:var(--inset);
  border:1px solid var(--line);border-radius:6px;padding:5px 10px;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1.35fr;gap:18px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:16px 16px 18px;
  box-shadow:0 1px 3px var(--shadow)}
.card h2{margin:0 0 12px;font-size:11px;letter-spacing:.16em;text-transform:uppercase;color:var(--dim);
  font-weight:600}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.chip{font-family:var(--mono);font-size:11.5px;padding:6px 11px;border-radius:20px;border:1px solid var(--line);
  background:var(--inset);color:var(--ink);cursor:pointer;transition:.12s}
.chip:hover{border-color:var(--accent)}
.chip.on{background:var(--accent);color:var(--bg);border-color:var(--accent)}
.field{display:flex;align-items:center;gap:9px;margin:9px 0;font-family:var(--mono);font-size:12px}
.field label{width:74px;color:var(--dim)}
.field input[type=text]{flex:1;background:var(--inset);border:1px solid var(--line);border-radius:6px;
  color:var(--ink);padding:6px 8px;font-family:var(--mono);font-size:12px}
.slab{margin:14px 0}
.slab .row{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:12px}
.slab .name{width:88px;display:flex;align-items:center;gap:6px}
.dot{width:7px;height:7px;border-radius:50%}
.dot.param{background:var(--param)} .dot.hyper{background:var(--hyper)}
.slab input[type=range]{flex:1;accent-color:var(--accent);height:3px}
.slab .num{width:74px;text-align:right}
.slab .num input{width:70px;background:var(--inset);border:1px solid var(--line);border-radius:5px;
  color:var(--ink);padding:4px 6px;font-family:var(--mono);font-size:12px;text-align:right}
.hint{font-family:var(--mono);font-size:11px;color:var(--dim);margin:2px 0 0 98px}
.legend{display:flex;gap:16px;font-family:var(--mono);font-size:11px;color:var(--dim);margin:2px 0 12px}
.legend span{display:flex;align-items:center;gap:6px}
.egroup{margin:10px 0}
.egroup .glabel{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;color:var(--dim);
  text-transform:uppercase;margin-bottom:6px}
.toggles{display:flex;flex-wrap:wrap;gap:7px}
.tg{font-family:var(--mono);font-size:11px;padding:5px 9px;border-radius:6px;border:1px solid var(--line);
  background:var(--inset);color:var(--dim);cursor:pointer;user-select:none}
.tg.on{color:var(--bg);background:var(--param);border-color:var(--param)}
.regimes{display:flex;flex-wrap:wrap;gap:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;width:fit-content}
.seg{font-family:var(--mono);font-size:11.5px;padding:7px 13px;cursor:pointer;color:var(--dim);
  border-right:1px solid var(--line)}
.seg:last-child{border-right:none}
.seg.on{background:var(--accent);color:var(--bg)}
.out{margin-top:18px}
.out pre{background:var(--inset);border:1px solid var(--line);border-radius:9px;padding:13px 14px;
  overflow-x:auto;font-family:var(--mono);font-size:11.5px;line-height:1.55;color:var(--ink);margin:8px 0 0}
.out .cmd{color:var(--good)}
.outhead{display:flex;align-items:center;gap:10px}
.copy{font-family:var(--mono);font-size:11px;color:var(--dim);background:var(--inset);border:1px solid var(--line);
  border-radius:6px;padding:5px 11px;cursor:pointer}
.copy:hover{border-color:var(--accent);color:var(--ink)}
.readout{font-family:var(--mono);font-size:12px;color:var(--dim);margin:10px 0 0;line-height:1.7}
.readout b{color:var(--ink)} .readout .k{color:var(--accent)}
</style>

<div class="wrap">
  <header class="top">
    <div>
      <div class="brand"><b>COCYCLE</b> · BAKE CONTROL</div>
      <div class="sub">signal &rarr; elementals &rarr; transpose to any regime</div>
    </div>
    <div class="spacer"></div>
    <button class="themetgl" id="themebtn">theme</button>
  </header>

  <div class="grid">
    <div>
      <div class="card">
        <h2>Signal source &amp; presets</h2>
        <div class="presets" id="presets"></div>
        <div class="field"><label>type</label>
          <select id="sigtype" style="flex:1;background:var(--inset);border:1px solid var(--line);border-radius:6px;color:var(--ink);padding:6px 8px;font-family:var(--mono);font-size:12px">
            <option value="audio">audio (1-ch)</option><option value="multichannel">multichannel (MWD)</option>
          </select></div>
        <div class="field"><label>path</label><input type="text" id="sigpath" value="track.wav" spellcheck="false"></div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Elementals — independent select</h2>
        <div id="elementals"></div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Regime (transpose target)</h2>
        <div class="regimes" id="regimes"></div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Parameters &amp; hyperparameters</h2>
        <div class="legend">
          <span><i class="dot param"></i>param — what is observed</span>
          <span><i class="dot hyper"></i>hyperparam — how the kernel is shaped</span>
        </div>
        <div id="sliders"></div>
      </div>

      <div class="card out">
        <div class="outhead"><h2 style="margin:0">Generated bake config</h2><div class="spacer" style="flex:1"></div>
          <button class="copy" id="copyjson">copy json</button><button class="copy" id="copycli">copy cli</button></div>
        <div class="readout" id="readout"></div>
        <pre id="cfgout"></pre>
        <pre id="cliout"></pre>
      </div>
    </div>
  </div>
</div>

<script>
const D = /*__DATA__*/;
const state = {sig:{type:"audio",path:"track.wav"}, params:{}, elementals:new Set(), regime:"field", preset:null};
D.params.forEach(p=> state.params[p.name]=p.value);
D.all.forEach(n=> state.elementals.add(n));           // start with everything on

const $ = s=>document.querySelector(s);
function esc(s){return String(s).replace(/</g,"&lt;")}

// theme
(function(){const b=$("#themebtn");b.onclick=()=>{const r=document.documentElement;
  const cur=r.getAttribute("data-theme")|| (matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme", cur==="dark"?"light":"dark"); render();};})();

function renderPresets(){
  const el=$("#presets"); el.innerHTML="";
  Object.keys(D.presets).forEach(name=>{
    const c=document.createElement("div"); c.className="chip"+(state.preset===name?" on":""); c.textContent=name;
    c.onclick=()=>applyPreset(name); el.appendChild(c);
  });
}
function applyPreset(name){
  const p=D.presets[name]; state.preset=name;
  state.sig.type=p.signal.type; Object.assign(state.params, p.params);
  $("#sigtype").value=state.sig.type;
  renderPresets(); renderSliders(); update();
}

function renderSliders(){
  const el=$("#sliders"); el.innerHTML="";
  D.params.forEach(p=>{
    const wrap=document.createElement("div"); wrap.className="slab";
    const isInt = p.type.includes("int");
    wrap.innerHTML =
      '<div class="row"><div class="name"><i class="dot '+p.kind+'"></i>'+p.name+'</div>'+
      '<input type="range" min="'+p.min+'" max="'+p.max+'" step="'+p.step+'" value="'+state.params[p.name]+'">'+
      '<div class="num"><input type="text" value="'+state.params[p.name]+'"></div></div>';
    const rng=wrap.querySelector('input[type=range]'), num=wrap.querySelector('.num input');
    const set=v=>{v=isInt?Math.round(+v):+v; if(isNaN(v))return; state.params[p.name]=v; rng.value=v; num.value=v; update();};
    rng.oninput=e=>set(e.target.value); num.onchange=e=>set(e.target.value);
    el.appendChild(wrap);
  });
}

function renderElementals(){
  const el=$("#elementals"); el.innerHTML="";
  Object.entries(D.groups).forEach(([g,names])=>{
    const box=document.createElement("div"); box.className="egroup";
    box.innerHTML='<div class="glabel">'+g+'</div>';
    const tg=document.createElement("div"); tg.className="toggles";
    names.forEach(n=>{
      const t=document.createElement("div"); t.className="tg"+(state.elementals.has(n)?" on":""); t.textContent=n;
      t.onclick=()=>{ state.elementals.has(n)?state.elementals.delete(n):state.elementals.add(n);
        t.classList.toggle("on"); update(); };
      tg.appendChild(t);
    });
    box.appendChild(tg); el.appendChild(box);
  });
}

function renderRegimes(){
  const el=$("#regimes"); el.innerHTML="";
  D.regimes.forEach(([k,label])=>{
    const s=document.createElement("div"); s.className="seg"+(state.regime===k?" on":""); s.textContent=label;
    s.onclick=()=>{state.regime=k; renderRegimes(); update();}; el.appendChild(s);
  });
}

function cfg(){
  const ordered = D.all.filter(n=>state.elementals.has(n));
  return {signal:{type:state.sig.type, path:$("#sigpath").value}, params:{...state.params},
          elementals:ordered, regime:state.regime};
}
function update(){
  state.sig.type=$("#sigtype").value; state.sig.path=$("#sigpath").value;
  const c=cfg();
  $("#cfgout").textContent = JSON.stringify(c, null, 2);
  $("#cliout").innerHTML = '<span class="cmd"># save the json above as cocycle_bake.json, then:\n'+
    'python -m cocycle.signal --config cocycle_bake.json --out timeline.npz</span>';
  const dl=c.params.dom_fmin, dh=c.params.dom_fmax;
  $("#readout").innerHTML =
    '<span class="k">'+c.elementals.length+'</span>/'+D.all.length+' elementals &nbsp;·&nbsp; target '+
    '<b>'+c.params.fps+' fps</b> &nbsp;·&nbsp; regime <b>'+c.regime+'</b><br>'+
    'periodicity band <b>'+dl+'&ndash;'+dh+' Hz</b> = <b>'+Math.round(dl*60)+'&ndash;'+Math.round(dh*60)+' RPM</b>'+
    ' &nbsp;('+ (c.signal.type==="multichannel"?"drilling / vibration":"pitch / audio") +')';
}
function render(){renderPresets();renderSliders();renderElementals();renderRegimes();update();}

$("#sigtype").onchange=update; $("#sigpath").oninput=update;
$("#copyjson").onclick=()=>navigator.clipboard&&navigator.clipboard.writeText($("#cfgout").textContent);
$("#copycli").onclick=()=>navigator.clipboard&&navigator.clipboard.writeText($("#cliout").textContent);
render();
</script>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "cocycle_panel.html"
    open(out, "w", encoding="utf-8").write(build_html())
    print(f"wrote {out} ({len(D_names())} params, {len(ELEMENTALS)} elementals)")


def D_names():
    return params_schema()


if __name__ == "__main__":
    main()
