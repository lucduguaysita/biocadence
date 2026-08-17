"""Build a self-contained HTML "model inspector" from a trained model.

It Monte-Carlo samples the model and plans a few sequences, then renders six
panels (all inline SVG, no libraries, light/dark aware):
  - headline stat tiles
  - inter-key interval distribution with the think-pause fence
  - a typing "rhythm strip" showing bursts and pauses over time
  - dwell-time distribution
  - P(pause) by context
  - mouse movement-time vs distance (with the Fitts-style fit)
  - sampled mouse trajectories

Usage:
  from biocadence.model import Model
  from biocadence.viz import write_report
  write_report(Model.load("me.model.json"), "report.html")
"""

import json
import math

import numpy as np

from .schema import token_to_char


def _pair_keys(ops):
    """Return per-keystroke (token, t_down, dwell) in press order."""
    from collections import defaultdict, deque
    ups = defaultdict(deque)
    for o in ops:
        if o["op"] == "key_up":
            ups[o["token"]].append(o["t"])
    for tok in ups:
        ups[tok] = deque(sorted(ups[tok]))
    out = []
    for o in ops:
        if o["op"] != "key_down":
            continue
        tok, t = o["token"], o["t"]
        dq = ups[tok]
        while dq and dq[0] < t - 1e-9:
            dq.popleft()
        dwell = (dq.popleft() - t) if dq else None
        out.append((tok, t, dwell))
    out.sort(key=lambda r: r[1])
    return out


def _disp_char(token):
    ch = token_to_char(token)
    if ch == " ":
        return "_"
    if ch == "\n":
        return "/"
    if ch is None:
        return "<"        # backspace
    return ch


def compute_data(model, seed=0):
    from .planner import Planner
    fence_ms = model.p["flight"]["fence"] * 1000.0

    # A varied paragraph so every context (word/clause/sentence) occurs.
    para = ("The quick brown fox jumps over the lazy dog. Pack my box with "
            "five dozen liquor jugs, then rest. How vexingly quick daft "
            "zebras jump; amazingly few discotheques provide jukeboxes today.")
    ops = Planner(model, seed=seed, cursor=(0, 0)).build([{"type": "type", "text": para}])
    keys = _pair_keys(ops)
    downs = [t for _, t, _ in keys]
    dwells = [d * 1000.0 for _, _, d in keys if d is not None]
    gaps = [(downs[i] - downs[i - 1]) * 1000.0 for i in range(1, len(downs))]

    dur_min = (downs[-1] - downs[0]) / 60.0 if len(downs) > 1 else 1e-9
    nchars = sum(1 for tok, _, _ in keys if len(tok) == 1 or tok == "Key.space")
    wpm = (nchars / 5.0) / dur_min

    def hist(vals, lo, hi, n):
        counts, edges = np.histogram(np.clip(vals, lo, hi), bins=n, range=(lo, hi))
        return {"counts": counts.tolist(), "edges": [round(e, 1) for e in edges]}

    gmax = float(min(np.percentile(gaps, 99), 650)) if gaps else 600
    interval_hist = hist(gaps, 0, gmax, 30)
    dwell_hist = hist(dwells, 0, float(min(np.percentile(dwells, 99), 240)), 24)

    # Short sentence for the rhythm strip (a sentence boundary -> a visible pause)
    strip_text = "type like a human. pauses make it real."
    sops = Planner(model, seed=seed + 1, cursor=(0, 0)).build(
        [{"type": "type", "text": strip_text}])
    skeys = _pair_keys(sops)
    st0 = skeys[0][1]
    marks = []
    prev = None
    for tok, t, d in skeys:
        gap = None if prev is None else (t - prev) * 1000.0
        marks.append({
            "c": _disp_char(tok),
            "t": (t - st0) * 1000.0,
            "dwell": (d or 0.02) * 1000.0,
            "gap": gap,
            "pause": bool(gap is not None and gap > fence_ms),
        })
        prev = t
    strip_total = (skeys[-1][1] - st0) * 1000.0 + marks[-1]["dwell"]

    # Fitts-style movement-time vs distance
    rng = np.random.default_rng(seed + 2)
    fpts = []
    for D in np.geomspace(25, 1400, 44):
        for _ in range(2):
            fpts.append({"d": float(D), "t": model.sample_move_time(rng, float(D)) * 1000.0})
    mt = model.p["mouse"]["move_time"]

    # A few sampled trajectories to the same target (shows curvature/overshoot)
    traj = []
    start, target = (200, 760), (1480, 250)
    for k in range(2):
        tops = Planner(model, seed=seed + 10 + k, cursor=start).build(
            [{"type": "move", "to": list(target)}])
        traj.append([{"x": o["x"], "y": o["y"]} for o in tops if o["op"] == "mouse_move"])

    return {
        "meta": model.p["meta"],
        "fence_ms": round(fence_ms, 1),
        "kpis": {
            "wpm": round(wpm, 1),
            "median_gap": round(float(np.median(gaps)), 0) if gaps else 0,
            "median_dwell": round(float(np.median(dwells)), 0) if dwells else 0,
            "click_ms": round(math.exp(model.p["click"]["duration"]["mu"]) * 1000, 0),
            "overshoot_pct": round(model.p["mouse"]["overshoot_prob"] * 100, 0),
        },
        "interval_hist": interval_hist,
        "dwell_hist": dwell_hist,
        "pause_prob": {k: round(v, 3) for k, v in model.p["pause"]["prob"].items()},
        "strip": {"marks": marks, "total": strip_total, "fence_ms": round(fence_ms, 1)},
        "fitts": {"points": fpts, "a_ms": mt["a"] * 1000.0, "b_ms": mt["b"] * 1000.0,
                  "maxD": 1450},
        "traj": {"paths": traj, "start": {"x": start[0], "y": start[1]},
                 "target": {"x": target[0], "y": target[1]}, "w": 1600, "h": 900},
    }


def build_html(model, seed=0, name=None, date=None):
    import datetime
    data = compute_data(model, seed=seed)
    data["name"] = name or "Input style model"
    data["date"] = date or datetime.date.today().isoformat()
    return _TEMPLATE.replace("/*__DATA__*/{}", json.dumps(data))


def write_report(model, path, seed=0, name=None, date=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(model, seed=seed, name=name, date=date))
    return path


_TEMPLATE = r"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Input style model</title>
<style>
  :root{
    --plane:#0d0d0d; --surface-1:#1a1a19; --text-primary:#ffffff;
    --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
    --series-1:#3987e5; --series-2:#d95926; --border:rgba(255,255,255,0.10);
  }
  :root[data-theme="light"]{
    --plane:#f9f9f7; --surface-1:#fcfcfb; --text-primary:#0b0b0b;
    --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
    --series-1:#2a78d6; --series-2:#eb6834; --border:rgba(11,11,11,0.10);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--text-primary);
    font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
  .wrap{max-width:1120px;margin:0 auto;padding:28px 20px 64px}
  header{display:flex;align-items:baseline;justify-content:space-between;gap:16px;flex-wrap:wrap}
  h1{font-size:22px;margin:0 0 2px}
  .sub{color:var(--text-secondary);font-size:13.5px;margin:0}
  button#theme{background:var(--surface-1);color:var(--text-secondary);
    border:1px solid var(--border);border-radius:8px;padding:7px 12px;font-size:13px;cursor:pointer}
  .kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0 8px}
  .tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:14px 16px}
  .tile .v{font-size:26px;font-weight:650;letter-spacing:-.01em}
  .tile .l{font-size:12px;color:var(--text-secondary);margin-top:3px}
  .tile .u{font-size:14px;color:var(--muted);font-weight:500;margin-left:3px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px}
  .card{background:var(--surface-1);border:1px solid var(--border);border-radius:14px;padding:16px 16px 10px}
  .card h2{font-size:14.5px;margin:0 0 2px}
  .card p{font-size:12.5px;color:var(--text-secondary);margin:0 0 8px}
  .card.wide{grid-column:1 / -1}
  svg{width:100%;height:auto;display:block}
  .ax{fill:var(--muted);font-size:11px}
  .axtitle{fill:var(--text-secondary);font-size:11.5px}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);margin:2px 2px 8px}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:middle}
  #tip{position:fixed;pointer-events:none;opacity:0;transform:translate(-50%,-120%);
    background:var(--text-primary);color:var(--plane);font-size:12px;padding:5px 8px;
    border-radius:7px;white-space:nowrap;transition:opacity .08s;z-index:9;font-variant-numeric:tabular-nums}
  @media(max-width:760px){.kpis{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1 id="rptTitle">Input style model</h1>
      <p class="sub" id="rptSub">Sampled from your fitted keystroke, click and mouse-motion distributions.</p>
    </div>
    <button id="theme">Light</button>
  </header>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="card wide">
      <h2>Typing rhythm</h2>
      <p>Each mark is a keystroke; bar height is how long the key was held. Wide gaps are think-time; orange marks follow a pause past the fence.</p>
      <div id="strip"></div>
    </div>
    <div class="card">
      <h2>Inter-key interval</h2>
      <p>Press-to-press gaps. The cluster is fluent typing; the tail past the dashed fence is think-time.</p>
      <div id="intervals"></div>
    </div>
    <div class="card">
      <h2>Key hold (dwell)</h2>
      <p>How long each key stays down.</p>
      <div id="dwell"></div>
    </div>
    <div class="card">
      <h2>Pause likelihood by context</h2>
      <p>Chance of inserting a think-pause, by what precedes the key.</p>
      <div id="pause"></div>
    </div>
    <div class="card">
      <h2>Mouse move time vs distance</h2>
      <p>Longer moves take longer, sub-linearly. The line is the fitted a + b*sqrt(distance).</p>
      <div id="fitts"></div>
    </div>
    <div class="card wide">
      <h2>Sampled mouse paths</h2>
      <p>Two draws from the same start to the same target: curved approach, tremor, and occasional overshoot.</p>
      <div class="legend"><span><i style="background:var(--series-1)"></i>path A</span><span><i style="background:var(--series-2)"></i>path B</span></div>
      <div id="traj"></div>
    </div>
  </div>
  <p class="sub" style="margin-top:22px">Model summary: <span id="metatext"></span></p>
</div>
<div id="tip"></div>
<script>
const DATA = /*__DATA__*/{};
const NS="http://www.w3.org/2000/svg";
const tip=document.getElementById("tip");
function el(t,a){const e=document.createElementNS(NS,t);for(const k in(a||{}))e.setAttribute(k,a[k]);return e;}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function showTip(ev,html){tip.innerHTML=html;tip.style.left=ev.clientX+"px";tip.style.top=ev.clientY+"px";tip.style.opacity=1;}
function hideTip(){tip.style.opacity=0;}
function hook(node,html){node.addEventListener("mousemove",e=>showTip(e,html));node.addEventListener("mouseleave",hideTip);}

function kpis(){
  document.getElementById("rptTitle").textContent = DATA.name || "Input style model";
  document.getElementById("rptSub").textContent =
    "Keystroke, click and mouse-motion profile. Generated " + (DATA.date || "");
  const k=DATA.kpis, box=document.getElementById("kpis");
  const rows=[["WPM",k.wpm,""],["Inter-key",k.median_gap,"ms"],["Key hold",k.median_dwell,"ms"],
    ["Click",k.click_ms,"ms"],["Overshoot",k.overshoot_pct,"%"]];
  for(const [l,v,u] of rows){
    const d=document.createElement("div");d.className="tile";
    d.innerHTML=`<div class="v">${v}<span class="u">${u}</span></div><div class="l">${l}</div>`;
    box.appendChild(d);
  }
  document.getElementById("metatext").textContent=
    `${DATA.meta.n_keys} keys, ${DATA.meta.n_clicks} clicks, ${DATA.meta.n_move_segments} move segments. Pause fence ${DATA.fence_ms} ms.`;
}

function histChart(id,H,unit,fenceMs){
  const W=520,Hh=210,pl=42,pr=20,pt=12,pb=34;
  const s=el("svg",{viewBox:`0 0 ${W} ${Hh}`}),iw=W-pl-pr,ih=Hh-pt-pb;
  const max=Math.max(...H.counts,1),n=H.counts.length;
  const x0=H.edges[0],x1=H.edges[H.edges.length-1];
  const X=v=>pl+(v-x0)/(x1-x0)*iw, Y=v=>pt+ih-v/max*ih;
  for(let g=0;g<=4;g++){const yy=pt+ih-g/4*ih;
    s.appendChild(el("line",{x1:pl,y1:yy,x2:W-pr,y2:yy,stroke:css("--grid"),"stroke-width":1}));
    const tx=el("text",{x:pl-6,y:yy+3,"text-anchor":"end",class:"ax"});tx.textContent=Math.round(max*g/4);s.appendChild(tx);}
  const bw=iw/n;
  for(let i=0;i<n;i++){
    const h=H.counts[i]/max*ih, x=X(H.edges[i])+1, y=pt+ih-h;
    const r=el("rect",{x:x,y:y,width:Math.max(bw-2,1),height:Math.max(h,0),rx:3,fill:css("--series-1")});
    hook(r,`${H.edges[i]}-${H.edges[i+1]} ${unit}: <b>${H.counts[i]}</b>`);
    s.appendChild(r);
  }
  for(let g=0;g<=5;g++){const xv=x0+(x1-x0)*g/5;const an=g==0?"start":(g==5?"end":"middle");const tx=el("text",{x:X(xv),y:Hh-14,"text-anchor":an,class:"ax"});tx.textContent=Math.round(xv);s.appendChild(tx);}
  const at=el("text",{x:pl+iw/2,y:Hh-1,"text-anchor":"middle",class:"axtitle"});at.textContent=unit;s.appendChild(at);
  if(fenceMs&&fenceMs<x1){const fx=X(fenceMs);
    s.appendChild(el("line",{x1:fx,y1:pt,x2:fx,y2:pt+ih,stroke:css("--series-2"),"stroke-width":2,"stroke-dasharray":"4 3"}));
    const ft=el("text",{x:fx+4,y:pt+10,class:"ax",fill:css("--series-2")});ft.textContent="fence";s.appendChild(ft);}
  document.getElementById(id).innerHTML="";document.getElementById(id).appendChild(s);
}

function strip(){
  const S=DATA.strip,W=1040,Hh=150,pl=8,pr=8,pt=16,pb=26;
  const s=el("svg",{viewBox:`0 0 ${W} ${Hh}`}),iw=W-pl-pr,ih=Hh-pt-pb;
  const T=S.total*1.02, X=t=>pl+t/T*iw;
  const dmax=Math.max(...S.marks.map(m=>m.dwell),1);
  s.appendChild(el("line",{x1:pl,y1:pt+ih,x2:W-pr,y2:pt+ih,stroke:css("--axis"),"stroke-width":1}));
  let prev=null;
  for(const m of S.marks){
    const x=X(m.t), h=Math.max(m.dwell/dmax*ih,4), y=pt+ih-h;
    if(m.pause&&prev!=null){s.appendChild(el("rect",{x:prev,y:pt,width:x-prev,height:ih,fill:css("--series-2"),opacity:0.10}));}
    const col=m.pause?css("--series-2"):css("--series-1");
    const r=el("rect",{x:x,y:y,width:3.4,height:h,rx:1.5,fill:col});
    hook(r,`'${m.c}'  hold ${Math.round(m.dwell)} ms${m.gap!=null?`, gap ${Math.round(m.gap)} ms`:""}`);
    s.appendChild(r);
    const tc=el("text",{x:x+1.7,y:pt+ih+13,"text-anchor":"middle",class:"ax"});tc.textContent=m.c;s.appendChild(tc);
    prev=x+3.4;
  }
  const at=el("text",{x:pl,y:12,class:"axtitle"});at.textContent="time (left to right), one column per keystroke";s.appendChild(at);
  document.getElementById("strip").innerHTML="";document.getElementById("strip").appendChild(s);
}

function pauseBars(){
  const P=DATA.pause_prob,order=["mid","word","clause","sentence"];
  const rows=order.filter(k=>k in P);
  const W=520,Hh=210,pl=64,pr=16,pt=10,pb=28;
  const s=el("svg",{viewBox:`0 0 ${W} ${Hh}`}),iw=W-pl-pr,ih=Hh-pt-pb;
  const max=Math.max(...rows.map(k=>P[k]),0.05)*1.18;
  const bh=ih/rows.length*0.62, step=ih/rows.length;
  for(let g=0;g<=4;g++){const xx=pl+iw*g/4;
    s.appendChild(el("line",{x1:xx,y1:pt,x2:xx,y2:pt+ih,stroke:css("--grid"),"stroke-width":1}));
    const tx=el("text",{x:xx,y:Hh-12,"text-anchor":"middle",class:"ax"});tx.textContent=Math.round(max*100*g/4)+"%";s.appendChild(tx);}
  rows.forEach((k,i)=>{
    const y=pt+i*step+(step-bh)/2, w=P[k]/max*iw;
    const r=el("rect",{x:pl,y:y,width:Math.max(w,1),height:bh,rx:4,fill:css("--series-1")});
    hook(r,`${k}: <b>${Math.round(P[k]*100)}%</b> chance of a pause`);s.appendChild(r);
    const tl=el("text",{x:pl-8,y:y+bh/2+4,"text-anchor":"end",class:"ax"});tl.textContent=k;s.appendChild(tl);
    const vl=el("text",{x:pl+w+6,y:y+bh/2+4,class:"ax",fill:css("--text-secondary")});vl.textContent=Math.round(P[k]*100)+"%";s.appendChild(vl);
  });
  document.getElementById("pause").innerHTML="";document.getElementById("pause").appendChild(s);
}

function fitts(){
  const F=DATA.fitts,W=520,Hh=230,pl=44,pr=22,pt=12,pb=34;
  const s=el("svg",{viewBox:`0 0 ${W} ${Hh}`}),iw=W-pl-pr,ih=Hh-pt-pb;
  const tmax=Math.max(...F.points.map(p=>p.t))*1.05;
  const X=d=>pl+d/F.maxD*iw, Y=t=>pt+ih-t/tmax*ih;
  for(let g=0;g<=4;g++){const yy=pt+ih-g/4*ih;
    s.appendChild(el("line",{x1:pl,y1:yy,x2:W-pr,y2:yy,stroke:css("--grid"),"stroke-width":1}));
    const tx=el("text",{x:pl-6,y:yy+3,"text-anchor":"end",class:"ax"});tx.textContent=Math.round(tmax*g/4);s.appendChild(tx);}
  for(const p of F.points){const c=el("circle",{cx:X(p.d),cy:Y(p.t),r:3,fill:css("--series-1"),opacity:0.55});
    hook(c,`${Math.round(p.d)} px: <b>${Math.round(p.t)} ms</b>`);s.appendChild(c);}
  let dpath="";for(let d=0;d<=F.maxD;d+=20){const t=F.a_ms+F.b_ms*Math.sqrt(d);dpath+=(d===0?"M":"L")+X(d)+" "+Y(t)+" ";}
  s.appendChild(el("path",{d:dpath,fill:"none",stroke:css("--series-2"),"stroke-width":2.5}));
  for(let g=0;g<=5;g++){const dv=F.maxD*g/5;const an=g==0?"start":(g==5?"end":"middle");const tx=el("text",{x:X(dv),y:Hh-14,"text-anchor":an,class:"ax"});tx.textContent=Math.round(dv);s.appendChild(tx);}
  const at=el("text",{x:pl+iw/2,y:Hh-1,"text-anchor":"middle",class:"axtitle"});at.textContent="distance (px)   |   y: move time (ms)";s.appendChild(at);
  document.getElementById("fitts").innerHTML="";document.getElementById("fitts").appendChild(s);
}

function traj(){
  const Tj=DATA.traj,W=1040;
  const all=[];Tj.paths.forEach(p=>p.forEach(q=>all.push(q)));all.push(Tj.start,Tj.target);
  let minx=Math.min(...all.map(p=>p.x)),maxx=Math.max(...all.map(p=>p.x));
  let miny=Math.min(...all.map(p=>p.y)),maxy=Math.max(...all.map(p=>p.y));
  const px=(maxx-minx)*0.07+28, py=(maxy-miny)*0.14+28;
  minx-=px;maxx+=px;miny-=py;maxy+=py;
  const bw=maxx-minx,bh=maxy-miny,Hh=Math.max(W*bh/bw,300);
  const X=x=>(x-minx)/bw*W, Y=y=>(y-miny)/bh*Hh;
  const s=el("svg",{viewBox:`0 0 ${W} ${Hh.toFixed(0)}`});
  s.appendChild(el("rect",{x:0,y:0,width:W,height:Hh,fill:css("--plane")}));
  const cols=[css("--series-1"),css("--series-2")];
  Tj.paths.forEach((p,i)=>{
    let d="";p.forEach((q,j)=>{d+=(j===0?"M":"L")+X(q.x).toFixed(1)+" "+Y(q.y).toFixed(1)+" ";});
    s.appendChild(el("path",{d:d,fill:"none",stroke:cols[i],"stroke-width":2.4,opacity:0.92}));
  });
  const S=Tj.start,G=Tj.target;
  s.appendChild(el("circle",{cx:X(S.x),cy:Y(S.y),r:7,fill:"none",stroke:css("--muted"),"stroke-width":2}));
  const st=el("text",{x:X(S.x)+12,y:Y(S.y)+4,class:"ax"});st.textContent="start";s.appendChild(st);
  s.appendChild(el("circle",{cx:X(G.x),cy:Y(G.y),r:7,fill:css("--text-primary")}));
  const gt=el("text",{x:X(G.x)-12,y:Y(G.y)-10,"text-anchor":"end",class:"ax"});gt.textContent="target";s.appendChild(gt);
  document.getElementById("traj").innerHTML="";document.getElementById("traj").appendChild(s);
}

function renderAll(){
  document.getElementById("kpis").innerHTML="";kpis();
  histChart("intervals",DATA.interval_hist,"ms",DATA.fence_ms);
  histChart("dwell",DATA.dwell_hist,"ms",null);
  strip();pauseBars();fitts();traj();
}
document.getElementById("theme").addEventListener("click",()=>{
  const r=document.documentElement,light=r.getAttribute("data-theme")==="light";
  r.setAttribute("data-theme",light?"dark":"light");
  document.getElementById("theme").textContent=light?"Light":"Dark";
  renderAll();
});
renderAll();
</script>
</body>
</html>
"""
