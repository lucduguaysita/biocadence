"""A small local web UI (studio) for recording, training, replaying, reporting.

Runs on the standard library plus numpy (statistical model + serving); the
optional neural mouse training also needs torch, which ComfyUI already ships.

The page has a Record toggle (with a timer) that captures a session only while
active, a mouse box that asks for single clicks, double clicks, drag-and-drop
and scrolling (each counted separately), and a typing box. Train fits the fast
statistical model (typing + mouse) and then trains the neural pointer model in
the background with a progress bar, a Stop button and a training timer. Replay
previews the typing, and Generate HTML builds the inspector. You choose whether
each Train adds to the existing data or starts fresh.

Start it with:  python -m biocadence ui
"""

import json
import os
import re
import shutil
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import features, viz, fingerprint
from .model import Model
from .planner import Planner
from .schema import dump_jsonl, load_jsonl

WORKDIR = "."
TRAIN = {"running": False, "phase": "idle", "step": 0, "total": 0,
         "loss": None, "done": False, "error": None, "device": None}
STOP = {"flag": False}
_PT = {"mtime": None, "bundle": None}
_DEV = {"info": None}


def _load_pt(path):
    """Load and cache the neural checkpoint, reloading only if the file changed."""
    mt = os.path.getmtime(path)
    if _PT["mtime"] != mt:
        from .neural.train import load_checkpoint
        _PT["bundle"] = load_checkpoint(path)
        _PT["mtime"] = mt
    return _PT["bundle"]


def _device_info():
    # What processor the neural training will use, for the studio badge.
    # torch is imported lazily (the studio runs on numpy) and the answer
    # is cached, since GPU availability is stable for the life of a run.
    if _DEV["info"] is not None:
        return _DEV["info"]
    info = {"torch": False, "device": "cpu", "name": None, "version": None}
    try:
        import torch
        info["torch"] = True
        info["version"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = "cuda"
            try:
                info["name"] = torch.cuda.get_device_name(0)
            except Exception:
                info["name"] = "CUDA device"
    except Exception as exc:
        info["error"] = str(exc)
    _DEV["info"] = info
    return info


def _paths():
    j = os.path.join
    return (j(WORKDIR, "me.jsonl"), j(WORKDIR, "me.model.json"),
            j(WORKDIR, "report.html"), j(WORKDIR, "me.pointer.pt"))


def _fp_path():
    return os.path.join(WORKDIR, "fingerprint.json")


# A profile is a named snapshot of these active-workspace artifacts.
ARTIFACTS = ["me.jsonl", "me.model.json", "me.pointer.pt", "fingerprint.json"]


def _profiles_dir():
    return os.path.join(WORKDIR, "profiles")


def _san(name):
    return re.sub(r"[^A-Za-z0-9 _-]", "_", name or "").strip()


def _profile_dir(name):
    return os.path.join(_profiles_dir(), name)


def _list_profiles():
    pd, out = _profiles_dir(), []
    if os.path.isdir(pd):
        for name in sorted(os.listdir(pd)):
            d = os.path.join(pd, name)
            if not os.path.isdir(d):
                continue
            info = {"name": name, "pointer": os.path.exists(os.path.join(d, "me.pointer.pt"))}
            fpp = os.path.join(d, "fingerprint.json")
            if os.path.exists(fpp):
                try:
                    fp = fingerprint.load(fpp)
                    info["n_keys"] = fp.get("n_keys")
                    info["n_mouse"] = fp.get("n_mouse")
                except Exception:
                    pass
            out.append(info)
    return out


def _save_events(new_events, append):
    me_jsonl = _paths()[0]
    combined = new_events
    if append and os.path.exists(me_jsonl):
        old = load_jsonl(me_jsonl)
        if old and new_events:
            off = old[-1]["t"] + 0.5
            new_events = [dict(e, t=e["t"] + off) for e in new_events]
            combined = old + new_events
        elif old:
            combined = old
    dump_jsonl(me_jsonl, combined)
    return combined


def _neural_worker(steps, preset):
    me_jsonl, _model, _report, pointer_path = _paths()
    try:
        from .neural.tokenizer import Tokenizer
        from .neural.train import train as ntrain
        TRAIN.update(running=True, phase="neural", step=0, total=steps,
                     device=_device_info()["device"],
                     loss=None, done=False, error=None)
        tok = Tokenizer()
        seqs = [tok.encode(load_jsonl(me_jsonl))]

        def prog(s, t, loss):
            TRAIN.update(step=s, total=t, loss=loss)

        ntrain(seqs, tok, preset=preset, steps=steps, batch_size=64,
               out_path=pointer_path, log_every=20,
               progress=prog, should_stop=lambda: STOP["flag"])
        TRAIN.update(running=False, done=True,
                     phase="stopped" if STOP["flag"] else "done")
    except Exception as exc:
        TRAIN.update(running=False, error=str(exc), phase="error", done=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, (bytes, bytearray)) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        return json.loads(self.rfile.read(n) if n else b"{}")

    def do_GET(self):
        me_jsonl, model_path, report_path, pointer_path = _paths()
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/report.html":
            if os.path.exists(report_path):
                with open(report_path, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            else:
                self._send(404, "no report yet", "text/plain")
        elif self.path == "/train_status":
            self._send(200, json.dumps(TRAIN))
        elif self.path == "/device":
            d = dict(_device_info())
            d["served_by"] = os.path.abspath(__file__)
            self._send(200, json.dumps(d))
        elif self.path == "/profiles":
            self._send(200, json.dumps({"profiles": _list_profiles()}))
        elif self.path == "/status":
            mouse = keys = 0
            if os.path.exists(me_jsonl):
                for e in load_jsonl(me_jsonl):
                    ty = e.get("type", "")
                    if ty.startswith("mouse") or ty == "scroll":
                        mouse += 1
                    elif ty.startswith("key"):
                        keys += 1
            self._send(200, json.dumps({"mouse": mouse, "keys": keys,
                       "model": os.path.exists(model_path),
                       "pointer": os.path.exists(pointer_path),
                       "fingerprint": os.path.exists(_fp_path())}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        me_jsonl, model_path, report_path, pointer_path = _paths()
        try:
            if self.path == "/train":
                body = self._read_json()
                events = body.get("events", [])
                append = bool(body.get("append", False))
                if not events and not (append and os.path.exists(me_jsonl)):
                    return self._send(400, json.dumps({"ok": False, "error": "nothing captured yet"}))
                combined = _save_events(events, append)
                feats = features.extract(combined)
                Model.fit(feats).save(model_path)
                fingerprint.save(fingerprint.enroll(combined), _fp_path())
                resp = {"ok": True, "meta": feats["meta"]}
                if body.get("neural"):
                    STOP["flag"] = False
                    TRAIN.update(running=True, phase="starting", step=0,
                                 device=_device_info()["device"],
                                 total=int(body.get("steps", 1500)), loss=None,
                                 done=False, error=None)
                    threading.Thread(target=_neural_worker,
                                     args=(int(body.get("steps", 1500)),
                                           body.get("preset", "base")),
                                     daemon=True).start()
                    resp["neural"] = True
                self._send(200, json.dumps(resp))

            elif self.path == "/train_stop":
                STOP["flag"] = True
                self._send(200, json.dumps({"ok": True}))

            elif self.path == "/plan":
                body = self._read_json()
                m = Model.load(model_path)
                tr = body.get("typo_rate", None)
                ops = Planner(m, seed=body.get("seed"),
                              typo_rate=None if tr is None else float(tr)).build(
                    [{"type": "type", "text": body.get("text", "")}])
                if body.get("inject"):
                    from . import replay_win
                    threading.Thread(target=lambda: replay_win.replay(ops, lead_in=3.0),
                                     daemon=True).start()
                    self._send(200, json.dumps({"ok": True, "injected": True}))
                else:
                    self._send(200, json.dumps({"ok": True, "ops": ops}))

            elif self.path == "/plan_mouse":
                if not os.path.exists(pointer_path):
                    return self._send(400, json.dumps({"ok": False, "error": "train the mouse model first"}))
                body = self._read_json()
                from .neural.sample import build_ops
                m, tok, cfg, dev = _load_pt(pointer_path)
                ops = build_ops(m, tok, dev, body.get("actions", []),
                                start_cursor=tuple(body.get("cursor", [50, 50])),
                                profile=float(body.get("profile", 0.6)))
                self._send(200, json.dumps({"ok": True, "ops": ops}))

            elif self.path == "/match":
                if not os.path.exists(_fp_path()):
                    return self._send(400, json.dumps({"ok": False, "error": "enroll first by pressing Train"}))
                events = self._read_json().get("events", [])
                if not events:
                    return self._send(400, json.dumps({"ok": False, "error": "record a fresh sample first"}))
                self._send(200, json.dumps(fingerprint.match(fingerprint.load(_fp_path()), events)))

            elif self.path == "/profile/save":
                name = _san(self._read_json().get("name"))
                if not name:
                    return self._send(400, json.dumps({"ok": False, "error": "give the profile a name"}))
                d = _profile_dir(name)
                os.makedirs(d, exist_ok=True)
                saved = []
                for a in ARTIFACTS:
                    src = os.path.join(WORKDIR, a)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(d, a))
                        saved.append(a)
                if not saved:
                    return self._send(400, json.dumps({"ok": False, "error": "nothing to save yet; press Train first"}))
                self._send(200, json.dumps({"ok": True, "name": name, "saved": saved}))

            elif self.path == "/profile/load":
                name = _san(self._read_json().get("name"))
                d = _profile_dir(name)
                if not os.path.isdir(d):
                    return self._send(400, json.dumps({"ok": False, "error": "no such profile"}))
                for a in ARTIFACTS:
                    src = os.path.join(d, a)
                    if os.path.exists(src):
                        shutil.copy2(src, os.path.join(WORKDIR, a))
                _PT["mtime"] = None   # force neural model reload
                self._send(200, json.dumps({"ok": True, "name": name}))

            elif self.path == "/profile/delete":
                name = _san(self._read_json().get("name"))
                d = _profile_dir(name)
                if os.path.isdir(d):
                    shutil.rmtree(d)
                self._send(200, json.dumps({"ok": True}))

            elif self.path == "/identify":
                events = self._read_json().get("events", [])
                if not events:
                    return self._send(400, json.dumps({"ok": False, "error": "record a sample first"}))
                results = []
                for p in _list_profiles():
                    fpp = os.path.join(_profile_dir(p["name"]), "fingerprint.json")
                    if not os.path.exists(fpp):
                        continue
                    try:
                        r = fingerprint.match(fingerprint.load(fpp), events)
                        if r.get("ok"):
                            results.append({"name": p["name"], "similarity": r["similarity"],
                                            "certainty": r["certainty"], "verdict": r["verdict"]})
                    except Exception:
                        pass
                results.sort(key=lambda x: -x["similarity"])
                self._send(200, json.dumps({"ok": True, "results": results}))

            elif self.path == "/report":
                import datetime
                name = (self._read_json().get("name") or "").strip() or None
                viz.write_report(Model.load(model_path), report_path, name=name,
                                 date=datetime.date.today().isoformat())
                self._send(200, json.dumps({"ok": True, "url": "/report.html"}))
            else:
                self._send(404, json.dumps({"ok": False, "error": "unknown endpoint"}))

        except FileNotFoundError:
            self._send(400, json.dumps({"ok": False, "error": "train a model first"}))
        except Exception as exc:
            self._send(500, json.dumps({"ok": False, "error": str(exc)}))


def run_ui(port=8765, workdir=".", open_browser=True):
    global WORKDIR
    WORKDIR = os.path.abspath(workdir)
    os.makedirs(WORKDIR, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"BioCadence studio running at {url}")
    print(f"workdir: {WORKDIR}   (me.jsonl, me.model.json, me.pointer.pt, report.html)")
    _d = _device_info()
    if _d["device"] == "cuda":
        print(f"compute: CUDA  ({_d.get('name') or 'GPU'}, torch {_d.get('version')})")
    elif _d.get("torch"):
        print(f"compute: CPU only  (torch {_d.get('version')} sees no CUDA GPU)")
    else:
        print("compute: CPU only  (PyTorch not installed)")
    print("Press Ctrl+C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        srv.shutdown()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BioCadence studio</title>
<style>
  :root{--plane:#0d0d0d;--surface:#1a1a19;--panel:#242322;--text:#fff;--dim:#c3c2b7;
    --muted:#898781;--line:rgba(255,255,255,.12);--blue:#3987e5;--orange:#d95926;--red:#e0524f;--green:#0ca30c}
  *{box-sizing:border-box}
  body{margin:0;background:var(--plane);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.45}
  .wrap{max-width:1120px;margin:0 auto;padding:22px 20px 60px}
  h1{font-size:21px;margin:0 0 2px}
  .sub{color:var(--dim);font-size:13px;margin:0 0 16px}
  .bar1{display:flex;align-items:center;gap:14px;flex-wrap:wrap;background:var(--surface);
    border:1px solid var(--line);border-radius:14px;padding:12px 16px;margin-bottom:14px}
  .timer{font-variant-numeric:tabular-nums;font-size:20px;font-weight:640;min-width:64px}
  .datainfo{font-size:12px;color:var(--dim);margin-left:auto}
  .datainfo b{color:var(--text)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  .card h2{font-size:14.5px;margin:0 0 4px}
  .card p{font-size:12.5px;color:var(--dim);margin:0 0 8px}
  .mprompt{font-size:12.5px;color:var(--blue);margin:0 0 8px;min-height:17px;font-weight:560}
  canvas{width:100%;height:340px;border-radius:10px;background:#151514;display:block;cursor:crosshair;touch-action:none}
  textarea{width:100%;height:300px;resize:vertical;border-radius:10px;border:1px solid var(--line);
    background:#151514;color:var(--text);padding:12px;font-size:14px;font-family:inherit;line-height:1.5}
  .count{font-size:12px;color:var(--muted);margin-top:8px}
  .count b{color:var(--text);font-variant-numeric:tabular-nums}
  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-top:14px;background:var(--surface);
    border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  input[type=text]{flex:1;min-width:200px;border-radius:9px;border:1px solid var(--line);background:#151514;color:var(--text);padding:9px 11px;font-size:13.5px}
  input[type=number]{width:78px;border-radius:9px;border:1px solid var(--line);background:#151514;color:var(--text);padding:8px;font-size:13px}
  select{border-radius:9px;border:1px solid var(--line);background:#151514;color:var(--text);padding:8px;font-size:13px}
  button{border:1px solid var(--line);border-radius:9px;padding:9px 15px;font-size:13.5px;font-weight:560;cursor:pointer;background:var(--panel);color:var(--text)}
  button.rec-btn{background:var(--green);border-color:transparent;min-width:96px}
  button.rec-btn.rec{background:var(--red)}
  button.primary{background:var(--blue);border-color:transparent}
  button.go{background:var(--orange);border-color:transparent}
  button:disabled{opacity:.5;cursor:not-allowed}
  .devchip{font-size:12px;padding:4px 11px;border-radius:20px;border:1px solid var(--line);color:var(--dim);white-space:nowrap;font-weight:560}
  .devchip.gpu{background:rgba(12,163,12,.16);border-color:transparent;color:#5fd66e}
  .devchip.cpu{background:rgba(217,89,38,.16);border-color:transparent;color:#eda183}
  label.chk{font-size:12.5px;color:var(--dim);display:flex;align-items:center;gap:6px}
  .status{margin-top:12px;font-size:13px;color:var(--dim);min-height:19px}
  .status b{color:var(--text)}
  .prog{display:none;margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:13px 16px}
  .prog .row{display:flex;align-items:center;gap:14px}
  .track{flex:1;height:9px;background:#151514;border-radius:6px;overflow:hidden}
  .fill{height:100%;width:0;background:var(--blue);transition:width .3s}
  .ptext{font-size:12.5px;color:var(--dim);margin-top:7px;font-variant-numeric:tabular-nums}
  .preview{margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
  .preview h2{font-size:14px;margin:0 0 8px}
  #screen{white-space:pre-wrap;min-height:60px;background:#151514;border-radius:10px;padding:12px;font-size:15px;font-family:ui-monospace,Consolas,monospace}
  .caret{display:inline-block;width:8px;margin-left:1px;background:var(--blue);animation:b 1s steps(1) infinite}
  @keyframes b{50%{opacity:0}}
  .scrollrow{display:flex;align-items:center;gap:10px;margin-top:10px}
  .scrolllabel{font-size:12px;color:var(--dim);white-space:nowrap}
  .scrollbox{flex:1;height:82px;overflow-y:scroll;background:#151514;border:1px solid var(--line);border-radius:8px}
  .scrolltall{height:660px;padding:8px 11px;color:var(--muted);font-size:12px;line-height:1.8}
  .matchcard{display:none;margin-top:12px;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:16px}
  .matchtop{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
  .matchpct{font-size:40px;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
  .matchverdict{font-size:14px;color:var(--dim)}
  .badge{font-size:12px;padding:3px 10px;border-radius:20px;border:1px solid var(--line);color:var(--dim)}
  .mtrack{height:12px;background:#151514;border-radius:7px;overflow:hidden;margin:12px 0 4px;position:relative}
  .mfill{height:100%;width:0;border-radius:7px;transition:width .5s}
  .mthr{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--muted);left:70%}
  .mfeat{font-size:11.5px;color:var(--muted);margin-top:9px;line-height:1.5}
  @media(max-width:760px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <h1>BioCadence studio</h1>
  <p class="sub">Record how you move and type, train a model of it, then replay and inspect it. Everything stays on this machine.</p>

  <div class="bar1">
    <button class="rec-btn" id="btnRecord">Record</button>
    <span class="timer" id="recTimer">00:00</span>
    <label class="chk"><input type="checkbox" id="append"> add to existing data (otherwise start fresh)</label>
    <span class="devchip" id="devBadgeTop" title="Which processor neural training will use">device: checking...</span>
    <span class="datainfo" id="dataInfo">saved on disk: 0 mouse, 0 keys</span>
  </div>

  <div class="bar1">
    <input type="text" id="profName" placeholder="profile name (e.g. alice)" style="flex:0 0 200px;min-width:0">
    <button id="btnSaveProf">Save profile</button>
    <select id="profList"><option value="">(profiles)</option></select>
    <button id="btnLoadProf">Load</button>
    <button id="btnDelProf">Delete</button>
    <button class="primary" id="btnIdentify">Identify</button>
    <span class="datainfo" id="profInfo">no profiles yet</span>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Mouse</h2>
      <p class="mprompt" id="mPrompt">Press Record, then do the tasks below.</p>
      <canvas id="cv" width="520" height="340"></canvas>
      <div class="scrollrow">
        <span class="scrolllabel">Scroll test</span>
        <div class="scrollbox" id="scrollbox"><div class="scrolltall">
          Scroll this panel with the wheel or by dragging the scrollbar. Your scrolling is captured while recording.<br><br>
          line 2<br>line 3<br>line 4<br>line 5<br>line 6<br>line 7<br>line 8<br>line 9<br>line 10<br>
          line 11<br>line 12<br>line 13<br>line 14<br>line 15<br>line 16<br>line 17<br>line 18<br>the end.
        </div></div>
      </div>
      <div class="count" id="mCount">moves <b>0</b> &nbsp; clicks <b>0</b> &nbsp; double <b>0</b> &nbsp; drag <b>0</b> &nbsp; scroll <b>0</b></div>
    </div>
    <div class="card">
      <h2>Typing</h2>
      <p>While recording, type this passage (or anything) at a natural pace. Non-sensitive text only.</p>
      <textarea id="ta" spellcheck="false" placeholder="The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."></textarea>
      <div class="count" id="kCount">keys <b>0</b></div>
    </div>
  </div>

  <div class="controls">
    <button class="primary" id="btnTrain">Train</button>
    <span class="devchip" id="devBadge" title="Which processor the neural training will use">device: checking...</span>
    <label class="chk">steps <input type="number" id="steps" value="1500" min="100" step="100"></label>
    <label class="chk">size
      <select id="preset"><option value="tiny">tiny</option><option value="base" selected>base</option><option value="large">large</option></select>
    </label>
    <span style="flex:1"></span>
    <input type="text" id="sample" value="Sphinx of some lazy brown fox, judge my vow.">
    <button id="btnReplay">Replay typing</button>
    <button id="btnReplayMouse">Replay mouse</button>
    <label class="chk"><input type="checkbox" id="inject"> for real (Win)</label>
    <button class="go" id="btnReport">Generate HTML</button>
    <button id="btnCheck">Check match</button>
    <button id="btnReset">Reset</button>
  </div>

  <div class="prog" id="progWrap">
    <div class="row">
      <div class="track"><div class="fill" id="bar"></div></div>
      <span class="timer" id="trainTimer">00:00</span>
      <button id="btnStop" style="background:var(--red);border-color:transparent">Stop training</button>
    </div>
    <div class="ptext" id="progText">starting...</div>
  </div>

  <div class="status" id="status">Press Record to start a session. Both mouse and typing are captured together.</div>

  <div class="matchcard" id="matchCard">
    <div class="matchtop">
      <span class="matchpct" id="matchPct">--</span>
      <span class="matchverdict" id="matchVerdict"></span>
      <span class="badge" id="matchCert"></span>
    </div>
    <div class="mtrack"><div class="mfill" id="matchFill"></div><div class="mthr" title="same-person threshold ~70%"></div></div>
    <div class="mfeat" id="matchFeat"></div>
  </div>

  <div class="preview">
    <h2>Typing preview</h2>
    <div id="screen"><span class="caret">&nbsp;</span></div>
  </div>
</div>

<script>
const events = [];
const T0 = performance.now();
const now = () => (performance.now() - T0) / 1000;
const $ = id => document.getElementById(id);
const BTN = {0:'left',1:'middle',2:'right'};
const fmt = s => { const m=Math.floor(s/60), x=Math.floor(s%60); return (m<10?'0':'')+m+':'+(x<10?'0':'')+x; };
let recording = false, recT0 = 0, recTimer = null;
let keyCount = 0;
const cnt = {moves:0, clicks:0, dbl:0, drags:0, scrolls:0};

function setStatus(h){ $('status').innerHTML = h; }
async function post(p, body){ const r = await fetch(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); return r.json(); }
async function refreshData(){ const s = await (await fetch('/status')).json();
  $('dataInfo').innerHTML = `saved: <b>${s.mouse}</b> mouse, <b>${s.keys}</b> keys &nbsp; typing model: <b>${s.model?'yes':'no'}</b> &nbsp; mouse model: <b>${s.pointer?'yes':'no'}</b> &nbsp; fingerprint: <b>${s.fingerprint?'yes':'no'}</b>`; }

async function fetchDevice(){
  try{
    const d = await (await fetch('/device')).json();
    console.log('BioCadence compute device:', d);
    const gpu = d.device === 'cuda';
    const label = gpu ? ('GPU: ' + (d.name || 'CUDA')) : 'CPU only (no GPU)';
    const title = gpu
      ? ('Neural training runs on ' + (d.name || 'the CUDA GPU') + ' (torch ' + (d.version || '?') + ').')
      : (d.torch
          ? ("This Python's torch " + (d.version || '') + " sees no CUDA GPU. Launch from your ComfyUI python to use the RTX.")
          : 'PyTorch is not installed in this Python, so training runs on CPU.');
    ['devBadge','devBadgeTop'].forEach(function(id){
      const el = $(id); if(!el) return;
      el.className = 'devchip ' + (gpu ? 'gpu' : 'cpu');
      el.textContent = label;
      el.title = title;
    });
  }catch(e){
    ['devBadge','devBadgeTop'].forEach(function(id){ const el=$(id); if(el) el.textContent='device: unknown'; });
  }
}

// ---------- record toggle ----------
$('btnRecord').addEventListener('click', () => {
  recording = !recording;
  const b = $('btnRecord');
  if(recording){
    b.textContent = 'Stop'; b.classList.add('rec'); recT0 = performance.now();
    recTimer = setInterval(() => $('recTimer').textContent = fmt((performance.now()-recT0)/1000), 200);
    setStatus('Recording. Click the dot, double-click the ringed dot, drag the square into the box, scroll, and type. Press Stop when done.');
    spawn();
  } else {
    b.textContent = 'Record'; b.classList.remove('rec'); clearInterval(recTimer);
    setStatus('Recorded this session. Press Train to build the model, or Record again to add more.');
  }
});

// ---------- typing capture ----------
function mapKey(e){
  const k = e.key;
  const named = {' ':'Key.space','Enter':'Key.enter','Backspace':'Key.backspace','Tab':'Key.tab',
    'Shift':'Key.shift','Control':'Key.ctrl','Alt':'Key.alt','CapsLock':'Key.caps_lock',
    'ArrowLeft':'Key.left','ArrowRight':'Key.right','ArrowUp':'Key.up','ArrowDown':'Key.down'};
  if(k in named) return named[k];
  if(k.length === 1) return k;
  return 'Key.' + k.toLowerCase();
}
$('ta').addEventListener('keydown', e => { if(!recording||e.repeat) return;
  events.push({t:now(),type:'key_press',key:mapKey(e),vk:e.keyCode||null});
  keyCount++; $('kCount').innerHTML = `keys <b>${keyCount}</b>`; });
$('ta').addEventListener('keyup', e => { if(!recording) return;
  events.push({t:now(),type:'key_release',key:mapKey(e),vk:e.keyCode||null}); });

// ---------- mouse capture + gesture tasks ----------
const cv = $('cv'), ctx = cv.getContext('2d');
let tgt = null, lastMove = -1;
const cycle = ['click','click','double','drag'];
let ci = 0;
function fit(){ const r = cv.getBoundingClientRect(); cv.width = r.width; cv.height = r.height; }
function rnd(a,b){ return a + Math.random()*(b-a); }
function bg(){ ctx.fillStyle = '#151514'; ctx.fillRect(0,0,cv.width,cv.height); }
function pos(e){ const r = cv.getBoundingClientRect(); return {x:e.clientX-r.left, y:e.clientY-r.top}; }
function near(px,py,x,y,r){ return Math.hypot(px-x,py-y) < r; }
function inZone(px,py,z){ return px>=z.x && px<=z.x+z.w && py>=z.y && py<=z.y+z.h; }
function mUpd(){ $('mCount').innerHTML = `moves <b>${cnt.moves}</b> &nbsp; clicks <b>${cnt.clicks}</b> &nbsp; double <b>${cnt.dbl}</b> &nbsp; drag <b>${cnt.drags}</b> &nbsp; scroll <b>${cnt.scrolls}</b>`; }
function prompt(){ const map = {click:'Click the orange dot', double:'Double-click the white-ringed dot', drag:'Drag the blue square into the outlined box'};
  $('mPrompt').textContent = map[tgt.type] + '   (and scroll the wheel over this area)'; }
function spawn(){
  const type = cycle[ci++ % cycle.length], m = 34;
  if(type === 'drag'){
    tgt = {type, obj:{x:rnd(m, cv.width*0.42), y:rnd(m, cv.height-m)},
           zone:{x:rnd(cv.width*0.6, cv.width-90), y:rnd(m, cv.height-70), w:72, h:54}, hold:false};
  } else {
    tgt = {type, x:rnd(m, cv.width-m), y:rnd(m, cv.height-m)};
  }
  draw(); prompt(); window.__tgt = tgt;
}
function draw(){
  bg();
  if(!tgt) return;
  if(tgt.type === 'drag'){
    const z = tgt.zone; ctx.setLineDash([6,4]); ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(57,135,229,.85)';
    ctx.strokeRect(z.x, z.y, z.w, z.h); ctx.setLineDash([]);
    const o = tgt.obj; ctx.fillStyle = '#3987e5'; ctx.fillRect(o.x-16, o.y-16, 32, 32);
  } else {
    ctx.beginPath(); ctx.arc(tgt.x, tgt.y, 11, 0, 7); ctx.fillStyle = '#d95926'; ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = tgt.type === 'double' ? '#fff' : 'rgba(255,255,255,.35)';
    if(tgt.type === 'double'){ ctx.beginPath(); ctx.arc(tgt.x, tgt.y, 18, 0, 7); ctx.stroke(); }
    else ctx.stroke();
  }
}
cv.addEventListener('mousemove', e => {
  const p = pos(e), t = now();
  if(recording && t - lastMove >= 0.008){ lastMove = t;
    events.push({t, type:'mouse_move', x:Math.round(p.x), y:Math.round(p.y)}); cnt.moves++; mUpd(); }
  if(tgt && tgt.type === 'drag' && tgt.hold){ tgt.obj.x = p.x; tgt.obj.y = p.y; draw(); }
  else if(recording){ ctx.fillStyle = 'rgba(57,135,229,.5)'; ctx.fillRect(p.x-1, p.y-1, 2, 2); }
});
cv.addEventListener('mousedown', e => { if(!recording) return; const p = pos(e);
  events.push({t:now(), type:'mouse_down', button:BTN[e.button]||'left', x:Math.round(p.x), y:Math.round(p.y)});
  if(tgt && tgt.type === 'drag' && near(p.x,p.y,tgt.obj.x,tgt.obj.y,22)) tgt.hold = true; });
cv.addEventListener('mouseup', e => { if(!recording) return; const p = pos(e);
  events.push({t:now(), type:'mouse_up', button:BTN[e.button]||'left', x:Math.round(p.x), y:Math.round(p.y)});
  if(tgt && tgt.type === 'click' && near(p.x,p.y,tgt.x,tgt.y,18)){ cnt.clicks++; spawn(); mUpd(); }
  else if(tgt && tgt.type === 'drag' && tgt.hold){
    if(inZone(p.x,p.y,tgt.zone)){ cnt.drags++; spawn(); mUpd(); } else { tgt.hold = false; draw(); } } });
cv.addEventListener('dblclick', e => { if(!recording) return; const p = pos(e);
  if(tgt && tgt.type === 'double' && near(p.x,p.y,tgt.x,tgt.y,20)){ cnt.dbl++; spawn(); mUpd(); } });
cv.addEventListener('wheel', e => { e.preventDefault(); if(!recording) return; const p = pos(e);
  events.push({t:now(), type:'scroll', x:Math.round(p.x), y:Math.round(p.y), dx:0, dy:e.deltaY<0?1:-1});
  cnt.scrolls++; mUpd(); }, {passive:false});
cv.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('resize', () => { fit(); if(tgt) draw(); });

// dedicated scroll panel: captures wheel and scrollbar drags reliably
const sb = $('scrollbox');
let lastTop = 0, lastScrollT = -1;
sb.addEventListener('scroll', () => {
  if(!recording){ lastTop = sb.scrollTop; return; }
  const t = now();
  if(t - lastScrollT < 0.016) return;
  lastScrollT = t;
  const d = sb.scrollTop - lastTop; lastTop = sb.scrollTop;
  if(d === 0) return;
  events.push({t, type:'scroll', x:0, y:0, dx:0, dy: d > 0 ? -1 : 1});
  cnt.scrolls++; mUpd();
});

// ---------- train (statistical + background neural) ----------
let watch = null, trainTimer = null;
$('btnTrain').addEventListener('click', async () => {
  const steps = parseInt($('steps').value) || 1500, preset = $('preset').value;
  setStatus('Training the typing and mouse model...');
  const r = await post('/train', {events, append:$('append').checked, neural:true, steps, preset});
  if(!r.ok){ setStatus('Error: ' + r.error); return; }
  const m = r.meta;
  setStatus(`Statistical model trained on <b>${m.n_keys}</b> keys and <b>${m.n_move_segments}</b> mouse moves (${m.n_clicks} clicks). Neural mouse model training started.`);
  refreshData();
  if(r.neural) watchTraining();
});
function watchTraining(){
  $('progWrap').style.display = 'block'; $('btnStop').style.display = 'inline-block';
  $('btnTrain').disabled = true;
  const t0 = performance.now();
  trainTimer = setInterval(() => $('trainTimer').textContent = fmt((performance.now()-t0)/1000), 200);
  watch = setInterval(async () => {
    const s = await (await fetch('/train_status')).json();
    const pct = s.total ? Math.round(100*s.step/s.total) : 0;
    $('bar').style.width = pct + '%';
    const dev = s.device ? ` on ${s.device}` : '';
    $('progText').textContent = `${s.phase}${dev}  step ${s.step}/${s.total}` + (s.loss!=null ? `  loss ${s.loss.toFixed(3)}` : '');
    if(s.done || s.error){
      clearInterval(watch); clearInterval(trainTimer);
      $('btnStop').style.display = 'none'; $('btnTrain').disabled = false;
      if(s.error) setStatus('Neural training error: ' + s.error);
      else setStatus(s.phase === 'stopped' ? 'Neural training stopped; partial mouse model saved.' : 'Neural mouse model trained. Both models are ready.');
      refreshData();
    }
  }, 500);
}
$('btnStop').addEventListener('click', async () => { await post('/train_stop', {}); setStatus('Stopping neural training...'); });

// ---------- replay / report / reset ----------
$('btnReplay').addEventListener('click', async () => {
  const text = $('sample').value;
  if($('inject').checked){ const r = await post('/plan', {text, inject:true});
    setStatus(r.ok ? 'Typing it into your focused window in 3 seconds (F12 aborts).' : 'Error: ' + r.error); return; }
  setStatus('Sampling your typing rhythm...');
  const r = await post('/plan', {text});
  if(!r.ok){ setStatus('Error: ' + r.error); return; }
  setStatus('Replaying your typing below.'); animate(r.ops);
});
$('btnReport').addEventListener('click', async () => {
  setStatus('Building report...');
  const name = ($('profName').value.trim() || $('profList').value || '');
  const r = await post('/report', {name});
  if(r.ok){ setStatus('Report ready.'); window.open(r.url, '_blank'); } else setStatus('Error: ' + r.error);
});
// ---------- profiles (save / load / delete / identify) ----------
async function refreshProfiles(){
  const s = await (await fetch('/profiles')).json();
  const sel = $('profList'), cur = sel.value;
  sel.innerHTML = '<option value="">(profiles)</option>' +
    s.profiles.map(p => `<option value="${p.name}">${p.name}${p.pointer ? ' *' : ''}</option>`).join('');
  if(cur) sel.value = cur;
  $('profInfo').textContent = s.profiles.length ? (s.profiles.length + ' profile' + (s.profiles.length > 1 ? 's' : '') + ' saved') : 'no profiles yet';
}
$('btnSaveProf').addEventListener('click', async () => {
  const name = $('profName').value.trim();
  if(!name){ setStatus('Enter a profile name first.'); return; }
  const r = await post('/profile/save', {name});
  setStatus(r.ok ? `Saved profile "${r.name}" (${r.saved.join(', ')}).` : 'Error: ' + r.error);
  refreshProfiles();
});
$('btnLoadProf').addEventListener('click', async () => {
  const name = $('profList').value;
  if(!name){ setStatus('Pick a profile to load.'); return; }
  const r = await post('/profile/load', {name});
  setStatus(r.ok ? `Loaded profile "${name}"; its model and fingerprint are now active.` : 'Error: ' + r.error);
  refreshData();
});
$('btnDelProf').addEventListener('click', async () => {
  const name = $('profList').value;
  if(!name){ setStatus('Pick a profile to delete.'); return; }
  await post('/profile/delete', {name}); setStatus(`Deleted profile "${name}".`); refreshProfiles();
});
$('btnIdentify').addEventListener('click', async () => {
  if(events.length === 0){ setStatus('Record a fresh sample first, then Identify.'); return; }
  setStatus('Identifying against saved profiles...');
  const r = await post('/identify', {events});
  if(!r.ok){ setStatus('Error: ' + r.error); return; }
  if(!r.results.length){ setStatus('No saved profiles to compare against. Save some friends first.'); return; }
  const best = r.results[0];
  const col = best.similarity >= 70 ? '#0ca30c' : (best.similarity >= 45 ? '#fab219' : '#d03b3b');
  $('matchCard').style.display = 'block';
  $('matchPct').textContent = best.similarity + '%'; $('matchPct').style.color = col;
  $('matchFill').style.width = best.similarity + '%'; $('matchFill').style.background = col;
  $('matchVerdict').textContent = 'best match: ' + best.name + ' -- ' + best.verdict;
  $('matchCert').textContent = 'certainty: ' + best.certainty;
  $('matchFeat').textContent = 'ranking: ' + r.results.map(x => `${x.name} ${x.similarity}%`).join('   |   ');
  setStatus(`Identified: ${best.name} at ${best.similarity}%.`);
});
$('btnCheck').addEventListener('click', async () => {
  if(events.length === 0){ setStatus('Record a fresh sample first (Record, do some typing and mouse, Stop), then Check match.'); return; }
  setStatus('Comparing this sample against your enrolled fingerprint...');
  const r = await post('/match', {events});
  if(!r.ok){ setStatus('Error: ' + r.error); return; }
  const col = r.similarity >= 70 ? '#0ca30c' : (r.similarity >= 45 ? '#fab219' : '#d03b3b');
  $('matchCard').style.display = 'block';
  $('matchPct').textContent = r.similarity + '%'; $('matchPct').style.color = col;
  $('matchFill').style.width = r.similarity + '%'; $('matchFill').style.background = col;
  $('matchVerdict').textContent = r.verdict + ' (distance ' + r.distance + ')';
  $('matchCert').textContent = 'certainty: ' + r.certainty + ' (' + r.certainty_pct + '%)';
  $('matchFeat').textContent = 'sample ' + r.n_sample + ' events vs enrolled ' + r.n_enrolled +
    '; ' + r.shared + ' shared features; biggest differences: ' + r.top.map(t => t.feature + ' (z' + t.z + ')').join(', ');
  setStatus('Match: ' + r.similarity + '% (' + r.verdict + ').');
});
$('btnReset').addEventListener('click', () => {
  events.length = 0; keyCount = 0; Object.keys(cnt).forEach(k => cnt[k] = 0);
  mUpd(); $('kCount').innerHTML = 'keys <b>0</b>'; $('ta').value = '';
  setStatus('Session buffer cleared. This does not touch data already saved on disk.');
});
function tokenToChar(tok){ const n = {'Key.space':' ','Key.enter':'\n','Key.tab':'\t','Key.backspace':'\b'};
  if(tok in n) return n[tok]; return tok.length === 1 ? tok : ''; }
let timers = [];
function animate(ops){
  timers.forEach(clearTimeout); timers = [];
  let buf = ''; const screen = $('screen'); screen.textContent = '';
  ops.filter(o => o.op === 'key_down').forEach(o => timers.push(setTimeout(() => {
    const ch = tokenToChar(o.token);
    if(ch === '\b') buf = buf.slice(0,-1); else if(ch) buf += ch;
    screen.textContent = buf;
    const c = document.createElement('span'); c.className = 'caret'; c.innerHTML = '&nbsp;'; screen.appendChild(c);
  }, o.t * 1000)));
}

// ---------- replay mouse (neural pointer model, animated on the canvas) ----------
let mtimers = [];
$('btnReplayMouse').addEventListener('click', async () => {
  const w = cv.width, h = cv.height;
  const pts = [[w*0.22,h*0.30],[w*0.75,h*0.62],[w*0.42,h*0.82]].map(p => [Math.round(p[0]), Math.round(p[1])]);
  const actions = []; pts.forEach(p => { actions.push({type:'move', to:p}); actions.push({type:'click', to:p}); });
  const cursor = [Math.round(w*0.10), Math.round(h*0.12)];
  setStatus('Sampling your mouse motion...');
  const r = await post('/plan_mouse', {actions, cursor, profile:0.6});
  if(!r.ok){ setStatus('Error: ' + r.error); return; }
  setStatus('Replaying your mouse path on the canvas (blue), aiming at the orange targets.');
  animateMouse(r.ops, pts);
});
function animateMouse(ops, pts){
  mtimers.forEach(clearTimeout); mtimers = [];
  bg();
  const path = [];
  ops.forEach(o => mtimers.push(setTimeout(() => {
    if(o.op === 'mouse_move'){ path.push([o.x, o.y]); drawPath(path, pts); }
    else if(o.op === 'mouse_down'){ drawClick(path[path.length-1] || [o.x, o.y]); }
  }, o.t * 1000)));
}
function drawPath(path, pts){
  bg();
  if(pts) pts.forEach(p => { ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, 7); ctx.fillStyle = 'rgba(217,89,38,.55)'; ctx.fill(); });
  ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2; ctx.beginPath();
  path.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]));
  ctx.stroke();
  const last = path[path.length-1];
  if(last){ ctx.beginPath(); ctx.arc(last[0], last[1], 5, 0, 7); ctx.fillStyle = '#fff'; ctx.fill(); }
}
function drawClick(p){ ctx.beginPath(); ctx.arc(p[0], p[1], 13, 0, 7); ctx.strokeStyle = '#d95926'; ctx.lineWidth = 2; ctx.stroke(); }

fit(); bg(); mUpd(); refreshData(); refreshProfiles(); fetchDevice();
</script>
</body>
</html>
"""
