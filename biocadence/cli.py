"""Command line interface.

  python -m biocadence record  --out me.jsonl
  python -m biocadence train   --out me.model.json me.jsonl [more.jsonl ...]
  python -m biocadence plan    --model me.model.json --text "hello" --out plan.json
  python -m biocadence replay  --model me.model.json --text "hello" [--dry-run]
  python -m biocadence inspect --model me.model.json
  python -m biocadence demo    --out-dir demo_out
"""

import argparse
import json
import os

from . import features, model as model_mod
from .schema import load_jsonl, dump_json


def _actions_from_args(args):
    if getattr(args, "actions", None):
        with open(args.actions, encoding="utf-8") as f:
            return json.load(f)
    if getattr(args, "text", None) is not None:
        return [{"type": "type", "text": args.text}]
    raise SystemExit("provide --text or --actions")


def _plan(args):
    from .planner import Planner
    m = model_mod.Model.load(args.model)
    actions = _actions_from_args(args)
    p = Planner(m, seed=args.seed, typo_rate=args.typo_rate,
                cursor=tuple(args.cursor))
    ops = p.build(actions)
    return m, ops


def _summarize_plan(ops):
    dur = ops[-1]["t"] if ops else 0.0
    chars = sum(1 for o in ops if o["op"] == "key_down"
                and (len(o["token"]) == 1 or o["token"] == "Key.space"))
    moves = sum(1 for o in ops if o["op"] == "mouse_move")
    clicks = sum(1 for o in ops if o["op"] == "mouse_down")
    wpm = (chars / 5.0) / (dur / 60.0) if dur > 0 else 0.0
    return {"duration_s": round(dur, 3), "chars": chars, "moves": moves,
            "clicks": clicks, "wpm": round(wpm, 1)}


def cmd_record(args):
    from .recorder import record_session
    record_session(args.out, stop_key=args.stop_key)


def cmd_train(args):
    feats = []
    for path in args.inputs:
        f = features.extract(load_jsonl(path))
        feats.append(f)
        print(f"{path}: {f['meta']}")
    m = model_mod.Model.fit(feats)
    m.save(args.out)
    print(f"model saved -> {args.out}  ({m.p['meta']})")


def cmd_plan(args):
    _m, ops = _plan(args)
    dump_json(args.out, ops)
    print(f"plan saved -> {args.out}  {_summarize_plan(ops)}")


def cmd_replay(args):
    from . import replay_win
    if args.plan:
        with open(args.plan, encoding="utf-8") as f:
            ops = json.load(f)
    else:
        _m, ops = _plan(args)
    print(f"plan: {_summarize_plan(ops)}")
    replay_win.replay(ops, use_scancode_for_chars=args.scancode,
                      speed=args.speed, lead_in=args.lead_in,
                      dry_run=args.dry_run,
                      on_op=(lambda o: print(f"  {o['t']:.3f} {o['op']} "
                                             f"{o.get('token') or o.get('button') or ''}"))
                      if args.verbose else None)


def cmd_inspect(args):
    m = model_mod.Model.load(args.model)
    p = m.p
    print(f"model meta: {p['meta']}")
    g = p["dwell"]["global"]
    print(f"dwell global median: {__import__('math').exp(g['mu'])*1000:.0f} ms  "
          f"(n={g['n']})")
    fg = p["flight"]["global"]
    print(f"fluent flight median: {__import__('math').exp(fg['mu'])*1000:.0f} ms  "
          f"(n={fg['n']})   pause fence: {p['flight']['fence']*1000:.0f} ms")
    print("pause P(context):", {k: round(v, 2) for k, v in p["pause"]["prob"].items()})
    mt = p["mouse"]["move_time"]
    print(f"mouse move_time: {mt['a']*1000:.0f} ms + {mt['b']*1000:.1f} ms*sqrt(px)  "
          f"overshoot p={p['mouse']['overshoot_prob']:.2f}")
    print(f"click median: {__import__('math').exp(p['click']['duration']['mu'])*1000:.0f} ms")


def cmd_neural_train(args):
    from .neural.tokenizer import Tokenizer
    from .neural.train import train
    tok = Tokenizer()
    seqs = []
    for path in args.inputs:
        seqs.append(tok.encode(load_jsonl(path)))
        print(f"{path}: {len(seqs[-1])} pointer tokens")
    train(seqs, tok, preset=args.preset, steps=args.steps, batch_size=args.batch,
          block_size=args.block, lr=args.lr, device=args.device,
          out_path=args.out, seed=args.seed)


def cmd_neural_replay(args):
    from .neural.train import load_checkpoint
    from .neural.sample import build_ops
    from . import replay_win
    m, tok, cfg, dev = load_checkpoint(args.model, device=args.device)
    if args.actions:
        with open(args.actions, encoding="utf-8") as f:
            actions = json.load(f)
    elif args.to:
        actions = [{"type": "move", "to": list(args.to)},
                   {"type": "click", "to": list(args.to)}]
    else:
        raise SystemExit("provide --actions or --to X Y")
    ops = build_ops(m, tok, dev, actions, start_cursor=tuple(args.cursor),
                    seed=args.seed, temperature=args.temperature,
                    pool_tokens=args.pool, profile=args.profile)
    print("plan:", _summarize_plan(ops))
    if args.out:
        dump_json(args.out, ops)
        print(f"plan saved -> {args.out}")
    replay_win.replay(ops, speed=args.speed, lead_in=args.lead_in,
                      dry_run=args.dry_run)


def cmd_ui(args):
    from .ui import run_ui
    run_ui(port=args.port, workdir=args.workdir, open_browser=not args.no_browser)


def cmd_fingerprint(args):
    from . import fingerprint
    fp = fingerprint.enroll(load_jsonl(args.input))
    fingerprint.save(fp, args.out)
    print(f"fingerprint enrolled from {args.input} -> {args.out}  "
          f"({fp['n_keys']} keys, {fp['n_mouse']} mouse, {len(fp['vector'])} features)")


def cmd_match(args):
    from . import fingerprint
    res = fingerprint.match(fingerprint.load(args.enrolled), load_jsonl(args.sample))
    print(json.dumps(res, indent=2))


def cmd_report(args):
    from .viz import write_report
    m = model_mod.Model.load(args.model)
    write_report(m, args.out, seed=args.seed, name=args.name)
    print(f"report -> {args.out}")


def cmd_demo(args):
    from .synth import synth_trace
    from .planner import Planner
    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw.jsonl")
    model_path = os.path.join(args.out_dir, "me.model.json")
    plan_path = os.path.join(args.out_dir, "plan.json")

    events = synth_trace(seed=args.seed)
    from .schema import dump_jsonl
    dump_jsonl(raw_path, events)
    feats = features.extract(events)
    print("captured (synthetic):", feats["meta"])

    m = model_mod.Model.fit(feats)
    m.save(model_path)

    sample = ("Sphinx of black quartz, judge my vow. "
              "The model now types this in your rhythm!")
    actions = [
        {"type": "type", "text": sample},
        {"type": "pause", "seconds": 0.6},
        {"type": "move", "to": [1240, 720]}, {"type": "click"},
        {"type": "move", "to": [360, 240]}, {"type": "click", "double": True},
    ]
    p = Planner(m, seed=args.seed, typo_rate=args.typo_rate, cursor=(960, 540))
    ops = p.build(actions)
    dump_json(plan_path, ops)
    print("sample plan:", _summarize_plan(ops))
    print(f"\nwrote:\n  {raw_path}\n  {model_path}\n  {plan_path}")


def build_parser():
    ap = argparse.ArgumentParser(prog="biocadence",
                                 description="Model and replay your input style.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="capture a session (Windows/pynput)")
    r.add_argument("--out", required=True)
    r.add_argument("--stop-key", default="f12")
    r.set_defaults(func=cmd_record)

    t = sub.add_parser("train", help="fit a model from one or more recordings")
    t.add_argument("inputs", nargs="+")
    t.add_argument("--out", required=True)
    t.set_defaults(func=cmd_train)

    def add_plan_args(sp):
        sp.add_argument("--model", required=True)
        sp.add_argument("--text", default=None)
        sp.add_argument("--actions", default=None, help="JSON action list")
        sp.add_argument("--typo-rate", type=float, default=None, dest="typo_rate",
                        help="override learned correction rate (0 disables corrections)")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--cursor", type=int, nargs=2, default=[0, 0])

    pl = sub.add_parser("plan", help="build a timed op stream and save it")
    add_plan_args(pl)
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=cmd_plan)

    rp = sub.add_parser("replay", help="inject a plan via Windows SendInput")
    add_plan_args(rp)
    rp.add_argument("--plan", default=None, help="prebuilt plan.json (skip modeling)")
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("--scancode", action="store_true", help="inject chars as scan codes")
    rp.add_argument("--speed", type=float, default=1.0)
    rp.add_argument("--lead-in", type=float, default=3.0, dest="lead_in")
    rp.add_argument("--verbose", action="store_true")
    rp.set_defaults(func=cmd_replay)

    ins = sub.add_parser("inspect", help="print a model summary")
    ins.add_argument("--model", required=True)
    ins.set_defaults(func=cmd_inspect)

    nt = sub.add_parser("neural-train", help="train the pointer transformer on recordings")
    nt.add_argument("inputs", nargs="+")
    nt.add_argument("--out", required=True)
    nt.add_argument("--preset", default="base", choices=["tiny", "base", "large"])
    nt.add_argument("--steps", type=int, default=3000)
    nt.add_argument("--batch", type=int, default=64)
    nt.add_argument("--block", type=int, default=None)
    nt.add_argument("--lr", type=float, default=3e-4)
    nt.add_argument("--seed", type=int, default=0)
    nt.add_argument("--device", default=None, help="cuda or cpu (auto if unset)")
    nt.set_defaults(func=cmd_neural_train)

    nr = sub.add_parser("neural-replay", help="sample the pointer model and inject")
    nr.add_argument("--model", required=True, help="a .pt checkpoint")
    nr.add_argument("--actions", default=None, help="JSON action list")
    nr.add_argument("--to", type=int, nargs=2, default=None, help="move+click to X Y")
    nr.add_argument("--cursor", type=int, nargs=2, default=[0, 0])
    nr.add_argument("--temperature", type=float, default=0.9)
    nr.add_argument("--profile", type=float, default=0.6,
                    help="ease-in/out blend: 0 pure model timing, 1 full min-jerk")
    nr.add_argument("--pool", type=int, default=1600, help="behavior tokens to sample from")
    nr.add_argument("--seed", type=int, default=0)
    nr.add_argument("--device", default=None)
    nr.add_argument("--out", default=None, help="also save the op stream here")
    nr.add_argument("--dry-run", action="store_true")
    nr.add_argument("--speed", type=float, default=1.0)
    nr.add_argument("--lead-in", type=float, default=3.0, dest="lead_in")
    nr.set_defaults(func=cmd_neural_replay)

    fpc = sub.add_parser("fingerprint", help="enroll a behavioral fingerprint from a recording")
    fpc.add_argument("input")
    fpc.add_argument("--out", required=True)
    fpc.set_defaults(func=cmd_fingerprint)

    mc = sub.add_parser("match", help="match a sample recording against an enrolled fingerprint")
    mc.add_argument("--enrolled", required=True)
    mc.add_argument("--sample", required=True)
    mc.set_defaults(func=cmd_match)

    u = sub.add_parser("ui", help="launch the local training/replay/report web UI")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--workdir", default=".")
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(func=cmd_ui)

    rep = sub.add_parser("report", help="build an HTML model inspector")
    rep.add_argument("--model", required=True)
    rep.add_argument("--out", required=True)
    rep.add_argument("--name", default=None, help="profile name shown in the header")
    rep.add_argument("--seed", type=int, default=0)
    rep.set_defaults(func=cmd_report)

    d = sub.add_parser("demo", help="synthetic end-to-end demo (no hardware)")
    d.add_argument("--out-dir", default="demo_out", dest="out_dir")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--typo-rate", type=float, default=0.0, dest="typo_rate")
    d.set_defaults(func=cmd_demo)

    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
