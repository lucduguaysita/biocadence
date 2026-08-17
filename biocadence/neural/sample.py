"""Sample the trained model and turn it into replay-ready op streams.

The model learns pointer *style* as relative motion. To aim that style at a
real target, we sample a pool of behavior, pull out approach-and-click gestures
(a run of moves ending in a left button press), pick the one whose length is
closest to the move we need, then rotate and scale that gesture so its net
displacement lands exactly on the target. Curvature, micro-timing and click
dwell are the model's; only the aim is imposed. A straight min-jerk fallback
covers the case where no gesture is available yet (e.g. a barely trained model).
"""

import math

import numpy as np
import torch

from .tokenizer import BOS, records_to_ops

MOVE_SPLIT_DT = 0.35     # a gap longer than this starts a new movement


def sample_records(model, tokenizer, device, n_tokens=1200, temperature=0.9,
                   top_k=None, seed=0):
    torch.manual_seed(seed)
    idx = torch.tensor([[BOS]], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=n_tokens, temperature=temperature,
                         top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def extract_click_gestures(records):
    """approach-moves + left_down .. left_up, with the approach trimmed to the
    last continuous movement."""
    gestures = []
    approach, hold = [], []
    down_dt = None
    state = "idle"
    for r in records:
        k = r["kind"]
        if state == "idle":
            if k == "move":
                if r["dt"] > MOVE_SPLIT_DT:
                    approach = []
                approach.append(r)
            elif k == "left_down":
                down_dt, hold, state = r["dt"], [], "hold"
            else:
                approach = []
        else:  # hold
            if k == "move":
                hold.append(r)
            elif k == "left_up":
                gestures.append({"approach": approach[:], "hold": hold[:],
                                 "down_dt": down_dt, "up_dt": r["dt"]})
                approach, state = [], "idle"
            elif k == "left_down":
                down_dt, hold = r["dt"], []
            else:
                approach, hold, state = [], [], "idle"
    return gestures


def extract_move_gestures(records):
    """maximal runs of consecutive moves (split on events or long gaps)."""
    runs, cur = [], []
    for r in records:
        if r["kind"] == "move" and r["dt"] <= MOVE_SPLIT_DT:
            cur.append(r)
        elif r["kind"] == "move":
            if len(cur) >= 3:
                runs.append(cur)
            cur = [r]
        else:
            if len(cur) >= 3:
                runs.append(cur)
            cur = []
    if len(cur) >= 3:
        runs.append(cur)
    return runs


def _net(moves):
    if not moves:
        return np.zeros(2)
    return np.array([[m["dx"], m["dy"]] for m in moves], float).sum(0)


def _retarget_moves(moves, start, target):
    """Rotate+scale a move run so its net displacement equals target-start."""
    d = np.array([[m["dx"], m["dy"]] for m in moves], float)
    N = d.sum(0)
    Nlen = float(np.hypot(*N))
    D = np.array([target[0] - start[0], target[1] - start[1]], float)
    Dlen = float(np.hypot(*D))
    if Nlen < 1e-3 or Dlen < 1e-3:
        return None
    s = Dlen / Nlen
    ang = math.atan2(D[1], D[0]) - math.atan2(N[1], N[0])
    c, sn = math.cos(ang), math.sin(ang)
    R = np.array([[c, -sn], [sn, c]])
    d2 = (d @ R.T) * s
    return [{"kind": "move", "dt": m["dt"], "dx": float(v[0]), "dy": float(v[1])}
            for m, v in zip(moves, d2)]


def _straight_fallback(start, target):
    """A plain min-jerk move when no learned gesture is available."""
    D = np.array([target[0] - start[0], target[1] - start[1]], float)
    dist = float(np.hypot(*D))
    if dist < 1.0:
        return []
    dur = 0.10 + 0.018 * math.sqrt(dist)
    n = max(3, int(dur * 120))
    recs, prev = [], np.array(start, float)
    for i in range(1, n + 1):
        tau = i / n
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
        p = np.array(start, float) + D * s
        recs.append({"kind": "move", "dt": dur / n,
                     "dx": float(p[0] - prev[0]), "dy": float(p[1] - prev[1])})
        prev = p
    return recs


def _reprofile(moves, blend):
    """Re-time a move run so along-path progress follows a min-jerk profile
    (ease in and out), blended with the model's own timing. blend=0 keeps the
    model timing, blend=1 is a full ballistic profile. Positions are untouched,
    so the path shape and landing point are unchanged; only the pace changes."""
    if blend <= 0 or len(moves) < 3:
        return moves
    d = np.array([[m["dx"], m["dy"]] for m in moves], float)
    seg = np.hypot(d[:, 0], d[:, 1])
    total = float(seg.sum())
    T = float(sum(m["dt"] for m in moves))
    if total < 1e-6 or T < 1e-6:
        return moves
    f = np.cumsum(seg) / total                      # arc fraction at each step end
    taus = np.linspace(0.0, 1.0, 257)
    mj = 10 * taus ** 3 - 15 * taus ** 4 + 6 * taus ** 5
    g = (1.0 - blend) * taus + blend * mj           # blended progress, monotonic
    t_new = np.interp(f, g, taus) * T
    out, t_prev = [], 0.0
    for m, tn in zip(moves, t_new):
        out.append({"kind": "move", "dt": max(float(tn - t_prev), 1e-4),
                    "dx": m["dx"], "dy": m["dy"]})
        t_prev = float(tn)
    return out


def _choose(gestures, start, target, key):
    """Gesture whose approach length is closest to the needed distance."""
    D = math.hypot(target[0] - start[0], target[1] - start[1])
    best, bestd = None, None
    for g in gestures:
        moves = g[key] if isinstance(g, dict) else g
        L = float(np.hypot(*_net(moves)))
        if L < 2.0:
            continue
        score = abs(L - D)
        if bestd is None or score < bestd:
            best, bestd = g, score
    return best


def _click_records(start, target, click_gestures, profile=0.0):
    g = _choose(click_gestures, start, target, "approach")
    if g is not None:
        approach = _retarget_moves(g["approach"], start, target)
    else:
        approach = None
    if approach is None:
        approach = _straight_fallback(start, target)
    approach = _reprofile(approach, profile)
    down_dt = g["down_dt"] if g else 0.08
    up_dt = g["up_dt"] if g else 0.07
    hold = g["hold"] if g else []
    # a click holds in place; only a real drag moves during the press. Drop
    # tiny sampled hold jitter so the click lands exactly on the target.
    if float(np.hypot(*_net(hold))) < 6.0:
        hold = []
    mini = list(approach)
    mini.append({"kind": "left_down", "dt": down_dt, "dx": 0.0, "dy": 0.0})
    mini.extend(hold)
    mini.append({"kind": "left_up", "dt": up_dt, "dx": 0.0, "dy": 0.0})
    return mini


def _move_records(start, target, move_gestures, profile=0.0):
    g = _choose(move_gestures, start, target, None)
    moves = None
    if g is not None:
        moves = _retarget_moves(g, start, target)
    if moves is None:
        moves = _straight_fallback(start, target)
    return _reprofile(moves, profile)


def build_ops(model, tokenizer, device, actions, start_cursor=(0, 0), seed=0,
              temperature=0.9, top_k=None, pool_tokens=1600, profile=0.6):
    records = sample_records(model, tokenizer, device, pool_tokens, temperature,
                             top_k, seed)
    click_g = extract_click_gestures(records)
    move_g = extract_move_gestures(records)
    ops, cursor, t = [], (float(start_cursor[0]), float(start_cursor[1])), 0.0
    for a in actions:
        kind = a["type"]
        if kind in ("click", "move"):
            target = tuple(a.get("to", cursor))
            mini = (_click_records(cursor, target, click_g, profile) if kind == "click"
                    else _move_records(cursor, target, move_g, profile))
            sub = records_to_ops(mini, cursor, t)
            ops.extend(sub)
            if sub:
                t = sub[-1]["t"] + 0.01
            cursor = target
        elif kind == "pause":
            t += float(a["seconds"])
        else:
            raise ValueError(f"unsupported action for neural sampler: {kind}")
    ops.sort(key=lambda o: o["t"])
    return ops
