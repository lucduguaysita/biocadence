"""Turn a raw event list into features the model learns from.

Three families of features come out of extract():

  keystrokes : one record per typed key with
               dwell   (how long the key was held)
               flight  (press-to-press latency from the previous key)
               context (what boundary precedes it: start/word/clause/
                        sentence/mid/idle) -- this is the hook the
                        think-time model hangs on.
  clicks     : one record per mouse click (down..up duration + interval).
  moves      : one record per mouse movement segment, describing its
               geometry (distance, duration, curvature, overshoot).
"""

import math

import numpy as np

from .schema import (
    KEY_PRESS, KEY_RELEASE, MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP,
    MODIFIER_TOKENS, SENTENCE_ENDERS, CLAUSE_ENDERS, token_to_char,
)

IDLE_RESET = 2.0   # a gap longer than this is "stepped away", not think-time
MIN_MOVE_DIST = 12.0  # px; ignore movement segments shorter than this


def _classify_context(last_chars, flight):
    """Label the gap that precedes the current key from recent typed chars."""
    if flight is None:
        return "start"
    if flight > IDLE_RESET:
        return "idle"
    c_prev = last_chars[-1] if len(last_chars) >= 1 else None
    c_prev2 = last_chars[-2] if len(last_chars) >= 2 else None
    if c_prev == " ":
        if c_prev2 in SENTENCE_ENDERS:
            return "sentence"
        if c_prev2 in CLAUSE_ENDERS:
            return "clause"
        return "word"
    if c_prev == "\n":
        return "sentence"
    if c_prev in SENTENCE_ENDERS:
        return "sentence"
    if c_prev in CLAUSE_ENDERS:
        return "clause"
    return "mid"


def _extract_keystrokes(events):
    keystrokes = []
    pending = {}           # token -> list of records awaiting their release
    last_chars = []        # produced-character history
    prev_press_t = None

    for ev in events:
        et = ev["type"]
        if et == KEY_PRESS:
            token = ev["key"]
            if token in MODIFIER_TOKENS:
                continue   # tracked as state via the char case of the token
            t = ev["t"]
            flight = None if prev_press_t is None else (t - prev_press_t)
            rec = {
                "token": token,
                "prev": last_chars_token(last_chars),
                "press_t": t,
                "dwell": None,
                "flight": flight,
                "context": _classify_context(last_chars, flight),
            }
            keystrokes.append(rec)
            pending.setdefault(token, []).append(rec)
            prev_press_t = t
            ch = token_to_char(token)
            if ch is not None:
                last_chars.append(ch)
        elif et == KEY_RELEASE:
            token = ev["key"]
            if token in MODIFIER_TOKENS:
                continue
            q = pending.get(token)
            if q:
                rec = q.pop(0)
                rec["dwell"] = ev["t"] - rec["press_t"]

    for rec in keystrokes:
        rec.pop("press_t", None)
    return keystrokes


def last_chars_token(last_chars):
    """The previous produced character, used as the digraph 'from' key."""
    return last_chars[-1] if last_chars else None


def _extract_clicks(events):
    clicks = []
    pending = {}
    last_down_t = None
    for ev in events:
        if ev["type"] == MOUSE_DOWN:
            pending.setdefault(ev["button"], []).append(ev["t"])
        elif ev["type"] == MOUSE_UP:
            q = pending.get(ev["button"])
            if q:
                down_t = q.pop(0)
                gap = None if last_down_t is None else (down_t - last_down_t)
                clicks.append({
                    "button": ev["button"],
                    "duration": ev["t"] - down_t,
                    "gap": gap,
                })
                last_down_t = down_t
    return clicks


def _segment_moves(events, move_gap):
    """Split mouse_move samples into contiguous movement segments."""
    segs = []
    cur = []
    last_t = None
    for ev in events:
        if ev["type"] != MOUSE_MOVE:
            # a click or key breaks a movement segment
            if ev["type"] in (MOUSE_DOWN, MOUSE_UP) and cur:
                segs.append(cur)
                cur = []
                last_t = None
            continue
        t = ev["t"]
        if last_t is not None and (t - last_t) > move_gap and cur:
            segs.append(cur)
            cur = []
        cur.append((t, ev["x"], ev["y"]))
        last_t = t
    if cur:
        segs.append(cur)
    return segs


def _segment_geometry(seg):
    if len(seg) < 3:
        return None
    arr = np.array(seg, dtype=float)
    times = arr[:, 0]
    pts = arr[:, 1:3]
    start, end = pts[0], pts[-1]
    D = float(np.hypot(*(end - start)))
    if D < MIN_MOVE_DIST:
        return None
    axis = (end - start) / D
    rel = pts - start
    along = rel @ axis
    perp = axis[0] * rel[:, 1] - axis[1] * rel[:, 0]   # signed cross product
    diffs = np.diff(pts, axis=0)
    path_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    duration = float(times[-1] - times[0])
    if duration <= 0:
        return None
    return {
        "distance": D,
        "duration": duration,
        "dev_ratio": float(np.max(np.abs(perp)) / D),
        "overshoot": float(max(0.0, (np.max(along) - D)) / D),
        "path_ratio": float(path_len / D),
        "n": int(len(seg)),
    }


def extract(events, move_gap=0.15):
    """Full feature extraction. Returns a dict of keystrokes/clicks/moves/meta."""
    events = sorted(events, key=lambda e: e["t"])
    keystrokes = _extract_keystrokes(events)
    clicks = _extract_clicks(events)
    moves = []
    for seg in _segment_moves(events, move_gap):
        g = _segment_geometry(seg)
        if g is not None:
            moves.append(g)
    duration = (events[-1]["t"] - events[0]["t"]) if events else 0.0
    return {
        "keystrokes": keystrokes,
        "clicks": clicks,
        "moves": moves,
        "meta": {
            "duration_s": duration,
            "n_events": len(events),
            "n_keys": len(keystrokes),
            "n_clicks": len(clicks),
            "n_move_segments": len(moves),
        },
    }
