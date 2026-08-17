"""Tokenize the pointer channel of a recording into an integer stream, and back.

Grammar. The stream is BOS, then a series of records, then EOS. Each record is
either a move or an event:

  move  record : [DT, MOVE, DX, DY]     (4 tokens)
  event record : [DT, EVENT]            (2 tokens)

DT is the time since the previous pointer event (quantized, log-spaced). DX and
DY are the change in cursor position for a move (quantized, signed, log-spaced
magnitude with a dedicated zero bin). EVENT is one of the button/scroll tokens.
Because prev-to-cur deltas are used, the model is translation invariant; the
sampler reintegrates deltas from a chosen start point.

Keystrokes are intentionally ignored here; this models the pointer only.
"""

import numpy as np

from ..schema import MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP, SCROLL

# ---- fixed vocabulary ids ----
PAD, BOS, EOS, MOVE = 0, 1, 2, 3
EVENTS = ["left_down", "left_up", "right_down", "right_up",
          "middle_down", "middle_up", "scroll_up", "scroll_down"]
EVENT_BASE = 4                       # left_down=4 .. scroll_down=11
EVENT_ID = {name: EVENT_BASE + i for i, name in enumerate(EVENTS)}
ID_EVENT = {v: k for k, v in EVENT_ID.items()}
SPECIAL_BASE = EVENT_BASE + len(EVENTS)   # first data-bin id = 12

_BTN_DOWN = {"left": "left_down", "right": "right_down", "middle": "middle_down"}
_BTN_UP = {"left": "left_up", "right": "right_up", "middle": "middle_up"}


class Tokenizer:
    def __init__(self, n_dt=32, n_mag=31, dt_min=0.004, dt_max=1.0, max_d=900.0):
        self.n_dt = n_dt
        self.n_mag = n_mag
        self.n_axis = 2 * n_mag + 1           # signed bins per axis, zero in the middle
        self.dt_min, self.dt_max, self.max_d = dt_min, dt_max, max_d
        self.dt_centers = np.geomspace(dt_min, dt_max, n_dt)
        self.mag_centers = np.geomspace(1.0, max_d, n_mag)
        self.base_dt = SPECIAL_BASE
        self.base_dx = self.base_dt + n_dt
        self.base_dy = self.base_dx + self.n_axis
        self.vocab_size = self.base_dy + self.n_axis

    # ---- config io ----
    def config(self):
        return {"n_dt": self.n_dt, "n_mag": self.n_mag, "dt_min": self.dt_min,
                "dt_max": self.dt_max, "max_d": self.max_d}

    @classmethod
    def from_config(cls, c):
        return cls(**c)

    # ---- scalar quantization ----
    def _dt_bin(self, dt):
        dt = min(max(dt, self.dt_min), self.dt_max)
        f = np.log(dt / self.dt_min) / np.log(self.dt_max / self.dt_min)
        return int(round(f * (self.n_dt - 1)))

    def _dt_val(self, b):
        return float(self.dt_centers[min(max(b, 0), self.n_dt - 1)])

    def _axis_bin(self, v):
        """Signed local bin in [0, n_axis-1], zero at index n_mag."""
        if abs(v) < 0.5:
            return self.n_mag
        m = min(abs(v), self.max_d)
        f = np.log(m) / np.log(self.max_d)
        idx = int(round(f * (self.n_mag - 1)))        # 0..n_mag-1
        return self.n_mag + (idx + 1) if v > 0 else self.n_mag - (idx + 1)

    def _axis_val(self, b):
        d = b - self.n_mag
        if d == 0:
            return 0.0
        mag = float(self.mag_centers[min(abs(d) - 1, self.n_mag - 1)])
        return mag if d > 0 else -mag

    # ---- raw events -> pointer records ----
    def events_to_records(self, events):
        events = sorted(events, key=lambda e: e["t"])
        recs = []
        last_t = None
        last_xy = None
        for ev in events:
            et = ev["type"]
            if et == MOUSE_MOVE:
                x, y = ev["x"], ev["y"]
                if last_xy is None:
                    last_xy = (x, y)
                    last_t = ev["t"] if last_t is None else last_t
                dx, dy = x - last_xy[0], y - last_xy[1]
                if dx == 0 and dy == 0:
                    continue
                dt = 0.008 if last_t is None else ev["t"] - last_t
                recs.append(("move", dt, dx, dy))
                last_xy = (x, y)
                last_t = ev["t"]
            elif et in (MOUSE_DOWN, MOUSE_UP):
                kind = (_BTN_DOWN if et == MOUSE_DOWN else _BTN_UP).get(ev["button"])
                if kind is None:
                    continue
                dt = 0.05 if last_t is None else ev["t"] - last_t
                recs.append((kind, dt, 0, 0))
                last_xy = (ev["x"], ev["y"])
                last_t = ev["t"]
            elif et == SCROLL:
                kind = "scroll_up" if ev.get("dy", 0) >= 0 else "scroll_down"
                dt = 0.05 if last_t is None else ev["t"] - last_t
                recs.append((kind, dt, 0, 0))
                last_t = ev["t"]
        return recs

    # ---- records <-> tokens ----
    def encode_records(self, recs, add_bos_eos=True):
        toks = [BOS] if add_bos_eos else []
        for kind, dt, dx, dy in recs:
            toks.append(self.base_dt + self._dt_bin(dt))
            if kind == "move":
                toks.append(MOVE)
                toks.append(self.base_dx + self._axis_bin(dx))
                toks.append(self.base_dy + self._axis_bin(dy))
            else:
                toks.append(EVENT_ID[kind])
        if add_bos_eos:
            toks.append(EOS)
        return toks

    def encode(self, events, add_bos_eos=True):
        return self.encode_records(self.events_to_records(events), add_bos_eos)

    def decode(self, toks):
        """Tokens -> list of records {kind, dt, dx, dy}. Tolerant of truncation."""
        recs = []
        i, n = 0, len(toks)
        while i < n:
            t = toks[i]
            if t in (PAD, BOS, EOS):
                i += 1
                continue
            if self.base_dt <= t < self.base_dt + self.n_dt:
                dt = self._dt_val(t - self.base_dt)
                if i + 1 >= n:
                    break
                nxt = toks[i + 1]
                if nxt == MOVE:
                    if i + 3 >= n:
                        break
                    dxb, dyb = toks[i + 2], toks[i + 3]
                    if not (self.base_dx <= dxb < self.base_dx + self.n_axis
                            and self.base_dy <= dyb < self.base_dy + self.n_axis):
                        i += 1
                        continue
                    recs.append({"kind": "move", "dt": dt,
                                 "dx": self._axis_val(dxb - self.base_dx),
                                 "dy": self._axis_val(dyb - self.base_dy)})
                    i += 4
                elif nxt in ID_EVENT:
                    recs.append({"kind": ID_EVENT[nxt], "dt": dt, "dx": 0.0, "dy": 0.0})
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        return recs


def records_to_ops(recs, start_xy=(0, 0), t0=0.0):
    """Reintegrate decoded records into an absolute op stream for replay_win."""
    ops = []
    x, y = float(start_xy[0]), float(start_xy[1])
    t = float(t0)
    btn = {"left_down": ("left", "mouse_down"), "left_up": ("left", "mouse_up"),
           "right_down": ("right", "mouse_down"), "right_up": ("right", "mouse_up"),
           "middle_down": ("middle", "mouse_down"), "middle_up": ("middle", "mouse_up")}
    for r in recs:
        t += r["dt"]
        k = r["kind"]
        if k == "move":
            x += r["dx"]
            y += r["dy"]
            ops.append({"t": t, "op": "mouse_move", "x": int(round(x)), "y": int(round(y))})
        elif k in btn:
            button, op = btn[k]
            ops.append({"t": t, "op": op, "button": button})
        elif k in ("scroll_up", "scroll_down"):
            ops.append({"t": t, "op": "scroll", "dy": 1 if k == "scroll_up" else -1})
    return ops
