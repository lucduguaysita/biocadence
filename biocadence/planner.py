"""Turn high-level actions into a concrete, timed stream of primitive ops by
sampling the model.

An action list looks like:
    [
      {"type": "move",  "to": [x, y]},
      {"type": "click", "button": "left"},
      {"type": "type",  "text": "Hello, world."},
      {"type": "key",   "token": "Key.enter"},
      {"type": "pause", "seconds": 0.8},
    ]

build() returns a flat list of ops, each with an absolute time `t` (seconds
from the start of the plan):
    {"t": .., "op": "key_down",   "token": ".."}
    {"t": .., "op": "key_up",     "token": ".."}
    {"t": .., "op": "mouse_move", "x": .., "y": ..}
    {"t": .., "op": "mouse_down", "button": ".."}
    {"t": .., "op": "mouse_up",   "button": ".."}

The replay engine just walks this list, sleeping until each op's time.
Typing uses press-to-press latencies (so fast keys can overlap, like real
rollover), context-aware think-pauses, and optional typo-and-correct.
"""

import numpy as np

from .schema import (
    char_to_token, SENTENCE_ENDERS, CLAUSE_ENDERS,
)

# QWERTY neighbors used to pick a *plausible* wrong key for a typo.
_ADJ = {
    "q": "wa", "w": "qeas", "e": "wrsd", "r": "etdf", "t": "rygf",
    "y": "tuhg", "u": "yijh", "i": "uokj", "o": "iplk", "p": "ol",
    "a": "qwsz", "s": "awedxz", "d": "serfcx", "f": "drtgvc", "g": "ftyhbv",
    "h": "gyujnb", "j": "huikmn", "k": "jiolm", "l": "kop", "z": "asx",
    "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk",
}


def _context(prev_chars, is_first):
    if is_first:
        return "start"
    c1 = prev_chars[-1] if len(prev_chars) >= 1 else None
    c2 = prev_chars[-2] if len(prev_chars) >= 2 else None
    if c1 == " ":
        if c2 in SENTENCE_ENDERS:
            return "sentence"
        if c2 in CLAUSE_ENDERS:
            return "clause"
        return "word"
    if c1 == "\n":
        return "sentence"
    if c1 in SENTENCE_ENDERS:
        return "sentence"
    if c1 in CLAUSE_ENDERS:
        return "clause"
    return "mid"


def _minjerk(tau):
    # smooth 0->1 with zero velocity/accel at both ends
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


class Planner:
    def __init__(self, model, seed=None, move_rate=120.0,
                 typo_rate=None, cursor=(0, 0)):
        self.m = model
        self.rng = np.random.default_rng(seed)
        self.move_rate = move_rate
        self.cursor = (float(cursor[0]), float(cursor[1]))
        # correction behavior: learned from the model by default; typo_rate
        # overrides it (0 disables corrections, >0 forces single-key typos).
        ty = getattr(model, "p", {}).get("typing", {})
        if typo_rate is None:
            self.corr_prob = float(ty.get("corr_prob", 0.0))
            self.mean_run = float(ty.get("mean_run", 1.0))
        else:
            self.corr_prob = float(typo_rate)
            self.mean_run = 1.0

    # ---- public ----
    def build(self, actions):
        self.ops = []
        self.t = 0.0                 # virtual clock
        self.last_down_t = None      # last key press time (for press-to-press)
        self.prev_char = None        # last produced character
        self.history = []            # produced-char history for context
        self._first_key = True
        for a in actions:
            kind = a["type"]
            if kind == "type":
                self._type_text(a["text"])
            elif kind == "key":
                self._key(a["token"])
            elif kind == "move":
                self._move_to(a["to"])
            elif kind == "click":
                self._click(a.get("button", "left"),
                            a.get("to"), a.get("double", False))
            elif kind == "pause":
                self.t += float(a["seconds"])
            else:
                raise ValueError(f"unknown action type: {kind}")
        self.ops.sort(key=lambda o: o["t"])
        return self.ops

    @property
    def duration(self):
        return self.ops[-1]["t"] if getattr(self, "ops", None) else 0.0

    # ---- typing ----
    def _emit_key(self, token, context, produced_char):
        """Schedule one key using press-to-press flight + dwell."""
        flight = self.m.sample_flight(self.rng, self.prev_char, token, context)
        dwell = self.m.sample_dwell(self.rng, token)
        t_down = self.t if self.last_down_t is None else self.last_down_t + flight
        t_down = max(t_down, self.t)
        t_up = t_down + dwell
        self.ops.append({"t": t_down, "op": "key_down", "token": token})
        self.ops.append({"t": t_up, "op": "key_up", "token": token})
        self.last_down_t = t_down
        self.t = max(self.t, t_down)         # clock tracks presses
        self.prev_char = produced_char if produced_char is not None else self.prev_char
        self._first_key = False

    def _type_text(self, text):
        i, n = 0, len(text)
        while i < n:
            # correction: type a short run ending in a typo, backspace the run,
            # then retype it correctly. Run length and rate come from the model.
            if (self.corr_prob > 0 and text[i].strip()
                    and self.rng.random() < self.corr_prob):
                r = 1 + int(self.rng.poisson(max(self.mean_run - 1.0, 0.0)))
                r = max(1, min(r, n - i, 6))
                chunk = text[i:i + r]
                wrong = list(chunk)
                lc = chunk[-1].lower()
                if lc in _ADJ:
                    w = str(self.rng.choice(list(_ADJ[lc])))
                    wrong[-1] = w.upper() if chunk[-1].isupper() else w
                for ch in wrong:
                    ctx = _context(self.history, self._first_key)
                    self._emit_key(char_to_token(ch), ctx, ch)
                    self.history.append(ch)
                # a beat to notice, then backspace the whole run
                self.t = self.last_down_t + self.m.sample_flight(
                    self.rng, self.prev_char, "Key.backspace", "mid") + 0.15
                for _ in range(r):
                    self._emit_key("Key.backspace", "mid", None)
                    if self.history:
                        self.history.pop()
                for ch in chunk:
                    self._emit_key(char_to_token(ch), "mid", ch)
                    self.history.append(ch)
                i += r
            else:
                ch = text[i]
                ctx = _context(self.history, self._first_key)
                self._emit_key(char_to_token(ch), ctx, ch)
                self.history.append(ch)
                i += 1

    def _key(self, token):
        context = _context(self.history, self._first_key)
        self._emit_key(token, context, None)

    # ---- mouse ----
    def _click(self, button="left", to=None, double=False):
        if to is not None:
            self._move_to(to)
        self.t += 0.5 * self.m.sample_flight(self.rng, None, "click", "mid")  # settle
        self._press_click(button)
        if double:
            self.t += 0.06
            self._press_click(button)

    def _press_click(self, button):
        dur = self.m.sample_click_duration(self.rng)
        self.ops.append({"t": self.t, "op": "mouse_down", "button": button})
        self.ops.append({"t": self.t + dur, "op": "mouse_up", "button": button})
        self.t += dur

    def _move_to(self, target):
        start = np.array(self.cursor, dtype=float)
        end = np.array([float(target[0]), float(target[1])], dtype=float)
        D = float(np.hypot(*(end - start)))
        if D < 2.0:
            self.ops.append({"t": self.t, "op": "mouse_move",
                             "x": int(round(end[0])), "y": int(round(end[1]))})
            self.cursor = (end[0], end[1])
            return
        duration = self.m.sample_move_time(self.rng, D)
        overshoot = self.m.sample_overshoot(self.rng)
        axis = (end - start) / D
        if overshoot > 0:
            past = end + axis * (overshoot * D)
            self._bezier(self.t, duration * 0.82, start, past, D)
            self._bezier(self.t + duration * 0.82, duration * 0.18, past, end,
                         overshoot * D + 1.0, curve=False)
        else:
            self._bezier(self.t, duration, start, end, D)
        self.t += duration
        self.cursor = (end[0], end[1])

    def _bezier(self, t0, dur, start, end, D, curve=True):
        seg = end - start
        segD = float(np.hypot(*seg)) or 1.0
        axis = seg / segD
        normal = np.array([-axis[1], axis[0]])
        if curve:
            dev = self.m.sample_dev_ratio(self.rng) * D
            side = 1.0 if self.rng.random() < 0.5 else -1.0
            frac = float(self.rng.uniform(0.35, 0.65))
            ctrl = start + seg * frac + normal * dev * side
        else:
            ctrl = start + seg * 0.5
        n = max(2, int(dur * self.move_rate))
        tremor = self.m.p["mouse"]["tremor_px"]
        for i in range(1, n + 1):
            tau = i / n
            s = _minjerk(tau)
            p = (1 - s) ** 2 * start + 2 * (1 - s) * s * ctrl + s ** 2 * end
            envelope = 1.0 - abs(2 * tau - 1)     # 0 at ends, 1 mid: precise landings
            p = p + normal * self.rng.normal(0.0, tremor) * envelope
            self.ops.append({"t": t0 + tau * dur, "op": "mouse_move",
                             "x": int(round(p[0])), "y": int(round(p[1]))})
