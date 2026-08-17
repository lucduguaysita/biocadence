"""Synthetic trace generator.

Produces a raw event list in the same schema a real recorder would, with
believable human structure (per-key dwell, fluent press-to-press flight,
context think-pauses, uppercase via shift, curved mouse moves with occasional
overshoot, and clicks). It exists so the whole extract -> fit -> sample
pipeline can be exercised and validated without a physical recording. It is
NOT the model; it is a stand-in for "you, typing", used for demos and tests.
"""

import numpy as np

from .schema import (
    KEY_PRESS, KEY_RELEASE, MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP,
    SENTENCE_ENDERS, CLAUSE_ENDERS,
)

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. Pack my box with five "
    "dozen liquor jugs! How razorback jumping frogs can level six piqued "
    "gymnasts. Amazingly, few discotheques provide jukeboxes."
)


def _lognormal(rng, median, sigma):
    return float(np.exp(rng.normal(np.log(median), sigma)))


def _minjerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


def _emit_move(events, rng, t0, start, end, duration, rate=120.0,
               tremor=0.8, overshoot=0.0):
    seg = np.array(end, float) - np.array(start, float)
    D = float(np.hypot(*seg)) or 1.0
    axis = seg / D
    normal = np.array([-axis[1], axis[0]])
    target = np.array(end, float)
    if overshoot > 0:
        target = np.array(end, float) + axis * (overshoot * D)
    dev = _lognormal(rng, 0.06, 0.5) * D
    side = 1.0 if rng.random() < 0.5 else -1.0
    ctrl = np.array(start, float) + (target - np.array(start, float)) * \
        rng.uniform(0.35, 0.65) + normal * dev * side
    n = max(2, int(duration * rate))
    p0 = np.array(start, float)
    for i in range(1, n + 1):
        tau = i / n
        s = _minjerk(tau)
        p = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s ** 2 * target
        env = 1.0 - abs(2 * tau - 1)
        p = p + normal * rng.normal(0.0, tremor) * env
        events.append({"t": t0 + tau * duration, "type": MOUSE_MOVE,
                       "x": int(round(p[0])), "y": int(round(p[1]))})
    t_end = t0 + duration
    if overshoot > 0:
        # short corrective move back onto the target
        cdur = duration * 0.18
        for i in range(1, 6):
            tau = i / 5
            s = _minjerk(tau)
            p = (1 - s) * target + s * np.array(end, float)
            events.append({"t": t_end + tau * cdur, "type": MOUSE_MOVE,
                           "x": int(round(p[0])), "y": int(round(p[1]))})
        t_end += cdur
    return t_end


def synth_trace(seed=0, text=DEFAULT_TEXT, n_mouse=14, screen=(1920, 1080),
                dwell_k=1.0, flight_k=1.0, mouse_k=1.0):
    """dwell_k / flight_k / mouse_k scale the timing medians, so different
    values simulate a different person (used to test fingerprint separation)."""
    rng = np.random.default_rng(seed)
    events = []
    t = 0.30
    history = []
    prev_down = None

    for ch in text:
        # context pause before this character
        c1 = history[-1] if history else None
        c2 = history[-2] if len(history) >= 2 else None
        pause = 0.0
        if c1 == " " and c2 in SENTENCE_ENDERS:
            pause = _lognormal(rng, 0.75, 0.45)              # between sentences
        elif c1 == " " and c2 in CLAUSE_ENDERS:
            if rng.random() < 0.6:
                pause = _lognormal(rng, 0.30, 0.5)           # after a clause
        elif c1 == " ":
            if rng.random() < 0.45:
                pause = _lognormal(rng, 0.22, 0.5)           # between words
        elif rng.random() < 0.05:
            pause = _lognormal(rng, 0.28, 0.6)               # occasional hitch

        flight = _lognormal(rng, 0.125 * flight_k, 0.30)
        t_down = t if prev_down is None else prev_down + flight + pause
        dwell = _lognormal(rng, 0.085 * dwell_k, 0.28)

        if ch == " ":
            token = "Key.space"
        elif ch == "\n":
            token = "Key.enter"
        else:
            token = ch

        with_shift = ch.isalpha() and ch.isupper()
        if with_shift:
            events.append({"t": t_down - 0.03, "type": KEY_PRESS,
                           "key": "Key.shift", "vk": 160})
        events.append({"t": t_down, "type": KEY_PRESS, "key": token, "vk": None})
        events.append({"t": t_down + dwell, "type": KEY_RELEASE, "key": token, "vk": None})
        if with_shift:
            events.append({"t": t_down + dwell + 0.02, "type": KEY_RELEASE,
                           "key": "Key.shift", "vk": 160})

        prev_down = t_down
        t = t_down
        history.append(ch)

    # mouse: alternate move + click a few times after the typing
    cursor = (rng.integers(200, screen[0] - 200), rng.integers(200, screen[1] - 200))
    t += 0.8
    for _ in range(n_mouse):
        target = (int(rng.integers(50, screen[0] - 50)),
                  int(rng.integers(50, screen[1] - 50)))
        D = float(np.hypot(target[0] - cursor[0], target[1] - cursor[1]))
        duration = mouse_k * (0.10 + 0.020 * (D ** 0.5) * _lognormal(rng, 1.0, 0.18))
        overshoot = _lognormal(rng, 0.05, 0.4) if rng.random() < 0.35 else 0.0
        t = _emit_move(events, rng, t, cursor, target, duration, overshoot=overshoot)
        cursor = target
        # a short settle, then a click
        t += _lognormal(rng, 0.12, 0.4)
        cdur = _lognormal(rng, 0.07 * mouse_k, 0.3)
        events.append({"t": t, "type": MOUSE_DOWN, "button": "left",
                       "x": target[0], "y": target[1]})
        events.append({"t": t + cdur, "type": MOUSE_UP, "button": "left",
                       "x": target[0], "y": target[1]})
        t += cdur + _lognormal(rng, 0.9, 0.5)

    events.sort(key=lambda e: e["t"])
    return events
