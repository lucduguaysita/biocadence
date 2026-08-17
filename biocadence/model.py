"""The generative style model: fit timing/motion distributions from features,
sample from them, and save/load as portable JSON.

Design choices worth knowing:

* Dwell and flight are modeled as log-normal (positive, right-skewed timings).
  Specific keys/digraphs back off to a coarse key class, then to a global
  distribution, so unseen characters still get a plausible timing.

* Think-time is modeled as an ADDITIVE pause on top of the fluent flight:
      flight = fluent_base + (pause if a pause is triggered)
  Each context (mid-word / after-word / after-clause / after-sentence) has its
  own P(pause) and its own pause magnitude distribution. That is what gives
  generated typing a human burst-and-hesitate rhythm instead of a metronome.

* Mouse movement time follows a Fitts-style fit, duration ~ a + b*sqrt(dist),
  with log-normal noise; curvature, overshoot and path inefficiency are
  sampled from fitted distributions and realized by the planner.
"""

import numpy as np

from .schema import key_class, key_class as _kc

SEP = "\x1f"          # separator for composite dict keys
MIN_N = 4               # min samples before trusting a specific bucket
FENCE_Q = 0.95          # flights above this quantile of fluent are "paused"

# sane clamps (seconds) so a fat tail never emits an absurd value
DWELL_CLAMP = (0.012, 0.60)
FLUENT_CLAMP = (0.010, 2.0)
PAUSE_CLAMP = (0.05, 12.0)
CLICK_CLAMP = (0.02, 0.60)


def fit_lognorm(values, default_sigma=0.20):
    v = np.asarray([x for x in values if x is not None and x > 0], dtype=float)
    if v.size == 0:
        return None
    logs = np.log(v)
    mu = float(np.mean(logs))
    sigma = float(np.std(logs)) if v.size > 1 else default_sigma
    return {"mu": mu, "sigma": max(sigma, 1e-3), "n": int(v.size)}


def sample_lognorm(rng, params, clamp=None):
    x = float(np.exp(rng.normal(params["mu"], params["sigma"])))
    if clamp is not None:
        x = min(max(x, clamp[0]), clamp[1])
    return x


def _group_fit(items, key_fn, val_fn, default_sigma=0.20):
    buckets = {}
    for it in items:
        val = val_fn(it)
        if val is None or val <= 0:
            continue
        buckets.setdefault(key_fn(it), []).append(val)
    return {k: fit_lognorm(v, default_sigma) for k, v in buckets.items()
            if fit_lognorm(v, default_sigma) is not None}


class Model:
    def __init__(self, params):
        self.p = params

    # ---------- fitting ----------
    @classmethod
    def fit(cls, feats):
        if isinstance(feats, dict):
            feats = [feats]
        keys = [k for f in feats for k in f["keystrokes"]]
        clicks = [c for f in feats for c in f["clicks"]]
        moves = [m for f in feats for m in f["moves"]]

        p = {"version": "0.1.0"}
        p["dwell"] = cls._fit_dwell(keys)
        fluent_median = cls._fit_flight(keys, p)   # fills p["flight"], returns median
        cls._fit_pause(keys, p, fluent_median)
        p["click"] = cls._fit_click(clicks)
        p["mouse"] = cls._fit_mouse(moves)
        p["typing"] = cls._fit_typing(keys)
        p["meta"] = {
            "n_keys": len(keys),
            "n_clicks": len(clicks),
            "n_move_segments": len(moves),
        }
        return cls(p)

    @staticmethod
    def _fit_typing(keys):
        """How the user corrects: how often they backspace and how many in a
        row, so replay can reproduce the same rate of typo-and-correct."""
        n = len(keys)
        tokens = [k["token"] for k in keys]
        n_bs = sum(1 for t in tokens if t == "Key.backspace")
        n_char = sum(1 for t in tokens if len(t) == 1 or t == "Key.space")
        runs, i = [], 0
        while i < len(tokens):
            if tokens[i] == "Key.backspace":
                j = i
                while j < len(tokens) and tokens[j] == "Key.backspace":
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
        mean_run = float(np.mean(runs)) if runs else 1.0
        corr_prob = float(min(len(runs) / n_char, 0.25)) if n_char else 0.0
        return {"backspace_rate": float(n_bs / n) if n else 0.0,
                "mean_run": max(mean_run, 1.0), "corr_prob": corr_prob,
                "n_backspace": n_bs}

    @staticmethod
    def _fit_dwell(keys):
        return {
            "by_token": _group_fit(keys, lambda k: k["token"], lambda k: k["dwell"]),
            "by_class": _group_fit(keys, lambda k: key_class(k["token"]), lambda k: k["dwell"]),
            "global": fit_lognorm([k["dwell"] for k in keys]) or {"mu": np.log(0.09), "sigma": 0.3, "n": 0},
        }

    @staticmethod
    def _fit_flight(keys, p):
        mid = [k["flight"] for k in keys
               if k["context"] == "mid" and k["flight"] and k["flight"] > 0]
        fence = float(np.quantile(mid, FENCE_Q)) if len(mid) >= 5 else 0.40
        fluent_ctx = ("mid", "word", "clause", "sentence")
        fluent_keys = [k for k in keys
                       if k["context"] in fluent_ctx and k["flight"]
                       and 0 < k["flight"] <= fence]
        fluent_vals = [k["flight"] for k in fluent_keys]
        fluent_median = float(np.median(fluent_vals)) if fluent_vals else 0.13

        def dg(k):
            return f"{k['prev']}{SEP}{k['token']}"

        def dg_class(k):
            return f"{_kc(str(k['prev'])) if k['prev'] else '^'}{SEP}{key_class(k['token'])}"

        p["flight"] = {
            "by_digraph": _group_fit(fluent_keys, dg, lambda k: k["flight"]),
            "by_class": _group_fit(fluent_keys, dg_class, lambda k: k["flight"]),
            "global": fit_lognorm(fluent_vals) or {"mu": np.log(0.13), "sigma": 0.3, "n": 0},
            "fence": fence,
            "fluent_median": fluent_median,
        }
        return fluent_median

    @staticmethod
    def _fit_pause(keys, p, fluent_median):
        fence = p["flight"]["fence"]
        contexts = ("mid", "word", "clause", "sentence")
        prob, dist = {}, {}
        all_extras = []
        for ctx in contexts:
            obs = [k["flight"] for k in keys
                   if k["context"] == ctx and k["flight"] and k["flight"] > 0]
            if not obs:
                prob[ctx] = 0.0
                continue
            paused = [x for x in obs if x > fence]
            prob[ctx] = (len(paused) + 0.5) / (len(obs) + 1.0)
            extras = [max(x - fluent_median, 1e-3) for x in paused]
            all_extras.extend(extras)
            d = fit_lognorm(extras)
            if d is not None and d["n"] >= MIN_N:
                dist[ctx] = d
        p["pause"] = {
            "prob": prob,
            "dist": dist,
            "global": fit_lognorm(all_extras) or {"mu": np.log(0.4), "sigma": 0.5, "n": 0},
        }

    @staticmethod
    def _fit_click(clicks):
        durs = [c["duration"] for c in clicks]
        gaps = [c["gap"] for c in clicks if c["gap"]]
        dbl = [g for g in gaps if g < 0.30]
        return {
            "duration": fit_lognorm(durs) or {"mu": np.log(0.07), "sigma": 0.3, "n": 0},
            "gap": fit_lognorm(gaps) or {"mu": np.log(1.2), "sigma": 0.6, "n": 0},
            "double_click_prob": (len(dbl) / len(gaps)) if gaps else 0.0,
        }

    @staticmethod
    def _fit_mouse(moves):
        if len(moves) >= 2:
            D = np.array([m["distance"] for m in moves], dtype=float)
            T = np.array([m["duration"] for m in moves], dtype=float)
            x = np.sqrt(D)
            b, a = np.polyfit(x, T, 1)              # T ~ a + b*sqrt(D)
            pred = np.clip(a + b * x, 0.02, None)
            logres = np.log(np.clip(T, 1e-3, None)) - np.log(pred)
            sigma = float(np.std(logres)) if len(moves) > 2 else 0.25
        else:
            a, b, sigma = 0.10, 0.020, 0.25
        dev = fit_lognorm([m["dev_ratio"] for m in moves]) or {"mu": np.log(0.06), "sigma": 0.5, "n": 0}
        pr = fit_lognorm([max(m["path_ratio"], 1.0) for m in moves]) or {"mu": np.log(1.05), "sigma": 0.15, "n": 0}
        over_vals = [m["overshoot"] for m in moves if m["overshoot"] > 0.02]
        over_prob = (len(over_vals) / len(moves)) if moves else 0.0
        over = fit_lognorm(over_vals) or {"mu": np.log(0.05), "sigma": 0.4, "n": 0}
        return {
            "move_time": {"a": float(a), "b": float(b), "sigma": max(sigma, 1e-3)},
            "dev_ratio": dev,
            "path_ratio": pr,
            "overshoot_prob": over_prob,
            "overshoot": over,
            "tremor_px": 0.7,
        }

    # ---------- sampling ----------
    def _dwell_params(self, token):
        d = self.p["dwell"]
        b = d["by_token"].get(token)
        if b and b["n"] >= MIN_N:
            return b
        b = d["by_class"].get(key_class(token))
        if b and b["n"] >= MIN_N:
            return b
        return d["global"]

    def _flight_params(self, prev_char, token):
        f = self.p["flight"]
        b = f["by_digraph"].get(f"{prev_char}{SEP}{token}")
        if b and b["n"] >= MIN_N:
            return b
        pc = _kc(str(prev_char)) if prev_char else "^"
        b = f["by_class"].get(f"{pc}{SEP}{key_class(token)}")
        if b and b["n"] >= MIN_N:
            return b
        return f["global"]

    def sample_dwell(self, rng, token):
        return sample_lognorm(rng, self._dwell_params(token), DWELL_CLAMP)

    def sample_flight(self, rng, prev_char, token, context):
        base = sample_lognorm(rng, self._flight_params(prev_char, token), FLUENT_CLAMP)
        if context in ("start", "idle"):
            return base
        prob = self.p["pause"]["prob"].get(context, 0.0)
        if rng.random() < prob:
            pd = self.p["pause"]["dist"].get(context) or self.p["pause"]["global"]
            base += sample_lognorm(rng, pd, PAUSE_CLAMP)
        return base

    def sample_click_duration(self, rng):
        return sample_lognorm(rng, self.p["click"]["duration"], CLICK_CLAMP)

    def sample_move_time(self, rng, distance):
        mt = self.p["mouse"]["move_time"]
        pred = max(mt["a"] + mt["b"] * (distance ** 0.5), 0.02)
        return float(pred * np.exp(rng.normal(0.0, mt["sigma"])))

    def sample_dev_ratio(self, rng):
        return sample_lognorm(rng, self.p["mouse"]["dev_ratio"], (0.0, 0.6))

    def sample_path_ratio(self, rng):
        return max(sample_lognorm(rng, self.p["mouse"]["path_ratio"], (1.0, 3.0)), 1.0)

    def sample_overshoot(self, rng):
        if rng.random() < self.p["mouse"]["overshoot_prob"]:
            return sample_lognorm(rng, self.p["mouse"]["overshoot"], (0.0, 0.5))
        return 0.0

    # ---------- io ----------
    def save(self, path):
        from .schema import dump_json
        dump_json(path, self.p)

    @classmethod
    def load(cls, path):
        from .schema import load_json
        return cls(load_json(path))
