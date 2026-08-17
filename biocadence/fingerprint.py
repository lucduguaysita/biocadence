"""Behavioral fingerprint: a stable, repeatable signature of how you type and
move, plus a matcher that scores whether a new sample is the same person.

It is derived from the STATISTICAL model's parameters (dwell, flight, pause,
click and mouse-motion summaries), which are stable per person across sessions.
It is deliberately not built from the neural weights, whose random
initialization would never reproduce.

Matching compares each feature in units of its own estimation noise: the
standard error of that feature is estimated by bootstrapping the recording, so
a feature measured from little data (which naturally wobbles session to
session) is automatically down-weighted, while a feature measured from lots of
data must really differ to count. The per-feature z-scores combine into one
distance, which maps to a match percentage. A certainty level reflects how much
data each sample had and how many features they share.

This is probabilistic behavioral matching, not a secure or cryptographic
identifier; treat it as one soft signal, never as the sole basis for a
security decision.
"""

import numpy as np

from . import features as _features
from .model import Model

# Per-feature session-to-session noise floor for the SAME person: how much this
# feature is expected to drift between two honest recordings. Small for stable
# keystroke timing; large for mouse stats that depend on where the targets were.
# Combined with the bootstrap standard error, it sets how much a feature may
# differ before it counts against a match. Only these features are compared.
FLOOR = {
    "dwell_ms": 6.0, "dwell_letter_ms": 6.0, "dwell_space_ms": 8.0,
    "flight_ms": 10.0, "flight_iqr_ms": 12.0, "wpm": 6.0,
    "pause_word": 0.06, "pause_clause": 0.07, "pause_sentence": 0.08, "pause_mid": 0.03,
    "pause_mag_ms": 90.0, "backspace_rate": 0.02,
    "mouse_a_ms": 70.0, "mouse_b_ms": 7.0, "dev_ratio": 0.035, "overshoot_p": 0.18,
    "path_ratio": 0.09, "click_ms": 14.0, "double_click_p": 0.16, "scroll_rate": 16.0,
}
K = 1.5           # width of the distance -> similarity falloff
BOOT = 24         # bootstrap resamples for standard-error estimation


def _med_ms(p):
    return float(np.exp(p["mu"]) * 1000.0)


def _compute(feats, n_scroll, dur_min):
    """Build the raw feature dict from an (possibly resampled) feature set."""
    ks, moves, clicks = feats["keystrokes"], feats["moves"], feats["clicks"]
    m = Model.fit(feats).p
    v = {}
    if len(ks) >= 20:
        v["dwell_ms"] = _med_ms(m["dwell"]["global"])
        for cls, key in (("letter", "dwell_letter_ms"), ("space", "dwell_space_ms")):
            b = m["dwell"]["by_class"].get(cls)
            if b and b["n"] >= 8:
                v[key] = _med_ms(b)
        v["flight_ms"] = _med_ms(m["flight"]["global"])
        fl = [k["flight"] for k in ks if k["flight"] and 0 < k["flight"] < 2]
        if len(fl) >= 10:
            v["flight_iqr_ms"] = float((np.percentile(fl, 75) - np.percentile(fl, 25)) * 1000)
        nchars = sum(1 for k in ks if len(k["token"]) == 1 or k["token"] == "Key.space")
        v["wpm"] = (nchars / 5.0) / dur_min
        for ctx, key in (("word", "pause_word"), ("clause", "pause_clause"),
                         ("sentence", "pause_sentence"), ("mid", "pause_mid")):
            if ctx in m["pause"]["prob"]:
                v[key] = float(m["pause"]["prob"][ctx])
        v["pause_mag_ms"] = _med_ms(m["pause"]["global"])
        v["backspace_rate"] = sum(1 for k in ks if k["token"] == "Key.backspace") / max(len(ks), 1)
    if len(moves) >= 5:
        mt = m["mouse"]["move_time"]
        v["mouse_a_ms"] = mt["a"] * 1000.0
        v["mouse_b_ms"] = mt["b"] * 1000.0
        v["dev_ratio"] = float(np.exp(m["mouse"]["dev_ratio"]["mu"]))
        v["overshoot_p"] = float(m["mouse"]["overshoot_prob"])
        v["path_ratio"] = float(np.exp(m["mouse"]["path_ratio"]["mu"]))
    if len(clicks) >= 3:
        v["click_ms"] = _med_ms(m["click"]["duration"])
        v["double_click_p"] = float(m["click"]["double_click_prob"])
    if n_scroll >= 3:
        v["scroll_rate"] = n_scroll / dur_min
    return v


def _prep(events):
    feats = _features.extract(events)
    n_scroll = sum(1 for e in events if e.get("type") == "scroll")
    dur_min = max(feats["meta"]["duration_s"] / 60.0, 1e-6)
    return feats, n_scroll, dur_min


def _bootstrap_se(feats, n_scroll, dur_min, seed=0):
    rng = np.random.default_rng(seed)
    ks, moves, clicks = feats["keystrokes"], feats["moves"], feats["clicks"]
    runs = []
    for _ in range(BOOT):
        def resample(lst):
            return [lst[i] for i in rng.integers(0, len(lst), len(lst))] if lst else []
        fb = {"keystrokes": resample(ks), "moves": resample(moves), "clicks": resample(clicks)}
        try:
            runs.append(_compute(fb, n_scroll, dur_min))
        except Exception:
            pass
    se = {}
    allkeys = set().union(*[r.keys() for r in runs]) if runs else set()
    for k in allkeys:
        vals = [r[k] for r in runs if k in r]
        if len(vals) >= 3:
            se[k] = float(np.std(vals))
    return se


def vector(events):
    feats, n_scroll, dur_min = _prep(events)
    v = _compute(feats, n_scroll, dur_min)
    return {"vector": v, "n_keys": len(feats["keystrokes"]),
            "n_mouse": len(feats["moves"]) + len(feats["clicks"]) + n_scroll}


def enroll(events, seed=0):
    """Store the fingerprint vector plus each feature's bootstrap standard
    error, so matching knows how much each feature is allowed to wobble."""
    feats, n_scroll, dur_min = _prep(events)
    v = _compute(feats, n_scroll, dur_min)
    se = _bootstrap_se(feats, n_scroll, dur_min, seed=seed)
    return {"vector": v, "se": se,
            "n_keys": len(feats["keystrokes"]),
            "n_mouse": len(feats["moves"]) + len(feats["clicks"]) + n_scroll}


def _distance(va, sea, vb, seb):
    shared = [k for k in va if k in vb and k in FLOOR]
    per = []
    for k in shared:
        denom = np.sqrt(sea.get(k, 0.0) ** 2 + seb.get(k, 0.0) ** 2 + FLOOR[k] ** 2)
        per.append((k, abs(va[k] - vb[k]) / denom))
    if not per:
        return None, []
    # Gently trimmed RMS: drop only the single worst-agreeing feature or two, so
    # one pathological feature cannot tank a genuine match, while an impostor
    # (many shifted features) is unaffected.
    zs = sorted(z for _, z in per)
    keep = max(6, int(round(len(zs) * 0.88)))
    D = float(np.sqrt(np.mean([z * z for z in zs[:keep]])))
    per.sort(key=lambda kv: -kv[1])
    return D, per


def match(enrolled, sample_events, seed=1):
    feats, n_scroll, dur_min = _prep(sample_events)
    sv = _compute(feats, n_scroll, dur_min)
    sse = _bootstrap_se(feats, n_scroll, dur_min, seed=seed)
    n_sample = len(feats["keystrokes"]) + len(feats["moves"]) + len(feats["clicks"]) + n_scroll
    D, per = _distance(enrolled["vector"], enrolled.get("se", {}), sv, sse)
    if D is None:
        return {"ok": False, "error": "the sample shares no features with the enrolled fingerprint"}
    similarity = 100.0 * float(np.exp(-0.5 * (D / K) ** 2))
    n_small = min(enrolled["n_keys"] + enrolled["n_mouse"], n_sample)
    shared_n = len(per)
    if n_small >= 500 and shared_n >= 10:
        cert = "high"
    elif n_small >= 180 and shared_n >= 6:
        cert = "medium"
    else:
        cert = "low"
    cert_pct = int(min(100, 100 * (min(n_small, 600) / 600.0) * (min(shared_n, 12) / 12.0)))
    verdict = ("likely the same person" if similarity >= 70 else
               "uncertain" if similarity >= 45 else "likely a different person")
    return {"ok": True, "similarity": round(similarity, 1), "verdict": verdict,
            "distance": round(D, 3), "certainty": cert, "certainty_pct": cert_pct,
            "shared": shared_n, "n_enrolled": enrolled["n_keys"] + enrolled["n_mouse"],
            "n_sample": n_sample,
            "top": [{"feature": k, "z": round(z, 2)} for k, z in per[:6]]}


def save(fp, path):
    from .schema import dump_json
    dump_json(path, fp)


def load(path):
    from .schema import load_json
    return load_json(path)
