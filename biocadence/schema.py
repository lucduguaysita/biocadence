"""Raw event schema and small shared helpers.

A recording is just a list of plain dicts (JSON-friendly, no custom types).
Timestamps `t` are seconds from a monotonic high-resolution clock
(time.perf_counter) captured at record time, so gaps between events are
accurate even if the wall clock changes.

Event shapes:
  {"t": float, "type": "key_press",   "key": "<token>", "vk": int|null}
  {"t": float, "type": "key_release", "key": "<token>", "vk": int|null}
  {"t": float, "type": "mouse_move",  "x": int, "y": int}
  {"t": float, "type": "mouse_down",  "button": "left|right|middle", "x": int, "y": int}
  {"t": float, "type": "mouse_up",    "button": "left|right|middle", "x": int, "y": int}
  {"t": float, "type": "scroll",      "x": int, "y": int, "dx": int, "dy": int}

A `<token>` is either a single printable character ("a", "A", "7", ".", " ")
or a special-key name in pynput form ("Key.enter", "Key.backspace", ...).
"""

import json

# Event type constants
KEY_PRESS = "key_press"
KEY_RELEASE = "key_release"
MOUSE_MOVE = "mouse_move"
MOUSE_DOWN = "mouse_down"
MOUSE_UP = "mouse_up"
SCROLL = "scroll"

# Modifier keys are tracked as held state, not modeled as timed tokens.
MODIFIER_TOKENS = {
    "Key.shift", "Key.shift_r", "Key.shift_l",
    "Key.ctrl", "Key.ctrl_r", "Key.ctrl_l",
    "Key.alt", "Key.alt_r", "Key.alt_l", "Key.alt_gr",
    "Key.cmd", "Key.cmd_r", "Key.cmd_l",
    "Key.caps_lock",
}

# Special keys that we DO model as timed typing tokens, mapped to the
# character they contribute to the produced text (backspace is an edit).
SPECIAL_TO_CHAR = {
    "Key.space": " ",
    "Key.enter": "\n",
    "Key.tab": "\t",
    "Key.backspace": "\b",
}

SENTENCE_ENDERS = {".", "!", "?"}
CLAUSE_ENDERS = {",", ";", ":"}


def key_class(token):
    """Coarse class used for backing off when a specific key/digraph is unseen."""
    if token in SPECIAL_TO_CHAR:
        name = token.split(".")[-1]
        return name  # 'space', 'enter', 'tab', 'backspace'
    if token in MODIFIER_TOKENS:
        return "modifier"
    if len(token) == 1:
        if token.isalpha():
            return "letter"
        if token.isdigit():
            return "digit"
        if token.isspace():
            return "space"
        return "punct"
    return "special"


def token_to_char(token):
    """The character a modeled token contributes to produced text.

    Returns None for tokens that do not add a character (backspace, modifiers,
    other special keys). Used when reconstructing the typed-character stream.
    """
    if len(token) == 1:
        return token
    if token in SPECIAL_TO_CHAR:
        ch = SPECIAL_TO_CHAR[token]
        return None if ch == "\b" else ch
    return None


def char_to_token(ch):
    """Inverse of token_to_char for planning: a target character -> the token
    the model is keyed on."""
    if ch == " ":
        return "Key.space"
    if ch == "\n":
        return "Key.enter"
    if ch == "\t":
        return "Key.tab"
    return ch


def load_jsonl(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def dump_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
