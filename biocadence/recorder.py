"""Record a real input session to raw JSONL using pynput.

Run this on the machine whose style you want to model (Windows for this
project, though pynput also works elsewhere for testing). It captures every
key press/release, mouse button, mouse move (decimated) and scroll with
high-resolution relative timestamps.

PRIVACY: this is, by construction, a keylogger while it runs. It writes the
actual characters you type to a local file. Record only non-sensitive text
(type out an article, a chat, some code) and never type passwords, card
numbers or secrets during a capture. The file stays on your machine; nothing
is uploaded. Delete recordings you no longer need.
"""

import threading
from time import perf_counter

from .schema import (
    KEY_PRESS, KEY_RELEASE, MOUSE_MOVE, MOUSE_DOWN, MOUSE_UP, SCROLL,
    dump_jsonl,
)


def _import_pynput():
    try:
        from pynput import keyboard, mouse
        return keyboard, mouse
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pynput is required for recording. Install it with:\n"
            "    pip install pynput\n"
            f"(import failed: {exc})"
        )


def _key_token(key):
    """Map a pynput key object to (token, vk).

    token is a single printable character for normal keys, or a 'Key.<name>'
    string for special keys, matching schema.py tokens.
    """
    char = getattr(key, "char", None)
    if char is not None:
        token = char
    else:
        token = str(key)  # e.g. 'Key.space', 'Key.enter'
    vk = getattr(key, "vk", None)
    if vk is None:
        val = getattr(key, "value", None)
        vk = getattr(val, "vk", None)
    return token, vk


def record_session(out_path, stop_key="f12", min_move_interval=0.008, quiet=False):
    """Capture a session and write it to out_path (JSONL). Returns the events.

    stop_key: name of the key that ends the recording (default F12).
    min_move_interval: minimum seconds between stored mouse_move samples, so a
        high-rate mouse does not flood the log. 0.008 keeps up to ~125 Hz.
    """
    keyboard, mouse = _import_pynput()
    events = []
    t0 = perf_counter()
    last_move_t = [-1.0]
    stop_token = "Key." + stop_key
    stop_event = threading.Event()

    def now():
        return perf_counter() - t0

    def on_press(key):
        token, vk = _key_token(key)
        if token == stop_token:
            stop_event.set()
            return False
        events.append({"t": now(), "type": KEY_PRESS, "key": token, "vk": vk})

    def on_release(key):
        token, vk = _key_token(key)
        if token == stop_token:
            return False
        events.append({"t": now(), "type": KEY_RELEASE, "key": token, "vk": vk})

    def on_move(x, y):
        t = now()
        if t - last_move_t[0] >= min_move_interval:
            events.append({"t": t, "type": MOUSE_MOVE, "x": int(x), "y": int(y)})
            last_move_t[0] = t

    def on_click(x, y, button, pressed):
        events.append({
            "t": now(),
            "type": MOUSE_DOWN if pressed else MOUSE_UP,
            "button": getattr(button, "name", str(button)),
            "x": int(x), "y": int(y),
        })

    def on_scroll(x, y, dx, dy):
        events.append({
            "t": now(), "type": SCROLL,
            "x": int(x), "y": int(y), "dx": int(dx), "dy": int(dy),
        })

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    ms_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)

    if not quiet:
        print("Recording. Type and move naturally.")
        print("Only type non-sensitive text; this stores the actual keys.")
        print(f"Press {stop_key.upper()} to stop.\n")

    kb_listener.start()
    ms_listener.start()
    try:
        stop_event.wait()
    except KeyboardInterrupt:
        pass
    finally:
        kb_listener.stop()
        ms_listener.stop()

    dump_jsonl(out_path, events)
    if not quiet:
        print(f"Captured {len(events)} events -> {out_path}")
    return events
