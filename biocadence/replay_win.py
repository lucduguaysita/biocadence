"""Windows SendInput injector: realize a planned op stream as real input.

This is the "simple keyboard injection" layer. It uses the modern SendInput
API through ctypes (not the legacy keybd_event/mouse_event). Characters go in
as Unicode by default (layout independent, no manual shift handling); special
keys and, optionally, characters go in as hardware scan codes for apps/games
that only read scan codes. Mouse moves use absolute virtual-desktop
coordinates; timing is driven to sub-millisecond with a raised timer
resolution and a hybrid sleep.

The module imports on any OS so the rest of the pipeline stays testable, but
the actual injection calls only work on Windows and raise a clear error
elsewhere. Hold F12 to abort a replay in progress.
"""

import ctypes
import sys
import time
from time import perf_counter

# Fixed-width ctypes types (avoids importing wintypes so this loads anywhere).
WORD = ctypes.c_ushort
DWORD = ctypes.c_uint32
LONG = ctypes.c_int32
ULONG_PTR = ctypes.c_size_t

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
WHEEL_DELTA = 120

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

VK_F12 = 0x7B

_BUTTON_FLAGS = {
    "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
    "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
}

# token -> virtual-key code for special keys (characters use Unicode instead)
_VK = {
    "Key.space": 0x20, "Key.enter": 0x0D, "Key.tab": 0x09, "Key.backspace": 0x08,
    "Key.esc": 0x1B, "Key.delete": 0x2E, "Key.insert": 0x2D,
    "Key.shift": 0x10, "Key.shift_l": 0xA0, "Key.shift_r": 0xA1,
    "Key.ctrl": 0x11, "Key.ctrl_l": 0xA2, "Key.ctrl_r": 0xA3,
    "Key.alt": 0x12, "Key.alt_l": 0xA4, "Key.alt_gr": 0xA5, "Key.alt_r": 0xA5,
    "Key.cmd": 0x5B, "Key.caps_lock": 0x14,
    "Key.left": 0x25, "Key.up": 0x26, "Key.right": 0x27, "Key.down": 0x28,
    "Key.home": 0x24, "Key.end": 0x23, "Key.page_up": 0x21, "Key.page_down": 0x22,
}
for _i in range(1, 13):
    _VK[f"Key.f{_i}"] = 0x70 + (_i - 1)

_EXTENDED = {0x25, 0x26, 0x27, 0x28, 0x2E, 0x2D, 0x24, 0x23, 0x21, 0x22, 0xA3, 0xA5}


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", LONG), ("dy", LONG), ("mouseData", DWORD),
                ("dwFlags", DWORD), ("time", DWORD), ("dwExtraInfo", ULONG_PTR)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", WORD), ("wScan", WORD), ("dwFlags", DWORD),
                ("time", DWORD), ("dwExtraInfo", ULONG_PTR)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", DWORD), ("u", _INPUTUNION)]


def _require_windows():
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "replay_win injects input through the Windows SendInput API and "
            "only runs on Windows. Build/inspect the plan anywhere; run the "
            "replay on your Windows machine."
        )
    return ctypes.WinDLL("user32", use_last_error=True)


class Injector:
    def __init__(self, use_scancode_for_chars=False):
        self.user32 = _require_windows()
        self.use_scancode_for_chars = use_scancode_for_chars
        self.vx = self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
        self.vy = self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
        self.vw = max(self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN), 1)
        self.vh = max(self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN), 1)

    def _send(self, inp):
        n = self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))
        if n != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def _key_scancode(self, vk, keyup, extended=False):
        scan = self.user32.MapVirtualKeyW(vk, 0)  # MAPVK_VK_TO_VSC
        flags = KEYEVENTF_SCANCODE
        if extended or vk in _EXTENDED:
            flags |= KEYEVENTF_EXTENDEDKEY
        if keyup:
            flags |= KEYEVENTF_KEYUP
        inp = _INPUT(type=INPUT_KEYBOARD,
                     u=_INPUTUNION(ki=_KEYBDINPUT(0, scan, flags, 0, 0)))
        self._send(inp)

    def _key_unicode(self, ch, keyup):
        # Send one SendInput per UTF-16 code unit. BMP chars are a single
        # unit; characters outside the BMP arrive as two surrogate units.
        flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if keyup else 0)
        raw = ch.encode("utf-16-le")
        units = [raw[i] | (raw[i + 1] << 8) for i in range(0, len(raw), 2)]
        for unit in units:
            inp = _INPUT(type=INPUT_KEYBOARD,
                         u=_INPUTUNION(ki=_KEYBDINPUT(0, unit, flags, 0, 0)))
            self._send(inp)

    def key(self, token, keyup):
        if token in _VK:
            self._key_scancode(_VK[token], keyup)
        elif len(token) == 1:
            if self.use_scancode_for_chars:
                vk = self.user32.VkKeyScanW(ord(token)) & 0xFF
                if vk not in (0, 0xFF):
                    self._key_scancode(vk, keyup)
                else:
                    self._key_unicode(token, keyup)
            else:
                self._key_unicode(token, keyup)
        else:
            # unknown 'Key.*' token: best-effort no-op rather than crash
            return

    def mouse_move(self, x, y):
        nx = int(round((x - self.vx) * 65535.0 / max(self.vw - 1, 1)))
        ny = int(round((y - self.vy) * 65535.0 / max(self.vh - 1, 1)))
        flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        inp = _INPUT(type=INPUT_MOUSE,
                     u=_INPUTUNION(mi=_MOUSEINPUT(nx, ny, 0, flags, 0, 0)))
        self._send(inp)

    def mouse_button(self, button, down):
        flag = _BUTTON_FLAGS[button][0 if down else 1]
        inp = _INPUT(type=INPUT_MOUSE,
                     u=_INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, 0)))
        self._send(inp)

    def mouse_wheel(self, clicks):
        data = (int(clicks) * WHEEL_DELTA) & 0xFFFFFFFF
        inp = _INPUT(type=INPUT_MOUSE,
                     u=_INPUTUNION(mi=_MOUSEINPUT(0, 0, data, MOUSEEVENTF_WHEEL, 0, 0)))
        self._send(inp)


def _sleep_until(target):
    """Hybrid wait: sleep most of the gap, spin the last ~1 ms for accuracy."""
    while True:
        rem = target - perf_counter()
        if rem <= 0:
            return
        if rem > 0.0015:
            time.sleep(rem - 0.0012)


def _abort_pressed(user32):
    return bool(user32.GetAsyncKeyState(VK_F12) & 0x8000)


def dispatch(inj, op):
    kind = op["op"]
    if kind == "key_down":
        inj.key(op["token"], keyup=False)
    elif kind == "key_up":
        inj.key(op["token"], keyup=True)
    elif kind == "mouse_move":
        inj.mouse_move(op["x"], op["y"])
    elif kind == "mouse_down":
        inj.mouse_button(op["button"], down=True)
    elif kind == "mouse_up":
        inj.mouse_button(op["button"], down=False)
    elif kind == "scroll":
        inj.mouse_wheel(op.get("dy", 1))


def replay(ops, use_scancode_for_chars=False, speed=1.0, lead_in=3.0,
           dry_run=False, on_op=None):
    """Play an op stream. Set dry_run=True to walk it without injecting
    (works on any OS, useful for logging/verification)."""
    if dry_run:
        for op in ops:
            if on_op:
                on_op(op)
        return

    inj = Injector(use_scancode_for_chars=use_scancode_for_chars)
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod(1)
    try:
        for i in range(int(lead_in), 0, -1):
            print(f"Replay starting in {i}... (focus the target window; F12 aborts)")
            time.sleep(1.0)
        t0 = perf_counter()
        for op in ops:
            _sleep_until(t0 + op["t"] / speed)
            if _abort_pressed(inj.user32):
                print("Aborted (F12).")
                break
            dispatch(inj, op)
            if on_op:
                on_op(op)
    finally:
        winmm.timeEndPeriod(1)
