"""biocadence: model your own keystroke, click and mouse-motion style and
replay it as human-like input through the Windows SendInput API.

Pipeline:
  record   capture a real session (pynput) -> raw JSONL of timed events
  features raw events -> timing and motion features
  model    fit + sample the generative style model (portable JSON)
  planner  target text / high-level actions -> concrete timed event stream
  replay   inject the event stream via Windows SendInput (ctypes)

The modules are import-light on purpose. recorder imports pynput and
replay_win imports Windows-only ctypes calls, so they are only imported
when you actually use them. features, model, planner and synth are pure
Python + numpy and run on any OS.
"""

__version__ = "0.1.0"
