"""Neural pointer-behavior model.

A small autoregressive transformer over a tokenized stream of pointer events
(moves, clicks, scroll). Drag and text-selection are not special cased; they
are simply left-button-held-move runs the model learns from data.

Modules:
  tokenizer  raw recorder events <-> integer token stream (quantized dt/dx/dy)
  model      compact decoder-only transformer (PyTorch)
  dataset    token stream -> training windows
  train      CUDA-aware training loop
  sample     autoregressive sampling, gesture extraction, retarget -> op stream

Train on the machine with the GPU. Everything except train/sample needs only
numpy; train/sample need torch.
"""

__all__ = ["tokenizer", "model", "dataset", "train", "sample"]
