"""Token sequences -> training batches (standard next-token LM windows)."""

import numpy as np
import torch


def concat_tokens(seqs):
    if not seqs:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])


def ensure_length(data, block_size):
    """Tile short data so at least one full window exists (useful for demos)."""
    if len(data) < block_size + 2:
        reps = (block_size + 2) // max(len(data), 1) + 1
        data = np.tile(data, reps)
    return data


def to_tensor(data):
    """np int array -> torch LongTensor. Uses the fast numpy bridge when torch
    and numpy agree (your machine), and falls back to a list copy when they do
    not (older torch against numpy 2.x)."""
    arr = np.ascontiguousarray(np.asarray(data), dtype=np.int64)
    try:
        return torch.from_numpy(arr).clone()
    except (RuntimeError, TypeError):
        return torch.tensor(arr.tolist(), dtype=torch.long)


def get_batch(data_t, block_size, batch_size, device, generator=None):
    hi = max(len(data_t) - block_size - 1, 1)
    ix = torch.randint(0, hi, (batch_size,), generator=generator).tolist()
    x = torch.stack([data_t[i:i + block_size] for i in ix])
    y = torch.stack([data_t[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
