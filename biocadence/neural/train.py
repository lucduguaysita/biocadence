"""CUDA-aware training loop for the pointer transformer.

On your machine this runs on the RTX 5090 automatically (device 'cuda'); here
it falls back to CPU. Checkpoints bundle the model weights, the model config,
and the tokenizer config so sampling is fully reproducible from one file.
"""

import math
import time

import numpy as np
import torch
from torch import optim

from .model import build_model, PointerGPT, GPTConfig
from .tokenizer import Tokenizer
from .dataset import concat_tokens, ensure_length, get_batch, to_tensor


def pick_device(pref=None):
    if pref:
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(token_seqs, tokenizer, preset="base", steps=2000, batch_size=64,
          lr=3e-4, warmup=100, weight_decay=0.1, grad_clip=1.0, block_size=None,
          device=None, log_every=100, out_path=None, seed=0,
          progress=None, should_stop=None):
    device = pick_device(device)
    torch.manual_seed(seed)
    model, cfg = build_model(tokenizer.vocab_size, preset, block_size)
    model.to(device)
    data = to_tensor(ensure_length(concat_tokens(token_seqs), cfg.block_size))

    opt = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                      weight_decay=weight_decay)

    def lr_at(step):
        if step < warmup:
            return lr * step / max(warmup, 1)
        p = (step - warmup) / max(steps - warmup, 1)
        return 0.5 * lr * (1.0 + math.cos(math.pi * min(p, 1.0)))

    gen = torch.Generator().manual_seed(seed)
    print(f"device {device} | params {model.num_params()/1e6:.2f}M | "
          f"tokens {len(data)} | vocab {tokenizer.vocab_size} | preset {preset}")
    model.train()
    t0 = time.time()
    recent = []
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(data, cfg.block_size, batch_size, device, gen)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        recent.append(loss.item())
        cur = float(np.mean(recent[-log_every:]))
        if step == 1 or step % log_every == 0:
            rate = step / (time.time() - t0)
            print(f"step {step:6d}/{steps}  loss {cur:.4f}"
                  f"  lr {lr_at(step):.2e}  {rate:.1f} it/s")
            if progress is not None:
                progress(step, steps, cur)
        if should_stop is not None and should_stop():
            print(f"stop requested at step {step}")
            if progress is not None:
                progress(step, steps, cur)
            break

    ckpt = {"model": model.state_dict(), "cfg": cfg.__dict__, "preset": preset,
            "tokenizer": tokenizer.config(), "vocab_size": tokenizer.vocab_size}
    if out_path:
        torch.save(ckpt, out_path)
        print(f"saved -> {out_path}")
    return model, ckpt


def load_checkpoint(path, device=None):
    device = pick_device(device)
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt["cfg"])
    model = PointerGPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    tok = Tokenizer.from_config(ckpt["tokenizer"])
    return model, tok, cfg, device
