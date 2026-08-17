"""A compact decoder-only transformer (nanoGPT style) over pointer tokens.

Uses torch.nn.functional.scaled_dot_product_attention (no triton needed), a
learned positional embedding, tied input/output embeddings, and cross-entropy
with PAD ignored. Presets range from a tiny CPU-testable model up to one sized
comfortably for a 32 GB GPU.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD


@dataclass
class GPTConfig:
    vocab_size: int = 170
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


PRESETS = {
    "tiny":  dict(n_layer=2, n_head=2, n_embd=64,  block_size=128, dropout=0.1),
    "base":  dict(n_layer=4, n_head=4, n_embd=128, block_size=256, dropout=0.1),
    "large": dict(n_layer=8, n_head=8, n_embd=384, block_size=512, dropout=0.1),
}


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn_qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.attn_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        q, k, v = self.attn_qkv(h).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        a = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        a = a.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.resid_drop(self.attn_proj(a))
        x = x + self.mlp(self.ln2(x))
        return x


class PointerGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight        # tie weights
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=PAD)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx


def build_model(vocab_size, preset="base", block_size=None):
    kw = dict(PRESETS[preset])
    if block_size is not None:
        kw["block_size"] = block_size
    cfg = GPTConfig(vocab_size=vocab_size, **kw)
    return PointerGPT(cfg), cfg
