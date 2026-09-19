"""
count n-gram and neural n-gram models, carried over from assignment 2 and 3.

we are not supposed to touch the interface (log_prob / next_token_log_probs)
because the perplexity and generate functions from evaluation.py call those
two methods and nothing else. everything else in here is just however we
happened to write it back then.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# count n-gram (assignment 2)
# ---------------------------------------------------------------------

class CountNgram:
    """plain old count based n-gram with add-one smoothing. k here is the
    order, i.e. we look at the previous k-1 tokens (context length k-1)."""

    def __init__(self, vocab_size, k=3):
        self.V = vocab_size
        self.k = k  # order, k=3 -> trigram
        self.ctx_len = k - 1
        self.counts = defaultdict(lambda: np.zeros(self.V, dtype=np.float64))

    def train(self, ids):
        ids = list(ids)
        pad = [0] * self.ctx_len
        ids = pad + ids
        for i in range(self.ctx_len, len(ids)):
            ctx = tuple(ids[i - self.ctx_len:i])
            self.counts[ctx][ids[i]] += 1

    def next_token_log_probs(self, context):
        ctx = tuple(context[-self.ctx_len:]) if self.ctx_len > 0 else ()
        if len(ctx) < self.ctx_len:
            ctx = tuple([0] * (self.ctx_len - len(ctx))) + ctx
        c = self.counts.get(ctx)
        if c is None:
            probs = np.ones(self.V, dtype=np.float64) / self.V
        else:
            smoothed = c + 1.0  # add one smoothing, simple but works ok
            probs = smoothed / smoothed.sum()
        return np.log(probs)

    def log_prob(self, token, context):
        return float(self.next_token_log_probs(context)[token])


# ---------------------------------------------------------------------
# neural n-gram (assignment 3), basically a tiny bengio style MLP
# ---------------------------------------------------------------------

class NeuralNgram(nn.Module):
    def __init__(self, vocab_size, k=4, n_embd=32, n_hidden=128):
        super().__init__()
        self.V = vocab_size
        self.k = k
        self.ctx_len = k - 1
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(n_embd * self.ctx_len, n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, x, y=None):
        # x: (B, ctx_len)
        emb = self.tok_emb(x)  # B, ctx_len, n_embd
        flat = emb.view(emb.shape[0], -1)
        h = torch.tanh(self.fc1(flat))
        logits = self.fc2(h)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits, y)
        return logits, loss

    @torch.no_grad()
    def next_token_log_probs(self, context):
        device = next(self.parameters()).device
        ctx = list(context[-self.ctx_len:])
        if len(ctx) < self.ctx_len:
            ctx = [0] * (self.ctx_len - len(ctx)) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = self.forward(x)
        logp = F.log_softmax(logits[0], dim=-1)
        return logp.cpu().numpy()

    def log_prob(self, token, context):
        return float(self.next_token_log_probs(context)[token])


def get_batch_ngram(ids, ctx_len, batch_size, device):
    ids_t = torch.tensor(ids, dtype=torch.long)
    n = len(ids_t) - ctx_len - 1
    starts = torch.randint(0, n, (batch_size,))
    xs = torch.stack([ids_t[s:s + ctx_len] for s in starts])
    ys = torch.stack([ids_t[s + ctx_len] for s in starts])
    return xs.to(device), ys.to(device)


def train_neural_ngram(model, ids, max_steps=1500, lr=1e-3, batch_size=64, device="cpu"):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(max_steps):
        xb, yb = get_batch_ngram(ids, model.ctx_len, batch_size, device)
        logits, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model
