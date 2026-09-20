"""
assignment 4 - the actual gpt. attention written by hand, no nn.MultiheadAttention
or nn.Transformer anywhere, per rule 2 of the sheet.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """one self attention head"""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # not a parameter, just gets moved to device with the module
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # B,T,hs
        q = self.query(x)   # B,T,hs
        v = self.value(x)   # B,T,hs

        hs = k.shape[-1]
        wei = q @ k.transpose(-2, -1) * (hs ** -0.5)  # B,T,T
        # careful, when generating T can be smaller than block_size so we
        # have to slice the mask down, otherwise this blows up
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)

        out = wei @ v  # B,T,hs
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)
        ])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.drop(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=3,
                 block_size=64, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide by n_head"
        self.block_size = block_size
        self.tok_emb_table = nn.Embedding(vocab_size, n_embd)
        self.pos_emb_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb_table(idx)  # B,T,C
        pos_emb = self.pos_emb_table(torch.arange(T, device=idx.device))  # T,C
        x = tok_emb + pos_emb  # broadcast over batch
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B,T,V

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    # ---- the interface the perplexity/generate functions rely on ----

    @torch.no_grad()
    def next_token_log_probs(self, context):
        device = next(self.parameters()).device
        ctx = context[-self.block_size:]
        idx = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = self.forward(idx)
        last_logits = logits[0, -1, :]
        logp = F.log_softmax(last_logits, dim=-1)
        return logp.cpu().numpy()

    def log_prob(self, token, context):
        return float(self.next_token_log_probs(context)[token])


def get_batch(ids, block_size, batch_size, device):
    """picks batch_size random windows of length block_size out of ids,
    and the same windows shifted by one as targets"""
    n = len(ids) - block_size - 1
    starts = torch.randint(0, n, (batch_size,))
    x = torch.stack([ids[i:i + block_size] for i in starts])
    y = torch.stack([ids[i + 1:i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, eval_iters=50):
    """splits is a dict like {'train': ids_tensor, 'val': ids_tensor}"""
    out = {}
    model.eval()
    for split, ids in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(ids, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model, train_ids, val_ids=None, max_steps=2000, lr=1e-3,
          batch_size=32, block_size=None, device="cpu", eval_every=200,
          eval_iters=30):
    """trains model with adamw, prints train/val loss every so often, and
    returns a little history dict so we can plot it later"""
    if block_size is None:
        block_size = model.block_size

    model.to(device)
    train_ids_t = torch.tensor(train_ids, dtype=torch.long)
    splits = {"train": train_ids_t}
    if val_ids is not None:
        splits["val"] = torch.tensor(val_ids, dtype=torch.long)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"step": [], "train_loss": [], "val_loss": []}

    model.train()
    for step in range(max_steps):
        xb, yb = get_batch(train_ids_t, block_size, batch_size, device)
        logits, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == max_steps - 1:
            losses = estimate_loss(model, splits, block_size, batch_size, device, eval_iters)
            history["step"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses.get("val", float("nan")))
            val_str = f"val {losses['val']:.4f}" if "val" in losses else ""
            print(f"step {step}: train {losses['train']:.4f} {val_str}")

    return history
