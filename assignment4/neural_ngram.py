"""
the neural n-gram from assignment 3, a small bengio-style MLP: embed the
n-1 context tokens, concat, one hidden layer, softmax over V. carried
over unchanged per rule 3 of assignment 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNGramLM(nn.Module):
    def __init__(self, n, vocab_size, n_embd=64, n_hidden=256):
        super().__init__()
        self.n = n
        self.ctx_len = n - 1
        self.V = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(n_embd * self.ctx_len, n_hidden)
        self.fc2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, idx, targets=None):
        emb = self.tok_emb(idx)          # B, ctx_len, n_embd
        flat = emb.view(emb.shape[0], -1)
        h = F.relu(self.fc1(flat))
        logits = self.fc2(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def next_token_log_probs(self, context):
        device = next(self.parameters()).device
        ctx = list(context[-self.ctx_len:])
        if len(ctx) < self.ctx_len:
            # pad on the left, happens mostly right after a <bos>
            ctx = [ctx[0] if ctx else 0] * (self.ctx_len - len(ctx)) + ctx
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = self.forward(x)
        logp = F.log_softmax(logits[0], dim=-1)
        return logp.cpu().numpy()

    def log_prob(self, token, context):
        return float(self.next_token_log_probs(context)[token])


def get_batch(ids, ctx_len, batch_size, device):
    ids_t = torch.tensor(ids, dtype=torch.long) if not torch.is_tensor(ids) else ids
    n = len(ids_t) - ctx_len - 1
    starts = torch.randint(0, n, (batch_size,))
    xs = torch.stack([ids_t[s:s + ctx_len] for s in starts])
    ys = torch.stack([ids_t[s + ctx_len] for s in starts])
    return xs.to(device), ys.to(device)


@torch.no_grad()
def estimate_loss(model, splits, batch_size, device, eval_iters=30):
    out = {}
    model.eval()
    for name, ids in splits.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(ids, model.ctx_len, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(model, train_ids, val_ids=None, max_steps=2000, lr=1e-3, batch_size=64,
          device="cpu", eval_every=200, eval_iters=30):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    splits = {"train": train_ids}
    if val_ids is not None:
        splits["val"] = val_ids

    history = {"step": [], "train_loss": [], "val_loss": []}
    model.train()
    for step in range(max_steps):
        xb, yb = get_batch(train_ids, model.ctx_len, batch_size, device)
        logits, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == max_steps - 1:
            losses = estimate_loss(model, splits, batch_size, device, eval_iters)
            history["step"].append(step)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses.get("val", float("nan")))
            val_str = f"val {losses['val']:.4f}" if "val" in losses else ""
            print(f"step {step}: train {losses['train']:.4f} {val_str}")

    return history
