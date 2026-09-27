"""
the neural n-gram from assignment 3: Jonas's real NeuralNGramLM class,
copied here verbatim (not a stand-in anymore). get_batch/estimate_loss/
train below are assignment 4's own training utilities, written because
assignment 3 kept that part inline in its notebook rather than in a
module -- they orchestrate the real class above without changing anything
inside it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNGramLM(nn.Module):

    def __init__(self, n, vocab_size, n_embd=64, n_hidden=256):
        super().__init__()

        self.n = n
        self.ctx_len = n - 1
        self.vocab_size = vocab_size

        # converting every token id into an embedding vector
        self.embedding = nn.Embedding(vocab_size, n_embd)

        # taking all context embeddings as one vector
        self.hidden = nn.Linear(self.ctx_len * n_embd, n_hidden)

        self.relu = nn.ReLU()

        # giving one output score for every token in vocabulary
        self.output = nn.Linear(n_hidden, vocab_size)


    def forward(self, idx, targets=None):

        # convert token ids into embeddings
        embeddings = self.embedding(idx)

        # put embeddings of the context into one vector
        flat = embeddings.view(embeddings.shape[0], -1)

        # hidden layer and ReLU
        hidden = self.relu(self.hidden(flat))

        # scores for every possible next token
        logits = self.output(hidden)

        # calculate loss only if targets are given
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


    @torch.no_grad()
    def next_token_log_probs(self, context):

        self.eval()

        # only use the previous n-1 tokens
        context = context[-self.ctx_len:]

        # convert context into tensor
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=next(self.parameters()).device
        )

        # get output scores
        logits, _ = self.forward(x)

        # convert scores into probabilities and then log probabilities
        probabilities = F.softmax(logits[0], dim=-1)
        log_probs = torch.log(probabilities)

        return log_probs.cpu().numpy()


    @torch.no_grad()
    def log_prob(self, token, context):

        # get probabilities for all possible next tokens
        log_probs = self.next_token_log_probs(context)

        # return probability of the wanted token
        return float(log_probs[token])


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
