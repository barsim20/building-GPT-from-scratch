
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
