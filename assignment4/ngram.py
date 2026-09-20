"""
the count-based n-gram engine from assignment 2, reused here as is (rule 3
of assignment 4 says don't rewrite it). add-one smoothing everywhere, no
special case for zero counts since the +1/+V formula already does the
right thing on its own for a context we never saw.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


class NGramLM:
    def __init__(self, n, vocab_size):
        self.n = n
        self.ctx_len = n - 1
        self.V = vocab_size
        # (context tuple) -> Counter-ish dict of next token -> count
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def fit(self, sequences):
        """sequences is a list of token-id lists, each one already padded
        with n-1 <bos> at the front (we do that outside, in the notebook,
        since the amount of padding depends on n)."""
        for seq in sequences:
            for i in range(self.ctx_len, len(seq)):
                ctx = tuple(seq[i - self.ctx_len:i])
                w = seq[i]
                self.ngram_counts[ctx][w] += 1
                self.context_counts[ctx] += 1

    def log_prob(self, token, context):
        ctx = tuple(context[-self.ctx_len:]) if self.ctx_len > 0 else ()
        c_h = self.context_counts.get(ctx, 0)
        c_hw = self.ngram_counts.get(ctx, {}).get(token, 0)
        p = (c_hw + 1) / (c_h + self.V)
        return math.log(p)

    def next_token_log_probs(self, context):
        ctx = tuple(context[-self.ctx_len:]) if self.ctx_len > 0 else ()
        c_h = self.context_counts.get(ctx, 0)
        row = self.ngram_counts.get(ctx, {})
        # a python loop over V here is not super fast but V is at most a
        # few thousand so it's not worth being clever about
        counts = np.zeros(self.V, dtype=np.float64)
        for tok, c in row.items():
            counts[tok] = c
        probs = (counts + 1.0) / (c_h + self.V)
        return np.log(probs)
