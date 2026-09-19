"""
perplexity + generate, unchanged since assignment 2 like the assignment says.
every model (count ngram, neural ngram, gpt) only needs to expose
next_token_log_probs(context) and log_prob(token, context), so this file
does not care which model it got.
"""

import math

import numpy as np
import torch


def perplexity(model, ids, chars_per_token=None):
    """returns (pp_per_token, pp_per_char).

    ids is the full list of token ids for e.g. the test split. we slide a
    context over it and accumulate log prob of the next real token.
    chars_per_token: average number of characters one token corresponds to
    in the original text, used to turn perplexity per token into perplexity
    per character (so that different k / different vocab sizes are
    comparable, per the assignment).
    """
    total_logprob = 0.0
    n = 0
    for i in range(1, len(ids)):
        context = ids[:i]
        target = ids[i]
        lp = model.log_prob(target, context)
        total_logprob += lp
        n += 1

    avg_nll = -total_logprob / n
    pp_token = math.exp(avg_nll)

    if chars_per_token is None:
        chars_per_token = 1.0
    # perplexity per char: scale the exponent by tokens-per-char, not just
    # divide the pp, because pp is already an exp of an average
    pp_char = math.exp(avg_nll / chars_per_token)
    return pp_token, pp_char


def generate(model, tok, prompt, mode="sample", max_tokens=100, seed=None):
    """generate text from a prompt. mode is 'sample' or 'argmax'."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ids = tok.encode(prompt)
    for _ in range(max_tokens):
        logp = model.next_token_log_probs(ids)
        if mode == "argmax":
            next_id = int(np.argmax(logp))
        elif mode == "sample":
            p = np.exp(logp)
            p = p / p.sum()
            next_id = int(np.random.choice(len(p), p=p))
        else:
            raise ValueError(f"unknown mode {mode}")
        ids.append(next_id)

    return tok.decode(ids)
