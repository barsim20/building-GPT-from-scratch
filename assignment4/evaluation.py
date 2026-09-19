"""
perplexity + generate, unchanged since assignment 2 (assignment 3 and 4
both just import these). works with any model that has log_prob and
next_token_log_probs, doesn't care whether that's the count n-gram, the
neural n-gram or the GPT.

convention we use everywhere: <bos> and <eos> are not part of the BPE
vocab itself (bpe.py doesn't know about them), we just tack them onto the
end of the id space -> bos_id = tok.vocab_size(), eos_id = tok.vocab_size()+1.
every module that needs them (ngram.py, neural_ngram.py, gpt.py,
the notebook) uses the same two helpers below so this only lives in one
place.
"""

import math

import numpy as np
import torch


def bos_id(tok):
    return tok.vocab_size()


def eos_id(tok):
    return tok.vocab_size() + 1


def total_vocab_size(tok):
    return tok.vocab_size() + 2  # + <bos>, <eos>


def perplexity(model, ids, chars_per_token=None, bos=None):
    """ids is one (long) list of token ids, <bos>/<eos> included inline.
    walks a growing context over the list and accumulates log P(target |
    everything before it). <bos> tokens are not counted as a target,
    since the model is never asked to predict them -- but they DO still
    show up inside other tokens' context, same as any other token.

    returns (pp_per_token, pp_per_char).
    """
    total_logprob = 0.0
    n = 0
    for i in range(1, len(ids)):
        target = ids[i]
        if bos is not None and target == bos:
            continue
        context = ids[:i]
        total_logprob += model.log_prob(target, context)
        n += 1

    avg_nll = -total_logprob / n
    pp_token = math.exp(avg_nll)

    if chars_per_token is None:
        chars_per_token = 1.0
    pp_char = math.exp(avg_nll / chars_per_token)
    return pp_token, pp_char


def generate(model, tok, prompt, mode="sample", max_tokens=100, seed=None):
    """mode is 'sample' or 'argmax'. stops early at <eos>, otherwise stops
    after max_tokens and says so."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    bos, eos = bos_id(tok), eos_id(tok)
    ctx_len = getattr(model, "ctx_len", 1)  # n-1 for the n-grams, just 1 "start" marker for the GPT

    prompt_ids = tok.encode(prompt)
    ids = [bos] * ctx_len + prompt_ids

    hit_limit = True
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
        if next_id == eos:
            hit_limit = False
            break

    if hit_limit:
        print("(hit max_tokens before <eos>)")

    # drop the priming <bos> padding and a trailing <eos>, if there is one,
    # so decode() gets clean token ids it actually knows about
    body = ids[ctx_len:]
    if body and body[-1] == eos:
        body = body[:-1]
    return tok.decode(body)
