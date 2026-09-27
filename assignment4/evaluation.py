"""
perplexity + generate, assignment 4's own evaluation harness (works with
any model that has log_prob and next_token_log_probs -- the count n-gram,
the neural n-gram or the GPT -- by walking the FULL growing context over a
long token stream, which none of assignment 2/3's own perplexity/generate
do: theirs work per short line and are windowed to exactly n-1 tokens by
the caller, which would cut the GPT off after one token of context and
defeat the point of comparing it to the n-grams. so these two functions
stay assignment 4's own code; what *did* change is that ngram.py and
neural_ngram.py now hold the real NGramLM (assignment 2) and NeuralNGramLM
(assignment 3) classes instead of stand-ins, and the two helpers below make
this evaluation harness call them exactly the way their own notebooks do.

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


def context_length(model):
    """how many previous tokens this model's log_prob/next_token_log_probs
    actually needs. the neural n-gram and the GPT crop to this length
    themselves (ctx_len / block_size respectively) -- but assignment 2's
    real NGramLM does not, it uses whatever context tuple it's handed
    directly as a dictionary key (see ngram.py), so a caller that wants to
    reuse it on a long, continuous token stream has to hand it exactly the
    n-1-token window itself."""
    if hasattr(model, "ctx_len"):
        return model.ctx_len
    if hasattr(model, "block_size"):
        return model.block_size
    if hasattr(model, "n"):
        return model.n - 1
    return 1


def crop_context(model, context):
    """no-op for the neural n-gram and the GPT, which already crop
    internally; windows the context down to context_length(model) for
    assignment 2's NGramLM, which does not."""
    if hasattr(model, "ctx_len") or hasattr(model, "block_size"):
        return context
    ctx_len = context_length(model)
    return context[-ctx_len:] if ctx_len > 0 else []


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
        context = crop_context(model, ids[:i])
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
    ctx_len = context_length(model)  # n-1 for the n-grams, just 1 "start" marker for the GPT

    prompt_ids = tok.encode(prompt)
    ids = [bos] * ctx_len + prompt_ids

    hit_limit = True
    for _ in range(max_tokens):
        logp = model.next_token_log_probs(crop_context(model, ids))
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
