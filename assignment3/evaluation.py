import math
import random

import numpy as np

from bpe import normalize


def perplexity(model, sequences, original_text_char_count=None):
    total_log_prob = 0
    N = 0  # Anzahl der vorhergesagten Tokens (inkl. <eos>, ohne <bos>)

    for seq in sequences:
        # Wir starten bei n-1, da die ersten n-1 <bos> Tokens nicht vorhergesagt werden
        for i in range(model.n - 1, len(seq)):
            context = seq[i - model.n + 1:i]
            token = seq[i]
            total_log_prob += model.log_prob(token, context)
            N += 1

    # PP = exp(-1/N * sum(log P))
    pp = math.exp(-total_log_prob / N)

    if original_text_char_count is not None:
        # PP_char = exp(-1/C * sum(log P))
        pp_char = math.exp(-total_log_prob / original_text_char_count)
        return pp, pp_char

    return pp


def generate(model, tokenizer, model_strategy, prompt, mode="sample", max_tokens=50, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # normalize prompt and encode it
    ids = tokenizer.encode(normalize(prompt, model_strategy), count_unk=False)
    # add <bos> tokens
    ids = [tokenizer.vocab["<bos>"]] * (model.n - 1) + ids

    # calculate token probabilities
    output = []
    for _ in range(max_tokens):
        context = ids[-(model.n-1):] if model.n > 1 else []
        log_probs = model.next_token_log_probs(context)

        # choose mode
        if mode == "argmax":
            next_token = np.argmax(log_probs)

        else: # mode == sample
            probs = np.exp(log_probs)
            next_token = np.random.choice(len(probs), p=probs)

        # early stop condition
        if next_token == tokenizer.vocab["<eos>"]:
            break

        ids.append(next_token)
        output.append(next_token)
    #latest stop condition
    else:
        print("Limit reached!")

        #return decoded tokens
    return tokenizer.decode(output)
