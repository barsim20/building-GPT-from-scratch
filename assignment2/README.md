| Name | Student number | Tasks | Share |
|---|---|---|---|
| Baran | _TODO_ | Assignment 1 (BPE tokenizer) | 25 % |
| Jeremy | _TODO_ | Assignment 2 (this folder: count n-gram) | 25 % |
| Jonas | _TODO_ | Assignment 3 (neural n-gram) | 25 % |
| Baran, Jeremy, Jonas | _TODO_ | Assignment 4 (GPT), together | 25 % |

> Shares are across the four assignments as a whole, not within this single
> folder: Jeremy owns Assignment 2 end to end, the same way Baran owns
> Assignment 1 and Jonas owns Assignment 3, and all three worked together on
> Assignment 4. See the root [`README.md`](../README.md) for the full
> breakdown.

One sentence about each member's work, in their own words: _TODO — replace
this line for each member._

## Assignment 2 — N-Grams

This folder contains the count-based n-gram language model, trained and
evaluated on the same *tiny Shakespeare* corpus as Assignment 1, per the
"GPT from scratch · Assignment 2" specification. It builds directly on
Assignment 1's BPE tokenizer (`bpe.py`, reused unchanged) and is in turn
reused as-is by Assignment 3 and Assignment 4.

### Contents

- `bpe.py` — Assignment 1's tokenizer (`BPETokenizer`, `normalize`), reused
  unchanged.
- `ngram.py` — the count-based n-gram engine (`NGramLM`): `fit(sequences)`,
  add-one smoothing `log_prob(token, context)`, and `next_token_log_probs`.
  Pulled out of the notebook into its own module, the same way `bpe.py` is
  its own module in Assignment 1, so Assignment 3 and Assignment 4 can
  `import` it instead of copy-pasting it.
- `evaluation.py` — `perplexity(model, sequences, original_text_char_count)`
  and `generate(model, tokenizer, model_strategy, prompt, mode, max_tokens,
  seed)`, also pulled out of the notebook for the same reason. Works with
  any model that has `log_prob`/`next_token_log_probs`, not just `NGramLM`.
- `assignment2.ipynb` — the notebook with all five tasks (data prep, the
  n-gram engine, perplexity, model selection over `n`/`k`, generation). All
  output cells are kept; it runs top to bottom with **Restart and Run All**.
  The three cells that used to define `NGramLM`, `perplexity` and `generate`
  inline now just `import` them from `ngram.py`/`evaluation.py` — nothing
  about what those functions *do* changed, only where the code lives.
- `data/shakespeare.txt` — the same tiny Shakespeare corpus as Assignment 1
  (identical file, copied over so this folder doesn't depend on a network
  connection). The notebook still falls back to downloading it from
  `karpathy/char-rnn` if the local file is ever missing, same as Assignment
  1's `assignment1.ipynb` does.
- `requirements.txt` — `numpy`, `pandas`, `matplotlib`, `requests`.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace assignment2.ipynb
```

or open `assignment2.ipynb` in Jupyter and Restart & Run All.

### Statement about AI use

Jeremy wrote the actual n-gram engine, the perplexity function, the
generation function, the model-selection grid search and its writeup, and
all of the notebook's analysis — none of that logic was changed here.

An AI assistant (Claude) was used, after the fact, to:
- Pull `NGramLM` (cell 12), `perplexity` (cell 20) and `generate` (cell 31)
  out of the notebook into `ngram.py`/`evaluation.py`, verbatim, and replace
  those three cells with the matching `import` statements, so this folder
  follows the same "shared engine lives in a `.py` module" structure as
  Assignment 1 and Assignment 4, and so Assignment 3/4 can reuse this code
  properly instead of re-implementing it.
- Add a local-file-first fallback to the data-loading cell (it now reads
  `data/shakespeare.txt` if present, and only hits the network if it isn't),
  matching Assignment 1's pattern, and copy that file into `data/`.
- Re-run the notebook top to bottom (**Restart and Run All**) after these
  changes to confirm every cell still executes and produces the same kind
  of output as before.

Every group member is expected to read through `ngram.py`, `evaluation.py`
and the notebook and be able to explain any part of it in the review
session, per rule 5 of the assignment.
