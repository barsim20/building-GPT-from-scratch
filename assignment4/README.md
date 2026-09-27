| Name | Student number | Tasks | Share |
|---|---|---|---|
| Baran | _TODO_ | Assignment 1 (BPE tokenizer) | 25 % |
| Jeremy | _TODO_ | Assignment 2 (count n-gram) | 25 % |
| Jonas | _TODO_ | Assignment 3 (neural n-gram) | 25 % |
| Baran, Jeremy, Jonas | _TODO_ | Assignment 4 (this folder: GPT), together | 25 % |

> Unlike Assignments 1–3, this one really was a three-way effort: all three
> of us worked on the GPT model, the integration with Assignments 2 and 3,
> and the experiments below. Shares are stated across the four assignments
> as a whole; see the root [`README.md`](../README.md) for the full
> breakdown. Student numbers are still placeholders — fill them in by hand
> before the ZIP is handed in.

One sentence about each member's work, in their own words: _TODO — replace
this line for each member._

## Assignment 4 — A Small GPT

Small GPT built from scratch (own attention, own causal mask, no
`nn.Transformer` / `nn.MultiheadAttention`), trained on the same
`<bos>`/`<eos>`-augmented BPE tokens as Assignments 2 and 3, compared
against the count n-gram (A2) and the neural n-gram (A3), plus a
generalization check on Wall Street Journal text.

### Contents

- `gpt.py` — `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`,
  `get_batch`, `estimate_loss`, `train`. New code for this assignment.
- `ngram.py` — `NGramLM`, Jeremy's real count-based n-gram engine from
  Assignment 2 (`fit(sequences)`, add-one smoothing
  `P(w|h) = (c(h,w)+1)/(c(h)+V)`, no special-casing of zero counts since the
  formula already does the right thing), copied over verbatim.
- `neural_ngram.py` — `NeuralNGramLM`, Jonas's real model from Assignment 3
  (embed the n-1 context tokens, concat, one hidden layer, softmax over V),
  copied over verbatim. `get_batch`/`estimate_loss`/`train` below it are
  this assignment's own training utilities (Assignment 3 kept that part
  inline in its notebook rather than in a module) — they train the real
  class above without changing anything inside it.
- `evaluation.py` — `perplexity` and `generate`, this assignment's own
  evaluation harness: it walks the *full* growing context over a long
  token stream and lets each model crop it to whatever it needs, which is
  what lets one pair of functions score the count n-gram, the neural
  n-gram and the GPT alike. This is deliberately **not** Assignment 2's
  `perplexity`/`generate` (those window context to exactly `n-1` tokens by
  convention, correct for an n-gram but it would cut the GPT off after a
  single token of context and defeat the entire comparison). What changed
  here is `context_length`/`crop_context`: two small helpers so this
  harness calls the real `NGramLM` from Assignment 2 exactly the way its
  own notebook does (a `(n-1)`-token window handed in from outside, since
  `NGramLM` doesn't crop it itself) while leaving the neural n-gram and the
  GPT untouched (they already crop internally). Also where the `<bos>`/
  `<eos>` convention shared by every model lives: they aren't part of the
  BPE vocabulary itself, so `bos_id(tok) = tok.vocab_size()` and
  `eos_id(tok) = tok.vocab_size() + 1`, and every model is constructed
  with `vocab_size = tok.vocab_size() + 2`.
- `bpe.py` — the Assignment 1 tokenizer (`BPETokenizer`, `normalize`),
  reused as is.
- `assignment4.ipynb` — the notebook with all five tasks, all output cells
  kept, runs top to bottom on a CPU with *Restart and Run All*.
- `build_notebook.py` — the script that generates `assignment4.ipynb`
  (kept for reproducibility, not required to run the assignment).
- `data/shakespeare.txt` — same corpus as Assignment 1–3. Split the
  Assignment 2 way: by line, 80 % / 10 % / 10 % train/val/test, no
  shuffling.
- `data/wsj_test.txt` — the exact Assignment 3 second-domain file (Penn
  Treebank `ptb.test.txt`, 1989 Wall Street Journal text). The notebook
  runs the sheet's own check on it (3761 lines, md5
  `8b80168b89c18661a38ef683c0dc3721`) and replaces the literal `<unk>`
  strings already in the file with `unknown` before encoding, so they
  don't collide with our own `<unk>` token.
- `requirements.txt`.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace assignment4.ipynb
```

or open in Jupyter and *Restart & Run All*. `FORCE_RETRAIN` at the top is
left at `True` since we did not bother saving checkpoints — everything,
including the six-run hyperparameter sweep, comfortably finishes inside
the time budget (see below), so caching did not seem worth the extra
complexity.

### Hyperparameters actually used

CPU column from the sheet everywhere (`block_size=64`, `n_embd=128`,
`n_head=4`, `n_layer=3`, `batch_size=32`, `lr=1e-3`, `dropout=0.1`), with
`max_steps=2000` for the one required run. The neural n-gram baseline uses
the Assignment 3 starting config (`n=3`, `n_embd=64`, `n_hidden=256`,
`batch_size=64`, `lr=1e-3`, `max_steps=2000`); the count n-gram uses the
same `n=3`.

For Experiment 1 (vary k) and Experiment 2 (the six-run hyperparameter
grid) we reduced the GPT's `max_steps` to 600 — ten full 2000-step runs
back to back would have pushed close to an hour of wall clock for very
little extra signal on a model this small, and the assignment only
requires the *one* full run to respect the 2000-step / 10-minute rule.
Perplexity is also computed on the first 3000 tokens of each split rather
than the whole split, again purely to keep total runtime reasonable —
scoring is a forward pass per token so it does not benefit from batching
the way training does.

We did not run the GPU column (no GPU available on the machine we used).

### Run time, machine, seed

- Seed: `1337` (`torch.manual_seed`, also seeds `random` and `numpy`).
- Machine: Linux container, CPU only, standard CPython, PyTorch CPU build.
- Total notebook run time: about **12 minutes**, top to bottom (the
  required 2000-step run alone is under 2.5 minutes, well inside the
  10-minute budget; the rest is the k-sweep, the hyperparameter grid, and
  the neural n-gram baseline).

### Final numbers (Experiment 3)

| Model | Shakespeare PP/char | WSJ PP/char | Params | Train time |
|---|---|---|---|---|
| Count n-gram (A2) | 9.63 | 13.55 | 235,079 counted n-grams + contexts | 0.3 s |
| Neural n-gram (A3) | 5.91 | 9.39 | 375,531 | 44.5 s |
| GPT (A4) | 4.84 | 9.00 | 876,331 | 142.0 s |

All three checks in the notebook pass: pre-training loss sits at 7.12 vs.
`ln(V_total) = 6.97`, the causal-mask sanity check shows a 0.0 difference
at position 0 and a nonzero difference at the last position when the last
token is perturbed, and the GPT's validation loss (3.90, `<bos>` excluded)
beats the neural n-gram's training loss (4.42, same convention) on the
same k. Generation now actually stops at `<eos>` on its own most of the
time (100 % for the neural n-gram and the GPT in our 20-try check, 10 %
for the count n-gram, which loses the thread too quickly to ever reach a
sentence end).

These numbers are Jeremy's real `NGramLM` and Jonas's real `NeuralNGramLM`
running inside this notebook now (see "Statement about AI use" below), not
re-implementations of them — and they come out effectively identical to
the ones this README reported before that swap (same formulas, same
architecture, same seed), which is exactly what should happen. The one
number that did move is the count n-gram's parameter count: it's now
counted the way Assignments 2 and 3 themselves count it (distinct n-grams
plus distinct contexts stored), rather than the old ad-hoc count.

### Statement about AI use

An AI assistant (Claude) helped write the GPT implementation
(`gpt.py`: the attention head, multi-head attention, feedforward block,
and the full model) from the written specification in the assignment
sheet, drafted the notebook structure and the experiment/plotting code,
and set up the Wall Street Journal file exactly as Assignment 3 specifies
(same URL, same md5/line-count check, same `<unk>` → `unknown` cleanup).

`ngram.py` and `neural_ngram.py` originally held from-scratch
re-implementations of `NGramLM` and `NeuralNGramLM`, written to match
Assignments 2 and 3's required interfaces as faithful stand-ins, because
Assignments 2 and 3 didn't exist as real code yet at the time. Now that
they do, an AI assistant (Claude) replaced those stand-ins with Jeremy's
and Jonas's actual classes, copied over verbatim, and adjusted only
`evaluation.py`'s two small `context_length`/`crop_context` helpers (plus
the one cell that counted the count n-gram's stored entries, and one that
primed generation with the right number of `<bos>` tokens) so this
notebook calls the real `NGramLM` the same way its own Assignment 2
notebook does. Nothing inside `NGramLM.fit`/`log_prob`, or inside
`NeuralNGramLM.forward`/`log_prob`, was changed; `gpt.py` was not touched
either. The notebook was re-executed end to end (**Restart and Run All**)
after the swap; the numbers above are from that run.

Every group member is expected to be able to walk through `gpt.py` line
by line in the review session, per rule 5 of the assignment.
