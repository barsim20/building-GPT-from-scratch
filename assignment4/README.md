| Name | Student number | Tasks | Share |
|---|---|---|---|
| _TODO: fill in_ | _TODO_ | _TODO_ | _TODO %_ |
| _TODO: fill in_ | _TODO_ | _TODO_ | _TODO %_ |
| _TODO: fill in_ | _TODO_ | _TODO_ | _TODO %_ |

> The shares must sum to 100. All group members must agree to this table
> before submission. **This table is a placeholder** — it must be filled in
> by hand with real names, student numbers, tasks and shares before the ZIP
> is handed in.

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
- `ngram.py` — `NGramLM`, the count-based n-gram engine from Assignment 2
  (`fit(sequences)`, add-one smoothing `P(w|h) = (c(h,w)+1)/(c(h)+V)`, no
  special-casing of zero counts since the formula already does the right
  thing), reused unchanged.
- `neural_ngram.py` — `NeuralNGramLM` from Assignment 3 (embed the n-1
  context tokens, concat, one hidden layer, softmax over V), plus its
  `train` function, reused unchanged.
- `evaluation.py` — `perplexity` and `generate`, unchanged since
  Assignment 2 per rule 3 of the sheet. Also where the `<bos>`/`<eos>`
  convention shared by every model lives: they aren't part of the BPE
  vocabulary itself, so `bos_id(tok) = tok.vocab_size()` and
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
- Total notebook run time: _see the printed run time in the last cell of
  the notebook, copied here after the final run_.

### Statement about AI use

An AI assistant (Claude) helped write the GPT implementation
(`gpt.py`: the attention head, multi-head attention, feedforward block,
and the full model) from the written specification in the assignment
sheet, wrote `ngram.py` and `neural_ngram.py` to match the exact required
interfaces from Assignments 2 and 3 (`NGramLM.fit`, `NeuralNGramLM`, the
`<bos>`/`<eos>`/`<unk>` vocabulary convention) so that Assignment 4 could
be run standalone against faithful stand-ins for the earlier assignments,
drafted the notebook structure and the experiment/plotting code, and set
up the Wall Street Journal file exactly as Assignment 3 specifies
(same URL, same md5/line-count check, same `<unk>` → `unknown` cleanup).
Every group member is expected to be able to walk through `gpt.py` line
by line in the review session, per rule 5 of the assignment.
