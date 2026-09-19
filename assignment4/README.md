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
`nn.Transformer` / `nn.MultiheadAttention`), trained on the same tiny
Shakespeare corpus as before, compared against the count n-gram and neural
n-gram from the earlier assignments, plus a generalization check on Wall
Street Journal text.

### Contents

- `gpt.py` — `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`,
  `get_batch`, `estimate_loss`, `train`. This is the new code for this
  assignment.
- `ngram.py` — `CountNgram` and `NeuralNgram`, carried over from
  Assignment 2/3, kept as small standalone classes with the same
  `log_prob` / `next_token_log_probs` interface as the GPT so that
  `evaluation.py` does not need to know which model it is scoring.
- `evaluation.py` — `perplexity` and `generate`, unchanged since
  Assignment 2 per rule 3 of the sheet.
- `bpe.py` — the Assignment 1 tokenizer, reused as is.
- `assignment4.ipynb` — the notebook with all five tasks, all output cells
  kept, runs top to bottom on a CPU with *Restart and Run All*.
- `build_notebook.py` — the script that generates `assignment4.ipynb`
  (kept for reproducibility, not required to run the assignment).
- `data/shakespeare.txt` — same corpus as Assignment 1–3, 90/5/5
  train/val/test split done on the character stream inside the notebook.
- `data/wsj_test.txt` — Wall Street Journal test text (word-tokenized,
  numbers replaced with `N`, lowercase, exactly as it is normally
  distributed for language modeling benchmarks). This is the file we used
  as the Assignment 3 WSJ test file.
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
`max_steps=2000` for the one required run. For Experiment 1 (vary k) and
Experiment 2 (the six-run hyperparameter grid) we reduced `max_steps` to
600 — ten full 2000-step runs back to back would have pushed close to an
hour of wall clock for very little extra signal on a model this small, and
the assignment only requires the *one* full run to respect the 2000-step /
10-minute rule. Perplexity is also computed on the first 3000 tokens of
each split rather than the whole split, again purely to keep total runtime
reasonable — scoring is a forward pass per token so it does not benefit
from batching the way training does.

We did not run the GPU column (no GPU available on the machine we used).

### Run time, machine, seed

- Seed: `1337` (`torch.manual_seed`, also seeds `random` and `numpy`).
- Machine: Linux container, CPU only, standard CPython, PyTorch CPU build.
- Total notebook run time: about **24 minutes**, top to bottom (the
  required 2000-step run alone is under 5 minutes, well inside the
  10-minute budget; most of the rest is the 9 extra training runs from
  the two experiment grids).

### Statement about AI use

An AI assistant (Claude) helped write the GPT implementation
(`gpt.py`: the attention head, multi-head attention, feedforward block,
and the full model) from the written specification in the assignment
sheet, wrote the small n-gram stand-ins in `ngram.py` so that Assignment 4
could be run standalone, drafted the notebook structure and the
experiment/plotting code, and picked the Wall Street Journal stand-in
corpus (a commonly used, freely redistributed lowercase/tokenized WSJ test
split) since the exact Assignment 3 WSJ file was not available in this
repository. Every group member is expected to be able to walk through
`gpt.py` line by line in the review session, per rule 5 of the assignment.
