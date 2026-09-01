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

## Assignment 1 — Byte Pair Encoding from Scratch

This folder contains a from-scratch Byte Pair Encoding (BPE) tokenizer,
trained and evaluated on the *tiny Shakespeare* corpus, per the
"GPT from scratch · Assignment 1" specification.

### Contents

- `bpe.py` — the tokenizer implementation (`normalize`, `BPETokenizer`,
  `WordTokenizer`), standard library only for the algorithm itself.
- `assignment1.ipynb` — the notebook that runs all five tasks: data
  preparation, merge training, encode/decode, the three experiments, and
  the measures (including the custom boundary-agreement accuracy measure).
  All output cells are kept; it runs top to bottom with **Restart and Run
  All**.
- `build_notebook.py` — the script that generates `assignment1.ipynb`
  programmatically (kept for reproducibility / re-generation; not required
  to run the assignment).
- `requirements.txt` — `numpy`, `pandas`, `matplotlib` (only used for
  tables/plots, per the rules).
- `data/` — the three corpora used. `shakespeare.txt` is downloaded
  automatically by the notebook if missing; `pride_and_prejudice.txt`
  (Group A, modern English prose) and `python_stdlib_source.txt`
  (Group B, non-literary text) are the two comparison corpora, both
  well over 200 kB. **Per the assignment rules, corpora are not meant to
  be handed in with the ZIP submission** — they are kept here only because
  this is a git repository, not a StudIP ZIP.
- `models/` — the saved merge files (`BPETokenizer.save`/`.load`), one per
  `k` value used in Experiment 1, one per normalization strategy used in
  Experiment 2, and one for the second word-boundary marker.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace assignment1.ipynb
```

or open `assignment1.ipynb` in Jupyter and Restart & Run All. `FORCE_RETRAIN`
at the top of the notebook can be set to `False` to load the cached models
in `models/` instead of retraining.

### Run time and machine

The full notebook (all `k` values up to 8000, all four normalization
strategies, both markers, all three experiments, the boundary-agreement
measure) ran end to end in about **64 seconds** on the machine used to
generate it (a Linux container, standard CPython 3.11, no GPU — BPE
training is pure Python on the standard library and does not need one).
Training `k = 8000` merges alone took about **3.3 seconds**, well inside
the 10-minute budget the assignment requires.

### Sources of the two comparison corpora

- **Group A (modern English prose):** *Pride and Prejudice* by Jane
  Austen, plain text from Project Gutenberg
  (`https://www.gutenberg.org/files/1342/1342-0.txt`), Gutenberg
  header/footer stripped, public domain.
- **Group B (non-literary text):** a concatenation of Python 3.11 standard
  library source files (`.py` files from the local CPython installation),
  chosen deterministically and shuffled with a fixed random seed for
  reproducibility.

### Statement about AI use

An AI assistant (Claude) was used to:
- Draft the `bpe.py` implementation (normalization, the trainer with its
  speed optimizations, encode/decode, save/load) from the assignment's
  written specification.
- Draft the notebook structure, the explanatory markdown text, and the
  experiment/plotting code.
- Select and download the two comparison corpora, and build a
  deterministic Group B corpus from local Python standard library source.
- Hand-write and sanity-check the 60-word morpheme list used for the
  boundary-agreement measure.

Every group member is expected to read through `bpe.py` and the notebook
and be able to explain any part of it in the review session, per rule 5 of
the assignment.
