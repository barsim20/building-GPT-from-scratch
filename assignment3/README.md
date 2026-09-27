| Name | Student number | Tasks | Share |
|---|---|---|---|
| Baran | _TODO_ | Assignment 1 (BPE tokenizer) | 25 % |
| Jeremy | _TODO_ | Assignment 2 (count n-gram) | 25 % |
| Jonas | _TODO_ | Assignment 3 (this folder: neural n-gram) | 25 % |
| Baran, Jeremy, Jonas | _TODO_ | Assignment 4 (GPT), together | 25 % |

> Shares are across the four assignments as a whole, not within this single
> folder: Jonas owns Assignment 3 end to end, the same way Baran owns
> Assignment 1 and Jeremy owns Assignment 2, and all three worked together on
> Assignment 4. See the root [`README.md`](../README.md) for the full
> breakdown.

One sentence about each member's work, in their own words: _TODO — replace
this line for each member._

## Assignment 3 — Neural N-Gram Language Model

This folder contains the neural n-gram language model (a small Bengio-style
MLP: embed the `n-1` context tokens, concatenate, one hidden layer, softmax
over `V`), trained and evaluated on the same corpus and the same BPE/`<bos>`/
`<eos>` convention as Assignment 2, per the "GPT from scratch · Assignment 3"
specification. It builds directly on Assignment 1 (`bpe.py`) and Assignment
2 (`ngram.py`, `evaluation.py`), and is in turn reused as-is by Assignment 4.

### Contents

- `bpe.py` — Assignment 1's tokenizer, reused unchanged.
- `ngram.py`, `evaluation.py` — Assignment 2's real `NGramLM` and
  `perplexity`/`generate`, reused unchanged, so Experiment 2 (count n-gram
  vs. neural n-gram) and Task 5 (generation) compare against and build on
  Jeremy's actual Assignment 2 code instead of a re-implementation of it.
- `neural_ngram.py` — `NeuralNGramLM`: `nn.Embedding` for the context
  tokens, one hidden `nn.Linear` + `ReLU`, one output `nn.Linear`, plus
  `log_prob`/`next_token_log_probs` so it drops into Assignment 2's
  `perplexity`/`generate` without any change to either. Jonas already wrote
  this as its own module in the original notebook (`%%writefile
  neural_ngram.py`); it's kept here exactly as he wrote it.
- `assignment3.ipynb` — the notebook with all tasks (batching, the model,
  the training loop, the three experiments, generation, the WSJ
  generalization check). All output cells are kept; it runs top to bottom
  with **Restart and Run All**. The three cells the notebook itself marked
  `# TEMPORARY ASSIGNMENT 2 CODE` (a local copy of `NGramLM`, and stand-in
  `perplexity_temp`/`generate_temp` functions, each with a comment saying
  what to replace them with) now do exactly that: `from ngram import
  NGramLM` and `from evaluation import perplexity, generate`, with the call
  sites updated to match (`generate` additionally takes the normalization
  `strategy`, which Assignment 2's version needs and the temporary stand-in
  didn't have). Nothing about what `NGramLM`, `perplexity` or `generate`
  themselves do changed — only that the notebook now calls Jeremy's real
  Assignment 2 code instead of a copy of it.
- `data/shakespeare.txt` — the same tiny Shakespeare corpus as Assignments
  1–2. `data/ptb.test.txt` — the Penn Treebank / Wall Street Journal test
  file used for the generalization check in Task 5.2 (same file reused
  again in Assignment 4). Both are downloaded automatically by the notebook
  if missing.
- `requirements.txt` — `torch`, `numpy`, `pandas`, `matplotlib`, `requests`.

### How to run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace assignment3.ipynb
```

or open `assignment3.ipynb` in Jupyter and Restart & Run All.

### Statement about AI use

Jonas wrote the actual `NeuralNGramLM` model, the training loop, the three
experiments and their writeup, the generation code, and the Wall Street
Journal generalization check — none of that logic was changed here. Jonas's
own notebook already flagged exactly which cells were temporary
Assignment-2 stand-ins and what they should be replaced with once
Assignment 2 existed as real code.

An AI assistant (Claude) was used, after the fact, to:
- Do exactly what those comments asked: replace the temporary `NGramLM`
  copy and the `perplexity_temp`/`generate_temp` stand-ins with imports of
  Jeremy's real Assignment 2 code, and update the handful of call sites
  that needed the extra `strategy` argument `generate` takes.
- Add a local-file-first fallback to both data-loading cells (Shakespeare
  and the Wall Street Journal file), matching Assignments 1 and 2, and copy
  both files into `data/`.
- Re-run the notebook top to bottom (**Restart and Run All**) after these
  changes to confirm every cell still executes; the reported perplexities,
  hyperparameter sweep and generated samples are Jonas's actual model
  running against Jeremy's actual n-gram code, not a re-implementation of
  either.

Every group member is expected to read through `neural_ngram.py` and the
notebook and be able to explain any part of it in the review session, per
rule 5 of the assignment.
