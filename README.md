# building-GPT-from-scratch

Coursework for the *GPT from scratch* module — four assignments, one group,
each building on the one before it:

```
Assignment 1 ──▶ Assignment 2 ──▶ Assignment 3 ──▶ Assignment 4
 BPE tokenizer     count n-gram     neural n-gram      GPT
   (Baran)           (Jeremy)          (Jonas)      (all three)
```

Every later assignment imports the earlier ones' actual code (`bpe.py`,
`ngram.py`, `neural_ngram.py`, `evaluation.py`) rather than re-implementing
it — see "How the assignments connect" below.

- [`assignment1/`](assignment1/) — **Byte Pair Encoding from scratch.** A
  BPE tokenizer implemented with only the Python standard library, trained
  on the tiny Shakespeare corpus, with the required experiments and
  measures. See [`assignment1/README.md`](assignment1/README.md) for
  details, how to run it, and the contribution table.
- [`assignment2/`](assignment2/) — **N-Grams.** A count-based n-gram
  language model with add-one smoothing, perplexity, and generation, built
  on Assignment 1's tokenizer. See
  [`assignment2/README.md`](assignment2/README.md) for details, how to run
  it, and the contribution table.
- [`assignment3/`](assignment3/) — **Neural N-Gram Language Model.** A
  small Bengio-style MLP n-gram model, compared against Assignment 2's
  count n-gram, with a generalization check on Wall Street Journal text.
  See [`assignment3/README.md`](assignment3/README.md) for details, how to
  run it, and the contribution table.
- [`assignment4/`](assignment4/) — **A small GPT.** A from-scratch GPT
  (own attention, own causal mask, no `nn.Transformer`), trained on the
  same Shakespeare corpus, compared against Assignment 2's count n-gram and
  Assignment 3's neural n-gram, with the same Wall Street Journal
  generalization check. See [`assignment4/README.md`](assignment4/README.md)
  for details, how to run it, and the contribution table.

Each folder also has its own copy of the assignment sheet it implements
(`BPE_Assignment.pdf`, `NGram_Assignment.pdf`, `Neural_NGram_Assignment.pdf`,
`GPT_Assignment.pdf`) and is runnable on its own with `pip install -r
requirements.txt` + `jupyter nbconvert --to notebook --execute --inplace
<notebook>.ipynb`, per rule set of the course (corpora and PDFs are kept
here for convenience since this is a git repository, not a StudIP ZIP;
they wouldn't normally ship in the submission itself).

## How the assignments connect

Each folder keeps its own copy of the modules it depends on (so every
assignment still runs standalone, the way the sheets require), but the
*content* of those modules only ever comes from the assignment that first
wrote it — nothing here re-implements another member's work:

- `bpe.py` (`BPETokenizer`, `normalize`) — written for Assignment 1,
  copied unchanged into Assignments 2, 3 and 4.
- `ngram.py` (`NGramLM`) and `evaluation.py` (`perplexity`, `generate`) —
  written for Assignment 2, copied unchanged into Assignment 3.
  Assignment 3's own notebook originally kept local, explicitly-marked
  `# TEMPORARY ASSIGNMENT 2 CODE` stand-ins for these (with a comment
  saying exactly what to import instead once Assignment 2 existed); that
  swap is now done, so Assignment 3 evaluates its neural n-gram with
  Jeremy's real count n-gram and Jeremy's real perplexity/generate
  functions, not a copy of them.
- `neural_ngram.py` (`NeuralNGramLM`) — written for Assignment 3, copied
  unchanged into Assignment 4.
- Assignment 4 has its own `evaluation.py`. Assignment 2/3's `perplexity`/
  `generate` are windowed to exactly `n-1` tokens of context by the
  caller, which is correct for an n-gram but would cut the GPT off after a
  single token of context and defeat the entire comparison — so Assignment
  4 keeps its own evaluation harness (walking the full growing context,
  which every model then crops to whatever *it* needs), while importing
  the real `NGramLM` and `NeuralNGramLM` classes from Assignments 2 and 3
  verbatim instead of re-implementing them. `gpt.py` (`Head`,
  `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`) is new code written
  for Assignment 4.

In short: whenever one assignment's notebook uses another assignment's
model or function, it's calling the real thing that group member wrote, not
a stand-in for it.

## Contributions

| Name | Student number | Primary responsibility | Share |
|---|---|---|---|
| Baran | _TODO_ | Assignment 1 — BPE tokenizer from scratch | 25 % |
| Jeremy | _TODO_ | Assignment 2 — count n-gram, perplexity, generation | 25 % |
| Jonas | _TODO_ | Assignment 3 — neural n-gram, training, experiments | 25 % |
| Baran, Jeremy, Jonas | _TODO_ | Assignment 4 — GPT, together | 25 % |

Each of us owned one assignment end to end (implementation, experiments,
and the write-up in that folder's notebook and README), and all three of us
worked together on Assignment 4, which ties the other three together. That
puts each member's overall share of the project at an equal 25 %: Baran,
Jeremy and Jonas each did one full assignment on their own (25 % each), and
Assignment 4 was split evenly between the three of us (a further 8.3 %
each), which nets out to a straight 25/25/25/25 once Assignment 4's third is
folded back into each person's own total.

> The per-assignment tables inside `assignment1/README.md` through
> `assignment4/README.md` repeat this same breakdown (as the sheets for
> each assignment require); this table is the one place it's stated for
> the project as a whole. Student numbers are still `_TODO_` — fill them in
> by hand, and add a one-sentence note in your own words for each member,
> before any of this is handed in.

### Statement about AI use

An AI assistant (Claude) was used throughout, per assignment, for the
specific things each folder's own README states (drafting code from a
written specification, building the notebook, writing analysis prose,
running experiments) — see each `assignment*/README.md` for exactly what
it did on that assignment. It was also used, after all four assignments
existed, to restructure the repository: turning the two uploaded raw
notebooks (Assignment 2 and Assignment 3) into folders with the same
layout as Assignments 1 and 4, extracting Jeremy's and Jonas's reusable
code into `.py` modules alongside their notebooks, wiring Assignment 3 and
4's already-flagged temporary stand-ins over to the real Assignment 2/3
code, and re-running every notebook end to end to confirm nothing broke.
No group member's algorithm, analysis, or written answers were rewritten
in the process — see the "How the assignments connect" section above and
each folder's own AI-use statement for exactly what changed and why.

Every group member is expected to be able to walk through their own
assignment's code in the review session, per rule 5 of the assignment
sheets.
