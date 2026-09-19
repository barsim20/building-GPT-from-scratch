# building-GPT-from-scratch

Coursework for the *GPT from scratch* module. Each assignment lives in its
own top-level folder.

- [`assignment1/`](assignment1/) — **Byte Pair Encoding from scratch.** A
  BPE tokenizer implemented with only the Python standard library, trained
  on the tiny Shakespeare corpus, with the required experiments and
  measures. See `assignment1/README.md` for details, how to run it, and
  the contribution table.
- [`assignment4/`](assignment4/) — **A small GPT.** A from-scratch GPT
  (own attention, own causal mask, no `nn.Transformer`), trained on the
  same Shakespeare corpus, compared against a count n-gram and a neural
  n-gram, with a generalization check on Wall Street Journal text. See
  `assignment4/README.md` for details, how to run it, and the
  contribution table.
