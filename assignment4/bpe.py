"""
Byte Pair Encoding (BPE) tokenizer, written from scratch.

This module implements everything the assignment asks for:
  - normalize(text, strategy): four text-cleaning strategies
  - BPETokenizer: trains merge rules on a corpus, then encodes / decodes text
  - WordTokenizer: a simple whole-word baseline tokenizer, used for comparison

Only the Python standard library is used for the algorithm itself
(re, json, time, unicodedata, collections, heapq). No tokenizer library
(sentencepiece, tokenizers, transformers, tiktoken, subword-nmt) is used.

Design notes for the trainer (why it is fast)
-----------------------------------------------
A naive BPE trainer recounts every pair of adjacent symbols, in every word,
after every single merge. That is correct but far too slow for k = 8000
merges on ~30,000 word types.

Instead this trainer:
  1. Counts pairs only ONCE at the start (`_count_pairs`).
  2. After a merge, only revisits the word types that actually contained the
     merged pair (`pair_to_words`), and only updates the pair counts that
     changed inside those words. Word types that never had the pair are
     left untouched.
  3. Uses a lazy-deletion max-heap to find "the most frequent pair" in
     O(log P) instead of scanning every pair (O(P)) at every step.

Determinism
-----------
Ties (two pairs with the same count) are broken by picking the pair that
is first in alphabetical order, i.e. the smaller Python tuple. This makes
`heap ordering (-count, pair)` do the right thing automatically: for equal
counts, the smaller pair tuple naturally sorts first.
"""

from __future__ import annotations

import heapq
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

UNK_TOKEN = "<unk>"


# ---------------------------------------------------------------------------
# Task 1: normalization
# ---------------------------------------------------------------------------

_DIGIT_RE = re.compile(r"\d")
_SPACE_RUN_RE = re.compile(r" +")
# "word" characters (kept together) vs. everything else (treated as
# punctuation and given its own spaces around it in the "split" strategy).
_PUNCT_RE = re.compile(r"([^\w\s])", re.UNICODE)


def normalize(text: str, strategy: str) -> str:
    """Normalize `text` using one of four strategies.

    raw   : unchanged.
    lower : lowercase only.
    clean : lower + Unicode NFKC + every digit -> '0' + collapse spaces.
    split : clean + put a space around every punctuation mark.
    """
    if strategy == "raw":
        return text
    if strategy == "lower":
        return text.lower()
    if strategy == "clean":
        t = text.lower()
        t = unicodedata.normalize("NFKC", t)
        t = _DIGIT_RE.sub("0", t)
        t = _SPACE_RUN_RE.sub(" ", t)
        return t
    if strategy == "split":
        t = normalize(text, "clean")
        t = _PUNCT_RE.sub(r" \1 ", t)
        t = _SPACE_RUN_RE.sub(" ", t).strip()
        return t
    raise ValueError(f"unknown normalization strategy: {strategy!r}")


NORMALIZE_STRATEGIES = ("raw", "lower", "clean", "split")


# ---------------------------------------------------------------------------
# word <-> symbol helpers (word-boundary markers)
# ---------------------------------------------------------------------------

def _word_to_symbols(word: str, end_of_word: str, marker_mode: str) -> List[str]:
    """Split a word into a list of one-character symbols plus the boundary
    marker, so that BPE can never merge symbols across a word boundary."""
    chars = list(word)
    if marker_mode == "suffix":
        return chars + [end_of_word]
    elif marker_mode == "prefix":
        return [end_of_word] + chars
    raise ValueError(f"unknown marker_mode: {marker_mode!r}")


def _symbols_to_word(symbols: List[str], end_of_word: str, marker_mode: str) -> str:
    """Undo `_word_to_symbols`: join symbols back into a plain word string,
    dropping the boundary marker."""
    text = "".join(symbols)
    if marker_mode == "suffix":
        if text.endswith(end_of_word):
            text = text[: -len(end_of_word)]
    else:
        if text.startswith(end_of_word):
            text = text[len(end_of_word):]
    return text


# ---------------------------------------------------------------------------
# Task 2: the BPE tokenizer
# ---------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self, num_merges: int = 1000, end_of_word: str = "</w>",
                 marker_mode: str = "suffix", strategy: str = "clean"):
        """
        num_merges  : k, the number of merge rules to learn.
        end_of_word : the marker string/character used at word boundaries.
        marker_mode : "suffix" (marker glued to the end of the word, the
                      default used for the main experiments) or "prefix"
                      (marker glued to the start of the word).
        strategy    : which `normalize` strategy encode() applies to input
                      text before tokenizing, so encode/decode are always
                      consistent with how the tokenizer was trained.
        """
        self.num_merges = num_merges
        self.end_of_word = end_of_word
        self.marker_mode = marker_mode
        self.strategy = strategy

        self.merges: List[Tuple[str, str]] = []     # learned order
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.vocab: Dict[str, int] = {}              # token string -> id
        self.base_chars: List[str] = []              # single-character alphabet
        self._base_char_set: set = set()

        self.train_time_seconds: float = 0.0
        self.unk_count: int = 0  # count of <unk> substitutions since last reset

    # -- training -----------------------------------------------------------

    def train(self, text: str, verbose: bool = False) -> None:
        t0 = time.time()

        # Step 1: word types and their counts (work on TYPES, not the full
        # text -- this is what makes the trainer fast).
        words = text.split()
        word_counts = Counter(words)

        # Step 2: split every word type into symbols (chars + marker).
        seqs: List[List[str]] = []
        weights: List[int] = []
        for w, c in word_counts.items():
            seqs.append(_word_to_symbols(w, self.end_of_word, self.marker_mode))
            weights.append(c)

        base_chars = set()
        for seq in seqs:
            base_chars.update(seq)
        self.base_chars = sorted(base_chars)
        self._base_char_set = set(self.base_chars)

        # Step 3: initial pair counts, counted once.
        pair_counts: Counter = Counter()
        pair_to_words: Dict[Tuple[str, str], set] = defaultdict(set)
        for i, seq in enumerate(seqs):
            w = weights[i]
            for a, b in zip(seq, seq[1:]):
                pair_counts[(a, b)] += w
                pair_to_words[(a, b)].add(i)

        # Max-heap (via negated counts) with lazy deletion: an entry is
        # valid only if it still matches the pair's current count.
        heap = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        def push(pair):
            heapq.heappush(heap, (-pair_counts[pair], pair))

        def pop_best():
            while heap:
                neg_c, p = heap[0]
                if pair_counts.get(p, 0) == -neg_c and -neg_c > 0:
                    return p
                heapq.heappop(heap)  # stale entry, discard
            return None

        merges: List[Tuple[str, str]] = []
        vocab = {UNK_TOKEN: 0}
        for ch in self.base_chars:
            vocab[ch] = len(vocab)

        for step in range(self.num_merges):
            best_pair = pop_best()
            if best_pair is None:
                break
            heapq.heappop(heap)  # consume the entry we just used

            new_symbol = best_pair[0] + best_pair[1]
            merges.append(best_pair)
            vocab[new_symbol] = len(vocab)

            affected = list(pair_to_words.get(best_pair, ()))
            for i in affected:
                seq = seqs[i]
                w = weights[i]
                if best_pair[0] not in seq:
                    continue

                old_pairs = list(zip(seq, seq[1:]))

                new_seq: List[str] = []
                j = 0
                n = len(seq)
                while j < n:
                    if (j < n - 1 and seq[j] == best_pair[0]
                            and seq[j + 1] == best_pair[1]):
                        new_seq.append(new_symbol)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                seqs[i] = new_seq
                new_pairs = list(zip(new_seq, new_seq[1:]))

                if new_pairs == old_pairs:
                    continue

                old_c = Counter(old_pairs)
                new_c = Counter(new_pairs)
                for p, cnt in old_c.items():
                    pair_counts[p] -= cnt * w
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        pair_to_words.pop(p, None)
                    else:
                        pair_to_words[p].discard(i)
                for p, cnt in new_c.items():
                    pair_counts[p] += cnt * w
                    pair_to_words[p].add(i)
                    push(p)

            pair_to_words.pop(best_pair, None)
            pair_counts.pop(best_pair, None)

            if verbose and (step + 1) % 1000 == 0:
                print(f"  merge {step + 1}/{self.num_merges}: {best_pair} -> {new_symbol!r}")

        self.merges = merges
        self.merge_ranks = {p: i for i, p in enumerate(merges)}
        self.vocab = vocab
        self.train_time_seconds = time.time() - t0

    # -- encode / decode ------------------------------------------------

    def _encode_word(self, word: str) -> List[str]:
        symbols = _word_to_symbols(word, self.end_of_word, self.marker_mode)
        # any character never seen during training becomes <unk>; <unk> is
        # atomic and never takes part in further merges.
        symbols = [s if (s in self._base_char_set) else UNK_TOKEN for s in symbols]

        if len(symbols) < 2:
            return symbols

        while True:
            # find the adjacent pair with the lowest merge rank (i.e. the
            # pair that was learned earliest) among pairs present.
            best_rank = None
            best_idx = None
            for idx in range(len(symbols) - 1):
                pair = (symbols[idx], symbols[idx + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = idx
            if best_idx is None:
                break
            a, b = symbols[best_idx], symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [a + b] + symbols[best_idx + 2:]
        return symbols

    def encode(self, text: str, count_unk: bool = True) -> List[int]:
        """Normalize `text` (using the strategy the tokenizer was trained
        with), split it into words on whitespace, and apply the learned
        merge rules to each word."""
        norm = normalize(text, self.strategy)
        ids: List[int] = []
        unk_local = 0
        for word in norm.split():
            for sym in self._encode_word(word):
                if sym == UNK_TOKEN:
                    unk_local += 1
                ids.append(self.vocab.get(sym, self.vocab[UNK_TOKEN]))
        if count_unk:
            self.unk_count += unk_local
        return ids

    def decode(self, ids: List[int]) -> str:
        id_to_token = {i: t for t, i in self.vocab.items()}
        symbols = [id_to_token.get(i, UNK_TOKEN) for i in ids]

        words: List[List[str]] = []
        current: List[str] = []
        for sym in symbols:
            current.append(sym)
            ends_word = (
                (self.marker_mode == "suffix" and sym.endswith(self.end_of_word))
                or (self.marker_mode == "prefix" and False)  # handled below
            )
            if ends_word:
                words.append(current)
                current = []
        if current:
            words.append(current)

        if self.marker_mode == "prefix":
            # re-split on the prefix marker instead
            words = []
            current = []
            for sym in symbols:
                if sym.startswith(self.end_of_word) and current:
                    words.append(current)
                    current = [sym]
                else:
                    current.append(sym)
            if current:
                words.append(current)

        out_words = [_symbols_to_word(w, self.end_of_word, self.marker_mode) for w in words]
        return " ".join(out_words)

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "num_merges": self.num_merges,
            "end_of_word": self.end_of_word,
            "marker_mode": self.marker_mode,
            "strategy": self.strategy,
            "merges": [[a, b] for a, b in self.merges],
            "vocab": self.vocab,
            "base_chars": self.base_chars,
            "train_time_seconds": self.train_time_seconds,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        tok = cls(
            num_merges=data["num_merges"],
            end_of_word=data["end_of_word"],
            marker_mode=data["marker_mode"],
            strategy=data.get("strategy", "clean"),
        )
        tok.merges = [tuple(p) for p in data["merges"]]
        tok.merge_ranks = {p: i for i, p in enumerate(tok.merges)}
        tok.vocab = data["vocab"]
        tok.base_chars = data["base_chars"]
        tok._base_char_set = set(tok.base_chars)
        tok.train_time_seconds = data.get("train_time_seconds", 0.0)
        return tok

    def vocab_size(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Word tokenizer baseline (Experiment 1)
# ---------------------------------------------------------------------------

class WordTokenizer:
    """Keeps the `max_words` most frequent whole words; everything else
    maps to <unk>. Used as a simple baseline against BPE."""

    def __init__(self, max_words: int = 8000, strategy: str = "clean"):
        self.max_words = max_words
        self.strategy = strategy
        self.vocab: Dict[str, int] = {}
        self.train_time_seconds: float = 0.0
        self.unk_count: int = 0

    def train(self, text: str) -> None:
        t0 = time.time()
        counts = Counter(text.split())
        most_common = [w for w, _ in counts.most_common(self.max_words)]
        self.vocab = {UNK_TOKEN: 0}
        for w in most_common:
            self.vocab[w] = len(self.vocab)
        self.train_time_seconds = time.time() - t0

    def encode(self, text: str, count_unk: bool = True) -> List[int]:
        norm = normalize(text, self.strategy)
        ids = []
        unk_local = 0
        for w in norm.split():
            if w in self.vocab:
                ids.append(self.vocab[w])
            else:
                ids.append(self.vocab[UNK_TOKEN])
                unk_local += 1
        if count_unk:
            self.unk_count += unk_local
        return ids

    def decode(self, ids: List[int]) -> str:
        id_to_token = {i: t for t, i in self.vocab.items()}
        return " ".join(id_to_token.get(i, UNK_TOKEN) for i in ids)

    def vocab_size(self) -> int:
        return len(self.vocab)
