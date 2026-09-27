from collections import Counter
import math
import numpy as np


class NGramLM:

    def __init__(self, n, vocab_size):

        # Only unigram, bigram and trigram models are allowed
        if n not in [1, 2, 3]:
            raise ValueError("n must be 1, 2 or 3")

        self.n = n
        self.vocab_size = vocab_size

        # Counts of complete n-grams
        self.ngram_counts = Counter()

        # Counts of the corresponding (n-1)-gram contexts
        self.context_counts = Counter()


    def fit(self, sequences):

        # Reset counts in case fit() is called again
        self.ngram_counts = Counter()
        self.context_counts = Counter()

        # Go through every sequence
        for sequence in sequences:

            # Start at the first token that should be predicted
            for i in range(self.n - 1, len(sequence)):

                # Previous n-1 tokens
                context = tuple(
                    sequence[i - self.n + 1:i]
                )

                # Current token
                token = sequence[i]

                # Complete n-gram
                ngram = context + (token,)

                # Count n-gram and context up
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1


    def log_prob(self, token, context):

        # Convert context to a tuple so it can be used as a dictionary key
        context = tuple(context)

        # Complete n-gram
        ngram = context + (token,)

        # Counts
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]



        # Add-one / Laplace smoothing
        probability = (
            (ngram_count + 1)
            /
            (context_count + self.vocab_size)
        )

        return math.log(probability)


    def next_token_log_probs(self, context):


        log_probs = np.array([
            self.log_prob(token, context) # calculate the log probability for every token given a context
            for token in range(self.vocab_size) # go trough every token in the vocabulary
        ])

        return log_probs
