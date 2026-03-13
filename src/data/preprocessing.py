from __future__ import annotations

import numpy as np


def compute_subsampling_probabilities(vocab, threshold=1e-5):
    vocab_size = len(vocab)
    keep_probs = np.ones(vocab_size, dtype=np.float32)

    for word, freq in vocab.word_freqs.items():
        idx = vocab.get_index(word)
        if freq > threshold:
            # P(discard) = 1 - sqrt(t / f(w))
            # P(keep) = sqrt(t / f(w))
            keep_probs[idx] = np.sqrt(threshold / freq)
    return keep_probs


def apply_subsampling(corpus, keep_probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    token_keep_probs = keep_probs[corpus]
    random_vals = rng.random(len(corpus), dtype=np.float32)
    mask = random_vals < token_keep_probs
    return corpus[mask]


class SkipGramBatchGenerate:
    def __init__(self, corpus, window_size=5, batch_size=256, rng=None):
        self.corpus = corpus
        self.window_size = window_size
        self.batch_size = batch_size
        self.rng = rng if rng is not None else np.random.default_rng()
        # Skip UNK token
        self.valid_positions = np.where(corpus != 0)[0]
        self.n_positions = len(self.valid_positions)

    def __len__(self):
        avg_pairs_per_pos = self.window_size
        total_pairs = self.n_positions * avg_pairs_per_pos
        return total_pairs // self.batch_size

    def generate_batch(self):
        centers = []
        contexts = []

        while len(centers) < self.batch_size:
            # Sample random center position
            pos_idx = self.rng.integers(self.n_positions)
            center_pos = self.valid_positions[pos_idx]
            center_word = self.corpus[center_pos]

            # Sample window size uniformly from [1, window_size]
            actual_window = self.rng.integers(1, self.window_size + 1)

            # Context word
            start = max(0, center_pos - actual_window)
            end = min(len(self.corpus), center_pos + actual_window + 1)

            for ctx_pos in range(start, end):
                if ctx_pos == center_pos:
                    continue
                ctx_word = self.corpus[ctx_pos]
                if ctx_word == 0:  # Skip UNK
                    continue
                centers.append(center_word)
                contexts.append(ctx_word)
                if len(centers) >= self.batch_size:
                    break

        return (
            np.array(centers[: self.batch_size], dtype=np.int32),
            np.array(contexts[: self.batch_size], dtype=np.int32),
        )

    def iterate_epoch(self):
        n = len(self)
        for _ in range(n):
            yield self.generate_batch()


def extract_all_skipgram_pairs(corpus, window_size=5):
    centers = []
    contexts = []
    for i, center_word in enumerate(corpus):
        if center_word == 0:
            continue
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)
        for j in range(start, end):
            if j == i:
                continue
            ctx_word = corpus[j]
            if ctx_word == 0:
                continue
            centers.append(center_word)
            contexts.append(ctx_word)

    return np.array(centers, dtype=np.int32), np.array(contexts, dtype=np.int32)


class NegativeSampler:
    def __init__(self, vocab, power=0.75, rng=None):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Noise dist P(w) = freq(w)^power
        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for word, count in vocab.word_counts.items():
            idx = vocab.get_index(word)
            if idx != 0:
                counts[idx] = count

        # Power transofrm
        powered = counts**power
        self.noise_distribution = powered / powered.sum()

    def sample(self, batch_size, n_negatives):
        return self.rng.choice(self.vocab_size, size=(batch_size, n_negatives), p=self.noise_distribution).astype(
            np.int32
        )


if __name__ == "__main__":
    from src.data.datasets import prepare_text8_dataset

    corpus, vocab = prepare_text8_dataset(max_vocab_size=30000)

    keep_probs = compute_subsampling_probabilities(vocab)
    for word in ["the", "of", "and", "in", "to"]:
        if word in vocab:
            idx = vocab.get_index(word)
            freq = vocab.word_freqs[word]
            print(f"{word}: freq={freq:.6f}, P(keep)={keep_probs[idx]:.4f}")

    subsampled = apply_subsampling(corpus, keep_probs, rng=np.random.default_rng(42))
    print(f"old: {len(corpus):,} tokens")
    print(f"new: {len(subsampled):,} tokens, {100 * len(subsampled) / len(corpus):.1f}%")

    print()
    print("Skip gram")
    batch_gen = SkipGramBatchGenerate(subsampled, window_size=5, batch_size=128, rng=np.random.default_rng(42))

    centers, contexts = batch_gen.generate_batch()
    for i in range(5):
        print(f"({vocab.get_word(centers[i])}, {vocab.get_word(contexts[i])})")

    print()
    print("Negative sampler")
    neg_sampler = NegativeSampler(vocab, rng=np.random.default_rng(42))
    negatives = neg_sampler.sample(batch_size=128, n_negatives=5)
    print(f"{[vocab.get_word(n) for n in negatives[0]]}")
