"""Note used atm. Created to be used for CL or TL."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.datasets import load_text8

DATA = Path(__file__).parent.parent.parent / "data"
UNK = "<UNK>"


class Vocabulary:
    """
    Build vocab with frequency-based filtering
    Vocab limits number of unique words, only focus on frequent words
    Rare words are replaced with a special <UNK> token
    """

    def __init__(self, word2idx, idx2word, word_counts):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_counts = word_counts

        total = sum(word_counts.values())
        self.word_freqs = {w: c / total for w, c in word_counts.items()}

    def __len__(self):
        return len(self.word2idx)

    def __contains__(self, word):
        return word in self.word2idx

    @classmethod
    def build_from_corpus(cls, tokens, max_vocab_size=50000, min_count=5):
        counter = Counter(tokens)
        filtered = {w: c for w, c in counter.items() if c >= min_count}
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        vocab_words = sorted_words[:max_vocab_size]

        word2idx = {UNK: 0}
        idx2word = {0: UNK}
        word_counts = {UNK: 0}

        for idx, (word, count) in enumerate(vocab_words, start=1):
            word2idx[word] = idx
            idx2word[idx] = word
            word_counts[word] = count

        unk_count = sum(c for w, c in counter.items() if w not in word2idx)
        word_counts[UNK] = unk_count
        return cls(word2idx, idx2word, word_counts)

    def get_index(self, word):
        return self.word2idx.get(word, 0)

    def get_word(self, idx):
        return self.idx2word.get(idx, UNK)

    def encode(self, tokens):
        return np.array([self.get_index(w) for w in tokens], dtype=np.int32)

    def save(self, path):
        data = {
            "word2idx": self.word2idx,
            "word_counts": self.word_counts,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path):
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        word2idx = data["word2idx"]
        idx2word = {int(v): k for k, v in word2idx.items()}
        word_counts = data["word_counts"]
        return cls(word2idx, idx2word, word_counts)


def prepare_text8_dataset(data_dir=None, max_vocab_size=50000, min_count=5, *, save_vocab=True):
    if data_dir is None:
        data_dir = DATA

    # Load tokens
    tokens = load_text8(data_dir)

    # Build vocab
    vocab = Vocabulary.build_from_corpus(tokens, max_vocab_size, min_count)

    # Save vocab
    if save_vocab:
        vocab_path = data_dir / "vocab.json"
        vocab.save(vocab_path)

    # Encode
    corpus = vocab.encode(tokens)
    return corpus, vocab


if __name__ == "__main__":
    corpus, vocab = prepare_text8_dataset()
    print(len(corpus))
    print(len(vocab))
