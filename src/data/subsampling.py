import numpy as np


def compute_discard_probs(word_freqs, threshold=1e-5):
    total_words = sum(word_freqs.values())
    if total_words == 0:
        return dict.fromkeys(word_freqs, 0.0)

    discard_probs = {}
    for word, count in word_freqs.items():
        # Freq f(w) = count(w) / total_words
        f_w = count / total_words

        if f_w <= threshold:
            # Not discard infrequent words
            discard_probs[word] = 0.0
        else:
            # P(discard) = 1 - sqrt(t / f(w))
            discard_probs[word] = 1.0 - np.sqrt(threshold / f_w)

    return discard_probs


def subsample_corpus(corpus_indices, discard_probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if len(corpus_indices) == 0:
        return np.array([], dtype=np.int64)

    # Vectorized ops
    corpus_arr = np.asarray(corpus_indices, dtype=np.int64)

    # Make keep probability array
    keep_probs = np.array([1.0 - discard_probs.get(idx, 0.0) for idx in corpus_arr])

    random_vals = rng.random(len(corpus_arr))
    mask = random_vals < keep_probs
    return corpus_arr[mask]


def subsample_sentences(sentences_indices, discard_probs, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    subsampled_sentences = []
    for sentence in sentences_indices:
        if len(sentence) == 0:
            continue

        subsampled = subsample_corpus(sentence, discard_probs, rng)
        if len(subsampled) > 0:
            subsampled_sentences.append(subsampled.tolist())

    return subsampled_sentences
