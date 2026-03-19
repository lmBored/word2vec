import numpy as np
from numba import njit, prange

from src.Distribution import NegativeSampleCache


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


@njit(parallel=True, cache=True)
def _update_embeddings(W, indices, grads, lr):
    n_updates = len(indices)
    n_dims = W.shape[1]
    for d in prange(n_dims):
        for i in range(n_updates):
            idx = indices[i]
            W[idx, d] -= lr * grads[i, d]


@njit(parallel=True, cache=True)
def _forward_backward_batch(W_center, W_context, center_idx, context_idx, neg_idx):
    B = len(center_idx)
    K = neg_idx.shape[1]
    D = W_center.shape[1]

    loss = 0.0

    # Grad arrays
    grad_center = np.zeros((B, D))
    grad_pos = np.zeros((B, D))
    grad_neg = np.zeros((B, K, D))

    for b in prange(B):
        c_idx = center_idx[b]
        ctx_idx = context_idx[b]

        v_w = W_center[c_idx]
        v_c = W_context[ctx_idx]

        # Pos score
        score_pos = 0.0
        for d in range(D):
            score_pos += v_c[d] * v_w[d]

        # Sigmoid
        if score_pos > 500:
            prob_pos = 1.0
        elif score_pos < -500:
            prob_pos = 0.0
        else:
            prob_pos = 1.0 / (1.0 + np.exp(-score_pos))

        # Loss
        loss += -np.log(prob_pos + 1e-10)

        # Gradients
        error_pos = prob_pos - 1.0
        for d in range(D):
            grad_center[b, d] = error_pos * v_c[d]
            grad_pos[b, d] = error_pos * v_w[d]

        # Negative samples
        for k in range(K):
            n_idx = neg_idx[b, k]
            v_neg = W_context[n_idx]

            score_neg = 0.0
            for d in range(D):
                score_neg += v_neg[d] * v_w[d]

            # Sigmoid
            if score_neg > 500:
                prob_neg = 1.0
            elif score_neg < -500:
                prob_neg = 0.0
            else:
                prob_neg = 1.0 / (1.0 + np.exp(-score_neg))

            # Loss
            loss += -np.log(1.0 - prob_neg + 1e-10)

            # Gradients
            for d in range(D):
                grad_center[b, d] += prob_neg * v_neg[d]
                grad_neg[b, k, d] = prob_neg * v_w[d]

    return loss, grad_center, grad_pos, grad_neg


class SGNS:
    def __init__(
        self,
        window_size=2,
        num_neg_samples=5,
        learning_rate=0.025,
        embedding_dim=100,
        seed=None,
        batch_size=512,
        subsample_threshold=1e-5,
    ):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.subsample_threshold = subsample_threshold

        if seed is not None:
            np.random.seed(seed)

        # Will be set by build_vocab
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = np.array([])
        self.word_freqs = np.array([])
        self.discard_probs = None
        self.noise_dist = np.array([])
        self.W_center = np.array([])
        self.W_context = np.array([])
        self.neg_cache = None

    def build_vocab(self, corpus, min_count=1):
        # Count word freq
        word_freq = {}
        for sentence in corpus:
            for word in sentence.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        vocab = [w for w, c in word_freq.items() if c >= min_count]
        self.vocab_size = len(vocab)

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}

        # Noise dist
        self.word_counts = np.array([word_freq[w] for w in vocab], dtype=np.float64)
        # P(w)^0.75 / sum P(w)^0.75
        # 0.75 exponent from the paper, to smooth the distribution
        smoothed = np.power(self.word_counts, 0.75)
        self.noise_dist = smoothed / smoothed.sum()

        self.neg_cache = NegativeSampleCache(self.noise_dist)

        # From OG paper, we should do subsampling
        # Word frequencies and discard probabilities for subsampling
        # P(discard | w) = 1 - sqrt(t / f(w))
        total_count = self.word_counts.sum()
        self.word_freqs = self.word_counts / total_count
        if self.subsample_threshold is not None:
            self.discard_probs = np.maximum(0, 1 - np.sqrt(self.subsample_threshold / self.word_freqs))
        else:
            self.discard_probs = None

        scale = 0.5 / self.embedding_dim
        self.W_center = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_context = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))

    def generate_training_pairs(self, corpus):
        pairs = []
        for sentence in corpus:
            words = sentence.lower().split()
            indices = [self.word2idx[w] for w in words if w in self.word2idx]

            # Apply subsampling
            if self.discard_probs is not None and len(indices) > 0:
                keep_mask = np.random.random(len(indices)) > self.discard_probs[indices]
                indices = [idx for idx, keep in zip(indices, keep_mask) if keep]

            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((center_idx, indices[j]))

        return pairs

    def sample_negatives_batch(self, context_indices, k):
        batch_size = len(context_indices)
        return self.neg_cache.sample_batch(batch_size, k)

    def forward_backward(self, center_idx, context_idx, neg_indices):
        v_w = self.W_center[center_idx]  # (D,)
        v_c = self.W_context[context_idx]  # (D,)
        v_neg = self.W_context[neg_indices]  # (K, D)

        # Forward
        # Pos sample
        score_pos = np.dot(v_c, v_w)  # scalar
        prob_pos = sigmoid(score_pos)  # sigma(v'_c.v_w)

        # Neg samples
        scores_neg = np.dot(v_neg, v_w)  # (K,)
        probs_neg = sigmoid(scores_neg)

        # Loss
        loss = -np.log(prob_pos + 1e-10) - np.sum(np.log(1 - probs_neg + 1e-10))

        # Backward
        error_pos = prob_pos - 1.0
        errors_neg = probs_neg

        grad_center = error_pos * v_c + np.dot(errors_neg, v_neg)  # (D,)
        grad_context_pos = error_pos * v_w  # (D,)
        grad_context_neg = np.outer(errors_neg, v_w)  # (K, D)

        return loss, grad_center, grad_context_pos, grad_context_neg

    def step_batch(self, center_indices, context_indices, lr):
        K = self.num_neg_samples

        neg_indices = self.sample_negatives_batch(context_indices, K)  # (B, K)
        loss, grad_center, grad_pos, grad_neg = _forward_backward_batch(
            self.W_center, self.W_context, center_indices, context_indices, neg_indices
        )

        # Update with current learning rate
        _update_embeddings(self.W_center, center_indices, grad_center, lr)
        _update_embeddings(self.W_context, context_indices, grad_pos, lr)

        flat_neg = neg_indices.ravel()
        flat_grad = grad_neg.reshape(-1, self.embedding_dim)
        _update_embeddings(self.W_context, flat_neg, flat_grad, lr)

        return loss

    def train(self, corpus, epochs=5, min_count=1, verbose=True, min_lr=0.0001):
        self.build_vocab(corpus, min_count)

        pairs = self.generate_training_pairs(corpus)
        pairs = np.array(pairs, dtype=np.int64)
        num_pairs = len(pairs)
        # Add (self.batch_size - 1) so any partially filled batch also gets counted
        total_batches = epochs * ((num_pairs + self.batch_size - 1) // self.batch_size)

        if verbose:
            print(f"Vocab size: {self.vocab_size}")
            print(f"Training pairs: {num_pairs:,}")
            print(f"Batch size: {self.batch_size}")

        losses = []
        batch_count = 0
        for epoch in range(epochs):
            # Shuffle pairs
            indices = np.random.permutation(num_pairs)
            pairs_shuffled = pairs[indices]

            epoch_loss = 0.0
            num_batches = (num_pairs + self.batch_size - 1) // self.batch_size

            for batch_idx in range(num_batches):
                # Linear decay lr
                progress = batch_count / total_batches
                lr = self.learning_rate * (1 - progress) + min_lr * progress
                batch_count += 1

                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, num_pairs)

                batch_pairs = pairs_shuffled[start:end]
                center_indices = batch_pairs[:, 0]
                context_indices = batch_pairs[:, 1]

                epoch_loss += self.step_batch(center_indices, context_indices, lr)

            avg_loss = epoch_loss / num_pairs
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
                self.save(f"checkpoints/sgns_epoch_{epoch + 1}.npz")

        return losses

    def get_embedding(self, word, combine=True):
        idx = self.word2idx[word.lower()]
        if combine:
            return (self.W_center[idx] + self.W_context[idx]) / 2.0
        return self.W_center[idx].copy()

    def most_similar(self, word, top_k=10):
        query_vec = self.get_embedding(word)
        query_norm = np.linalg.norm(query_vec)

        similarities = []
        for w, idx in self.word2idx.items():
            if w == word.lower():
                continue
            vec = self.get_embedding(w)
            sim = np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec) + 1e-10)
            similarities.append((w, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save(self, path):
        np.savez(path, W_center=self.W_center, W_context=self.W_context, vocab=list(self.word2idx.keys()))
