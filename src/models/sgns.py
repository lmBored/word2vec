import numpy as np

from src.Distribution import NegativeSampleCache


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class SGNS:
    def __init__(
        self, window_size=2, num_neg_samples=5, learning_rate=0.025, embedding_dim=100, seed=None, batch_size=512
    ):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        if seed is not None:
            np.random.seed(seed)

        # Will be set by build_vocab
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = np.array([])
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

        scale = 0.5 / self.embedding_dim
        self.W_center = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_context = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))

    def generate_training_pairs(self, corpus):
        pairs = []
        for sentence in corpus:
            words = sentence.lower().split()
            indices = [self.word2idx[w] for w in words if w in self.word2idx]

            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        pairs.append((center_idx, indices[j]))

        return pairs

    def sample_negatives(self, context_idx, k):
        negatives = []
        while len(negatives) < k:
            # Sample from noise dist
            samples = np.random.choice(self.vocab_size, size=k - len(negatives), p=self.noise_dist)
            # Filter true context word
            valid = samples[samples != context_idx]
            negatives.extend(valid.tolist())

        return np.array(negatives[:k], dtype=np.int64)

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

    def forward_backward_batch(self, center_indices, context_indices, neg_indices):
        v_w = self.W_center[center_indices]  # (B, D)
        v_c = self.W_context[context_indices]  # (B, D)
        v_neg = self.W_context[neg_indices]  # (B, K, D)

        # Forward
        scores_pos = np.sum(v_c * v_w, axis=1)  # (B,)
        probs_pos = sigmoid(scores_pos)  # (B,)
        # (B, K, D) @ (B, D, 1) -> (B, K, 1) -> (B, K)
        scores_neg = np.einsum("bkd,bd->bk", v_neg, v_w)  # (B, K)
        probs_neg = sigmoid(scores_neg)  # (B, K)

        # Loss
        loss_pos = -np.log(probs_pos + 1e-10)  # (B,)
        loss_neg = -np.sum(np.log(1 - probs_neg + 1e-10), axis=1)  # (B,)
        total_loss = np.sum(loss_pos + loss_neg)

        # Backward
        errors_pos = probs_pos - 1.0  # (B,)
        errors_neg = probs_neg  # (B, K)

        # Gradients
        grad_center = errors_pos[:, np.newaxis] * v_c + np.einsum("bk,bkd->bd", errors_neg, v_neg)
        grad_context_pos = errors_pos[:, np.newaxis] * v_w
        grad_context_neg = errors_neg[:, :, np.newaxis] * v_w[:, np.newaxis, :]

        return total_loss, grad_center, grad_context_pos, grad_context_neg

    def step(self, center_idx, context_idx):
        neg_indices = self.sample_negatives(context_idx, self.num_neg_samples)
        loss, grad_center, grad_pos, grad_neg = self.forward_backward(center_idx, context_idx, neg_indices)

        # updates
        self.W_center[center_idx] -= self.learning_rate * grad_center
        self.W_context[context_idx] -= self.learning_rate * grad_pos
        self.W_context[neg_indices] -= self.learning_rate * grad_neg

        return loss

    def step_batch(self, center_indices, context_indices):
        K = self.num_neg_samples

        neg_indices = self.sample_negatives_batch(context_indices, K)  # (B, K)
        loss, grad_center, grad_pos, grad_neg = self.forward_backward_batch(
            center_indices, context_indices, neg_indices
        )

        # Update
        np.add.at(self.W_center, center_indices, -self.learning_rate * grad_center)  # np.add.at for duplicate indices
        np.add.at(self.W_context, context_indices, -self.learning_rate * grad_pos)
        # neg_indices is (B, K), grad_neg is (B, K, D)
        flat_neg_indices = neg_indices.ravel()  # (B*K,)
        flat_grad_neg = grad_neg.reshape(-1, self.embedding_dim)  # (B*K, D)
        np.add.at(self.W_context, flat_neg_indices, -self.learning_rate * flat_grad_neg)

        return loss

    def train(self, corpus, epochs=5, min_count=1, verbose=True):
        self.build_vocab(corpus, min_count)

        pairs = self.generate_training_pairs(corpus)
        pairs = np.array(pairs, dtype=np.int64)
        num_pairs = len(pairs)

        if verbose:
            print(f"Vocab size: {self.vocab_size}")
            print(f"Training pairs: {num_pairs:,}")
            print(f"Batch size: {self.batch_size}")

        losses = []
        for epoch in range(epochs):
            # Shuffle pairs
            indices = np.random.permutation(num_pairs)
            pairs_shuffled = pairs[indices]

            epoch_loss = 0.0
            num_batches = (num_pairs + self.batch_size - 1) // self.batch_size

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, num_pairs)

                batch_pairs = pairs_shuffled[start:end]
                center_indices = batch_pairs[:, 0]
                context_indices = batch_pairs[:, 1]

                epoch_loss += self.step_batch(center_indices, context_indices)

            avg_loss = epoch_loss / num_pairs
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

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
