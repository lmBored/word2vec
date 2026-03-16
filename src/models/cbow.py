import numpy as np

from src.Distribution import NegativeSampleCache
from src.models.sgns import _update_embeddings


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class CBOW:
    def __init__(
        self, window_size=2, num_neg_samples=5, learning_rate=0.025, embedding_dim=100, seed=None, batch_size=512
    ):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_context_size = 2 * window_size

        if seed is not None:
            np.random.seed(seed)

        # Will be set by build_vocab
        self.vocab_size = 0
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = np.array([])
        self.noise_dist = np.array([])
        self.W_input = np.array([])
        self.W_output = np.array([])
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
        self.W_input = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_output = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))

    def generate_training_data(self, corpus):
        data = []
        max_ctx = self.max_context_size

        for sentence in corpus:
            words = sentence.lower().split()
            indices = [self.word2idx[w] for w in words if w in self.word2idx]

            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                context_indices = []
                for j in range(start, end):
                    if i != j:
                        context_indices.append(indices[j])

                # Only add if we have at least one context word
                if context_indices:
                    num_ctx = len(context_indices)
                    padded_ctx = np.zeros(max_ctx, dtype=np.int64)
                    mask = np.zeros(max_ctx, dtype=np.float64)
                    padded_ctx[:num_ctx] = context_indices
                    mask[:num_ctx] = 1.0
                    data.append((padded_ctx, mask, center_idx))

        return data

    def sample_negatives(self, center_idx, k):
        negatives = []
        while len(negatives) < k:
            samples = np.random.choice(self.vocab_size, size=k - len(negatives), p=self.noise_dist)
            valid = samples[samples != center_idx]
            negatives.extend(valid.tolist())
        return np.array(negatives[:k], dtype=np.int64)

    def sample_negatives_batch(self, center_indices, k):
        batch_size = len(center_indices)
        return self.neg_cache.sample_batch(batch_size, k)

    def forward_backward(self, context_indices, center_idx, neg_indices):
        context_indices_arr = np.array(context_indices)
        num_context = len(context_indices)

        v_context = self.W_input[context_indices_arr]  # (|C|, D)
        v_center = self.W_output[center_idx]  # (D,)
        v_neg = self.W_output[neg_indices]  # (K, D)

        # Forward
        h = np.mean(v_context, axis=0)  # (D,)

        # Positive sample
        score_pos = np.dot(v_center, h)
        prob_pos = sigmoid(score_pos)

        # Negative samples
        scores_neg = np.dot(v_neg, h)  # (K,)
        probs_neg = sigmoid(scores_neg)

        # CBOW Loss
        eps = 1e-10
        loss = -np.log(prob_pos + eps) - np.sum(np.log(1 - probs_neg + eps))

        # Backward
        error_pos = prob_pos - 1.0  # scalar
        errors_neg = probs_neg  # (K,)

        # Gradient for hidden layer h
        grad_h = error_pos * v_center + np.dot(errors_neg, v_neg)  # (D,)

        # Gradient
        grad_center_pos = error_pos * h  # (D,)
        grad_center_neg = np.outer(errors_neg, h)  # (K, D)
        grad_context = np.tile(grad_h / num_context, (num_context, 1))  # (|C|, D)
        return loss, grad_context, grad_center_pos, grad_center_neg, grad_h

    def forward_backward_batch(self, context_batch, mask_batch, center_indices, neg_indices):
        v_context = self.W_input[context_batch]  # (B, C, D)
        v_center = self.W_output[center_indices]  # (B, D)
        v_neg = self.W_output[neg_indices]  # (B, K, D)

        # Forward
        # h = masked mean of context embeddings
        # (B, C) -> (B, C, 1)
        mask_expanded = mask_batch[:, :, np.newaxis]  # (B, C, 1)
        masked_context = v_context * mask_expanded  # (B, C, D)
        context_sum = np.sum(masked_context, axis=1)  # (B, D)
        num_valid = np.sum(mask_batch, axis=1, keepdims=True)  # (B, 1)
        num_valid = np.maximum(num_valid, 1.0)
        h = context_sum / num_valid  # (B, D)

        scores_pos = np.sum(h * v_center, axis=1)  # (B,)
        probs_pos = sigmoid(scores_pos)  # (B,)

        # (B, K, D) @ (B, D) -> (B, K)
        scores_neg = np.einsum("bkd,bd->bk", v_neg, h)  # (B, K)
        probs_neg = sigmoid(scores_neg)  # (B, K)

        # Loss
        loss_pos = -np.log(probs_pos + 1e-10)  # (B,)
        loss_neg = -np.sum(np.log(1 - probs_neg + 1e-10), axis=1)  # (B,)
        total_loss = np.sum(loss_pos + loss_neg)

        # Backward
        errors_pos = probs_pos - 1.0  # (B,)
        errors_neg = probs_neg  # (B, K)

        # Gradients
        grad_h = errors_pos[:, np.newaxis] * v_center + np.einsum("bk,bkd->bd", errors_neg, v_neg)  # (B, D)
        grad_center_pos = errors_pos[:, np.newaxis] * h  # (B, D)
        grad_center_neg = errors_neg[:, :, np.newaxis] * h[:, np.newaxis, :]  # (B, K, D)
        grad_context = (grad_h / num_valid)[:, np.newaxis, :] * mask_expanded  # (B, C, D)

        return total_loss, grad_context, grad_center_pos, grad_center_neg

    def step(self, context_indices, center_idx):
        neg_indices = self.sample_negatives(center_idx, self.num_neg_samples)
        loss, grad_context, grad_pos, grad_neg, _ = self.forward_backward(context_indices, center_idx, neg_indices)

        # updates
        context_indices_arr = np.array(context_indices)
        self.W_input[context_indices_arr] -= self.learning_rate * grad_context
        self.W_output[center_idx] -= self.learning_rate * grad_pos
        self.W_output[neg_indices] -= self.learning_rate * grad_neg

        return loss

    def step_batch(self, context_batch, mask_batch, center_indices):
        K = self.num_neg_samples

        neg_indices = self.sample_negatives_batch(center_indices, K)  # (B, K)
        loss, grad_context, grad_pos, grad_neg = self.forward_backward_batch(
            context_batch, mask_batch, center_indices, neg_indices
        )

        # Update
        lr = self.learning_rate

        # Context
        # (B, C) indices, (B, C, D) gradients
        flat_context = context_batch.ravel()
        flat_grad_context = grad_context.reshape(-1, self.embedding_dim)
        _update_embeddings(self.W_input, flat_context, flat_grad_context, lr)

        # Center word
        _update_embeddings(self.W_output, center_indices, grad_pos, lr)

        # Negative sample
        flat_neg = neg_indices.ravel()
        flat_grad_neg = grad_neg.reshape(-1, self.embedding_dim)
        _update_embeddings(self.W_output, flat_neg, flat_grad_neg, lr)

        return loss

    def train(self, corpus, epochs=5, min_count=1, verbose=True):
        # Build vocab
        self.build_vocab(corpus, min_count)

        # Training data
        data = self.generate_training_data(corpus)
        num_samples = len(data)

        context_all = np.array([d[0] for d in data], dtype=np.int64)  # (N, C)
        mask_all = np.array([d[1] for d in data], dtype=np.float64)  # (N, C)
        center_all = np.array([d[2] for d in data], dtype=np.int64)  # (N,)

        if verbose:
            print(f"Vocab size: {self.vocab_size}")
            print(f"Training samples: {num_samples:,}")
            print(f"Batch size: {self.batch_size}")

        losses = []
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            context_shuffled = context_all[indices]
            mask_shuffled = mask_all[indices]
            center_shuffled = center_all[indices]

            epoch_loss = 0.0
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size

            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, num_samples)

                context_batch = context_shuffled[start:end]
                mask_batch = mask_shuffled[start:end]
                center_batch = center_shuffled[start:end]

                epoch_loss += self.step_batch(context_batch, mask_batch, center_batch)

            avg_loss = epoch_loss / num_samples
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def get_embedding(self, word, combine=True):
        idx = self.word2idx[word.lower()]
        if combine:
            return (self.W_input[idx] + self.W_output[idx]) / 2.0
        return self.W_input[idx].copy()

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
