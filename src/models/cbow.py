import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class CBOW:
    def __init__(self, window_size=2, num_neg_samples=5, learning_rate=0.025, embedding_dim=100, seed=None):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

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

        scale = 0.5 / self.embedding_dim
        self.W_input = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_output = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))

    def generate_training_data(self, corpus):
        data = []
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
                    data.append((context_indices, center_idx))

        return data

    def sample_negatives(self, center_idx, k):
        negatives = []
        while len(negatives) < k:
            samples = np.random.choice(self.vocab_size, size=k - len(negatives), p=self.noise_dist)
            valid = samples[samples != center_idx]
            negatives.extend(valid.tolist())
        return np.array(negatives[:k], dtype=np.int64)

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

    def step(self, context_indices, center_idx):
        neg_indices = self.sample_negatives(center_idx, self.num_neg_samples)
        loss, grad_context, grad_pos, grad_neg, _ = self.forward_backward(context_indices, center_idx, neg_indices)

        # updates
        context_indices_arr = np.array(context_indices)
        self.W_input[context_indices_arr] -= self.learning_rate * grad_context
        self.W_output[center_idx] -= self.learning_rate * grad_pos
        self.W_output[neg_indices] -= self.learning_rate * grad_neg

        return loss

    def train(self, corpus, epochs=5, min_count=1, verbose=True):
        # Build vocab
        self.build_vocab(corpus, min_count)

        # Training data
        data = self.generate_training_data(corpus)

        if verbose:
            print(f"Vocab size: {self.vocab_size}")
            print(f"len data: {len(data)}")

        losses = []
        for epoch in range(epochs):
            np.random.shuffle(data)
            epoch_loss = 0.0
            for context_indices, center_idx in data:
                epoch_loss += self.step(context_indices, center_idx)
            avg_loss = epoch_loss / len(data)
            losses.append(avg_loss)

            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return losses

    def get_embedding(self, word, combine=True) -> np.ndarray:
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
