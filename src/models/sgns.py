import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class SGNS:
    def __init__(self, window_size=2, num_neg_samples=5, learning_rate=0.025, embedding_dim=100, seed=None):
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim

        if seed is not None:
            np.random.seed(seed)

        # Will be set by build_vocab
        self.vocab_size = 0
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self.word_counts = np.array([])
        self.noise_dist = np.array([])
        self.W_center = np.array([])
        self.W_context = np.array([])

    def build_vocab(self, corpus, min_count=1):
        # Count word freq
        word_freq: dict[str, int] = {}
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

    def step(self, center_idx, context_idx) -> float:
        neg_indices = self.sample_negatives(context_idx, self.num_neg_samples)
        loss, grad_center, grad_pos, grad_neg = self.forward_backward(center_idx, context_idx, neg_indices)

        # updates
        self.W_center[center_idx] -= self.learning_rate * grad_center
        self.W_context[context_idx] -= self.learning_rate * grad_pos
        self.W_context[neg_indices] -= self.learning_rate * grad_neg

        return loss

    def train(self, corpus, epochs=5, min_count=1, verbose=True):
        self.build_vocab(corpus, min_count)

        pairs = self.generate_training_pairs(corpus)

        if verbose:
            print(f"Vocab size: {self.vocab_size}")
            print(f"len pairs: {len(pairs)}")

        losses = []
        for epoch in range(epochs):
            np.random.shuffle(pairs)

            epoch_loss = 0.0
            for center_idx, context_idx in pairs:
                epoch_loss += self.step(center_idx, context_idx)

            avg_loss = epoch_loss / len(pairs)
            losses.append(avg_loss)

            if verbose and epoch % 10 == 0:
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
