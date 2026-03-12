import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SGNS:
    def __init__(self, w=2, num_neg_samples=3, learning_rate=0.05):
        self.w = w
        self.num_neg_samples = num_neg_samples
        self.lr = learning_rate

    def build_vocab(self, corpus):
        dct = {}
        for sentence in corpus:
            for word in sentence.lower().split():
                if word in dct:
                    dct[word] += 1
                else:
                    dct[word] = 1

        vocab = list(dct.keys())
        self.vocab_size = len(vocab)

        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for i, w in enumerate(vocab)}
        print(self.word2idx)

        # higher range explodes dot product, 0.1 seems like a good spot
        self.w_center = np.random.uniform(-0.1, 0.1, (self.vocab_size, 10))
        self.w_context = np.random.uniform(-0.1, 0.1, (self.vocab_size, 10))

    def get_data(self, corpus):
        data = []
        for sentence in corpus:
            indices = [self.word2idx[w] for w in sentence.lower().split()]
            print(indices)
            for i, center_idx in enumerate(indices):
                start = max(0, i - self.w)
                end = min(len(indices), i + self.w + 1)
                for j in range(start, end):
                    if i != j:
                        data.append((center_idx, indices[j]))
        return data

    def get_negative_samples(self, context_idx):
        neg_samples = []
        while len(neg_samples) < self.num_neg_samples:
            rand = np.random.randint(0, self.vocab_size)
            if rand != context_idx:
                neg_samples.append(rand)
        return neg_samples

    def train(self, corpus, epochs=50):
        self.build_vocab(corpus)
        data = self.get_data(corpus)
        print(data)

        for epoch in range(epochs):
            loss = 0
            for center_idx, context_idx in data:
                # Get negative samplings
                neg_idx = self.get_negative_samples(context_idx)

                # Forward
                v_center = self.w_center[center_idx]  # (10, ...)
                v = self.w_context[context_idx]  # (10, ...)
                v_neg = self.w_context[neg_idx]

                score = np.dot(v_center, v)
                score_neg = np.dot(v_neg, v_center)
                prob = sigmoid(score)
                prob_neg = sigmoid(score_neg)

                # Negative sampling loss
                # 1e-7 to prevent log(0)
                loss += -np.log(prob + 1e-7) - np.sum(np.log(1 - prob_neg + 1e-7))

                # Gradient descent
                error = prob - 1
                error_neg = prob_neg - 0
                grad = error * v_center
                grad_neg = np.outer(error_neg, v_center)
                grad_center = error * v + np.dot(error_neg, v_neg)

                # Updates
                self.w_center[center_idx] -= self.lr * grad_center
                self.w_context[context_idx] -= self.lr * grad
                self.w_context[neg_idx] -= self.lr * grad_neg

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, loss = {loss}")

    def get_embedding(self, word):
        idx = self.word2idx[word]
        return (self.w_center[idx] + self.w_context[idx]) / 2.0


s = [
    "The task is to implement the optimization procedure",
    "machine learning is a mathematical optimization procedure",
    "deep learning allows us to implement machine learning",
    "the optimization procedure is a learning task",
]

m = SGNS()
m.train(s)


def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


vec1 = m.get_embedding("optimization")
vec2 = m.get_embedding("procedure")
vec3 = m.get_embedding("deep")
print(cosine(vec1, vec2))
print(cosine(vec1, vec3))
