import time

import numpy as np

from src.data.datasets import load_text8
from src.models.cbow import CBOW

tokens = load_text8()

chunk_size = 100
tokens_subset = tokens[:1000000]
corpus = [" ".join(tokens_subset[i : i + chunk_size]) for i in range(0, len(tokens_subset), chunk_size)]
print(f"Training chunks: {len(corpus):,}")


model = CBOW(window_size=5, num_neg_samples=5, learning_rate=0.025, embedding_dim=100, seed=67)

start_time = time.time()
losses = model.train(corpus, epochs=3, min_count=5, verbose=True)
train_time = time.time() - start_time

if __name__ == "__main__":
    print(f"Time: {train_time:.1f} seconds")
    print(f"Loss: {losses[-1]:.4f}")
    print()

    embeddings = (model.W_input + model.W_output) / 2

    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)

    print("Word similarities:")
    pairs = [("king", "queen"), ("man", "woman"), ("one", "two"), ("good", "great")]
    for w1, w2 in pairs:
        if w1 in model.word2idx and w2 in model.word2idx:
            sim = cosine_sim(embeddings[model.word2idx[w1]], embeddings[model.word2idx[w2]])
            print(f"cos({w1}, {w2}) = {sim:.3f}")

    print()
    print("Most similar to 'king':")
    if "king" in model.word2idx:
        king_vec = embeddings[model.word2idx["king"]]
        sims = [(w, cosine_sim(king_vec, embeddings[i])) for w, i in model.word2idx.items() if w != "king"]
        sims.sort(key=lambda x: x[1], reverse=True)
        for w, s in sims[:5]:
            print(f"{w}: {s:.3f}")
