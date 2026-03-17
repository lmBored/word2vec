import argparse
import time

import numpy as np

from src.data.datasets import load_text8
from src.evaluation.analogy import eval_analogy
from src.evaluation.similarity import evaluate_similarity
from src.models.cbow import CBOW
from src.models.sgns import SGNS


def train_model(model_type, corpus, epochs, embedding_dim, window_size, num_neg_samples, learning_rate, min_count):
    if model_type == "sgns":
        model = SGNS(
            window_size=window_size,
            num_neg_samples=num_neg_samples,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            seed=42,
        )
    else:
        model = CBOW(
            window_size=window_size,
            num_neg_samples=num_neg_samples,
            learning_rate=learning_rate,
            embedding_dim=embedding_dim,
            seed=42,
        )

    losses = model.train(corpus, epochs=epochs, min_count=min_count, verbose=True)
    return model, losses


def get_embeddings(model, model_type):
    if model_type == "sgns":
        return (model.W_center + model.W_context) / 2
    return (model.W_input + model.W_output) / 2


def run_evaluation(embeddings, word2idx):
    results = {}
    for dataset in ["wordsim353", "simlex999"]:
        results[dataset] = evaluate_similarity((embeddings, word2idx), word2idx, dataset=dataset)
    results["analogy"] = eval_analogy(embeddings, word2idx)
    return results


def print_results(results, train_time, vocab_size, losses):
    ws353 = results["wordsim353"]
    sl999 = results["simlex999"]
    analogy = results["analogy"]

    print("\nResults:")
    print(f"Train time: {train_time:.1f}s")
    print(f"Vocab size: {vocab_size:,}")
    print(f"Loss: {losses}")
    print(f"WordSim-353: rho={ws353['spearman']:.3f} ({ws353['num_pairs_evaluated']}/{ws353['num_pairs_total']})")
    print(f"SimLex-999: rho={sl999['spearman']:.3f} ({sl999['num_pairs_evaluated']}/{sl999['num_pairs_total']})")
    print(
        f"Analogy: {analogy['total_accuracy'] * 100:.1f}% ({analogy['correct_answers']}/{analogy['answered_questions']})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sgns", choices=["sgns", "cbow"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--neg", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1000000)
    args = parser.parse_args()

    tokens = load_text8()[: args.tokens]
    corpus = [" ".join(tokens[i : i + 100]) for i in range(0, len(tokens), 100)]

    start = time.time()
    model, losses = train_model(
        args.model, corpus, args.epochs, args.dim, args.window, args.neg, args.lr, args.min_count
    )
    train_time = time.time() - start

    embeddings = get_embeddings(model, args.model)
    results = run_evaluation(embeddings, model.word2idx)
    print_results(results, train_time, model.vocab_size, losses)


if __name__ == "__main__":
    main()
