import pathlib
import urllib.request

import numpy as np

semantic = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}

syntactic = {
    "gram1-adjective-to-adverb",
    "gram2-opposite",
    "gram3-comparative",
    "gram4-superlative",
    "gram5-present-participle",
    "gram6-nationality-adjective",
    "gram7-past-tense",
    "gram8-plural",
    "gram9-plural-verbs",
}


def download_analogy_dataset(
    data_dir="data/evaluation",
    url="https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt",
    force=False,
):
    data_path = pathlib.Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    filepath = data_path / "questions-words.txt"

    if filepath.exists() and not force:
        return str(filepath)

    urllib.request.urlretrieve(url, filepath)
    return str(filepath)


def load_analogy_dataset(filepath):
    categories = {}
    current_category = None

    with pathlib.Path(filepath).open(encoding="utf-8") as source:
        for raw_line in source:
            text = raw_line.strip()
            if not text:
                continue
            if text.startswith(":"):
                current_category = text[1:].strip().lower()
                categories[current_category] = []
                continue

            parts = text.lower().split()
            if current_category is not None and len(parts) == 4:
                categories[current_category].append(tuple(parts))

    return categories


def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-10)


def solve_analogy_3cosadd(a_vec, a_star_vec, b_star_vec, normalized_embeddings, exclude_indices):
    query = b_star_vec - a_star_vec + a_vec
    similarities = np.dot(normalized_embeddings, query)
    similarities[np.asarray(exclude_indices)] = -np.inf
    return int(np.argmax(similarities))


def _empty_results():
    return {
        "total_questions": 0,
        "answered_questions": 0,
        "correct_answers": 0,
        "semantic_correct": 0,
        "semantic_total": 0,
        "syntactic_correct": 0,
        "syntactic_total": 0,
    }


def _eval_category(analogies, word2idx, normalized, verbose):
    idx2word = {idx: word for word, idx in word2idx.items()}
    results = _empty_results()
    is_semantic = False

    for a, a_star, b_star, b_expected in analogies:
        results["total_questions"] += 1

        if not all(word in word2idx for word in (a, a_star, b_star, b_expected)):
            continue

        a_idx = word2idx[a]
        a_star_idx = word2idx[a_star]
        b_star_idx = word2idx[b_star]
        b_expected_idx = word2idx[b_expected]

        predicted_idx = solve_analogy_3cosadd(
            normalized[a_idx],
            normalized[a_star_idx],
            normalized[b_star_idx],
            normalized,
            (a_idx, a_star_idx, b_star_idx),
        )

        results["answered_questions"] += 1

        if is_semantic:
            results["semantic_total"] += 1
        else:
            results["syntactic_total"] += 1

        if predicted_idx == b_expected_idx:
            results["correct_answers"] += 1
            if is_semantic:
                results["semantic_correct"] += 1
            else:
                results["syntactic_correct"] += 1
        elif verbose:
            got = idx2word.get(predicted_idx, "<UNK>")
            print(f"{a}:{a_star} :: {b_star}:? expected {b_expected} got {got}")

    return results


def _finalize_results(results):
    if results["answered_questions"] > 0:
        results["total_accuracy"] = results["correct_answers"] / results["answered_questions"]
    else:
        results["total_accuracy"] = 0.0

    if results["semantic_total"] > 0:
        results["semantic_accuracy"] = results["semantic_correct"] / results["semantic_total"]
    else:
        results["semantic_accuracy"] = 0.0

    if results["syntactic_total"] > 0:
        results["syntactic_accuracy"] = results["syntactic_correct"] / results["syntactic_total"]
    else:
        results["syntactic_accuracy"] = 0.0

    if results["total_questions"] > 0:
        results["coverage"] = results["answered_questions"] / results["total_questions"]
    else:
        results["coverage"] = 0.0

    return results


def eval_analogy(embeddings, word2idx, data_dir="data/evaluation", verbose=False):
    filepath = download_analogy_dataset(data_dir)
    categories = load_analogy_dataset(filepath)
    normalized = normalize_embeddings(embeddings)

    results = {
        "category_scores": {},
        "total_questions": 0,
        "answered_questions": 0,
        "correct_answers": 0,
        "semantic_correct": 0,
        "semantic_total": 0,
        "syntactic_correct": 0,
        "syntactic_total": 0,
    }

    for category, analogies in categories.items():
        category_result = _eval_category(analogies, word2idx, normalized, verbose)
        results["total_questions"] += category_result["total_questions"]
        results["answered_questions"] += category_result["answered_questions"]
        results["correct_answers"] += category_result["correct_answers"]
        results["semantic_correct"] += category_result["semantic_correct"]
        results["semantic_total"] += category_result["semantic_total"]
        results["syntactic_correct"] += category_result["syntactic_correct"]
        results["syntactic_total"] += category_result["syntactic_total"]

        if category_result["answered_questions"] > 0:
            results["category_scores"][category] = {
                "accuracy": category_result["correct_answers"] / category_result["answered_questions"],
                "correct": category_result["correct_answers"],
                "total": category_result["answered_questions"],
            }

    return _finalize_results(results)


def print_analogy_results(results):
    print(f"Total questions: {results.get('total_questions', 0)}")
    print(f"Answered questions: {results.get('answered_questions', 0)}")
    print(f"Coverage: {100 * results.get('coverage', 0.0):.1f}%")
    print(f"Correct answers: {results.get('correct_answers', 0)}")
    print(f"Accuracy: {100 * results.get('total_accuracy', 0.0):.2f}%")
    print(f"Semantic accuracy:  {100 * results.get('semantic_accuracy', 0.0):.2f}%")
    print(f"Syntactic accuracy: {100 * results.get('syntactic_accuracy', 0.0):.2f}%")

    for category, scores in sorted(
        results.get("category_scores", {}).items(), key=lambda item: item[1]["accuracy"], reverse=True
    ):
        cat_type = "sem" if category in semantic else "syn"
        print(f"[{cat_type}] {category}: {100 * scores['accuracy']:.1f}% ({scores['correct']}/{scores['total']})")
