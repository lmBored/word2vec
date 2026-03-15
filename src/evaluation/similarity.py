import io
import pathlib
import urllib.request
import zipfile

import numpy as np
import scipy.stats

wordsim353 = "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt"
simlex999 = "https://fh295.github.io/SimLex-999.zip"
data = pathlib.Path(__file__).resolve().parents[2] / "data" / "evaluation"


def _download_wordsim353(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / "wordsim353.txt"

    if output_file.exists():
        return output_file

    last_error = None
    for url in wordsim353:
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                output_file.write_bytes(response.read())
        except Exception as exc:
            last_error = exc
        else:
            return output_file

    msg = "Could not download WordSim-353"
    raise RuntimeError(msg) from last_error


def _download_simlex999(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    output_file = data_dir / "SimLex-999.txt"

    if output_file.exists():
        return output_file

    with urllib.request.urlopen(simlex999, timeout=30) as response:
        zip_data = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_data)) as archive:
        for name in archive.namelist():
            if name.endswith("SimLex-999.txt"):
                with archive.open(name) as source:
                    output_file.write_bytes(source.read())
                return output_file

    raise FileNotFoundError


def _load_wordsim353(data_dir=None):
    if data_dir is None:
        data_dir = data

    filepath = _download_wordsim353(data_dir)
    pairs = []

    with filepath.open(encoding="utf-8") as source:
        for raw_line in source:
            parts = raw_line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                score = float(parts[2])
            except ValueError:
                continue
            pairs.append((parts[0].lower(), parts[1].lower(), score))

    return pairs


def _load_simlex999(data_dir=None):
    if data_dir is None:
        data_dir = data

    filepath = _download_simlex999(data_dir)
    pairs = []

    with filepath.open(encoding="utf-8") as source:
        header = source.readline().strip().split("\t")
        try:
            w1_idx = header.index("word1")
            w2_idx = header.index("word2")
            score_idx = header.index("SimLex999")
        except ValueError:
            w1_idx, w2_idx, score_idx = 0, 1, 3

        for raw_line in source:
            parts = raw_line.strip().split("\t")
            if len(parts) <= max(w1_idx, w2_idx, score_idx):
                continue
            try:
                score = float(parts[score_idx])
            except ValueError:
                continue
            pairs.append((parts[w1_idx].lower(), parts[w2_idx].lower(), score))

    return pairs


def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def _get_word_vector(word, embeddings, word2idx):
    if isinstance(embeddings, dict):
        return embeddings.get(word)

    if isinstance(embeddings, tuple) and len(embeddings) == 2:
        matrix = embeddings[0]
        idx = word2idx.get(word)
        if idx is None or idx < 0 or idx >= matrix.shape[0]:
            return None
        return matrix[idx]
    raise TypeError


def evaluate_similarity(embeddings, word2idx, dataset="wordsim353", data_dir=None, verbose=True):
    if data_dir is None:
        data_dir = data

    dataset_name = dataset.lower()
    if dataset_name == "wordsim353":
        pairs = _load_wordsim353(data_dir)
        label = "WordSim-353"
    elif dataset_name in {"simlex999", "simlex-999"}:
        pairs = _load_simlex999(data_dir)
        label = "SimLex-999"
    else:
        raise ValueError("Unknown dataset")

    human_scores = []
    model_scores = []
    oov_pairs = []

    for word1, word2, human_score in pairs:
        vec1 = _get_word_vector(word1, embeddings, word2idx)
        vec2 = _get_word_vector(word2, embeddings, word2idx)

        if vec1 is None or vec2 is None:
            oov_pairs.append((word1, word2))
            continue

        human_scores.append(human_score)
        model_scores.append(cosine_similarity(vec1, vec2))

    num_evaluated = len(human_scores)
    num_total = len(pairs)
    coverage = num_evaluated / num_total if num_total > 0 else 0.0

    if num_evaluated < 2:
        return {
            "spearman": 0.0,
            "pvalue": 1.0,
            "coverage": coverage,
            "num_pairs_evaluated": num_evaluated,
            "num_pairs_total": num_total,
            "dataset": label,
        }

    rho, pvalue = scipy.stats.spearmanr(human_scores, model_scores)
    rho = 0.0 if np.isnan(rho) else float(rho)
    pvalue = 1.0 if np.isnan(pvalue) else float(pvalue)

    return {
        "spearman": rho,
        "pvalue": pvalue,
        "coverage": coverage,
        "num_pairs_evaluated": num_evaluated,
        "num_pairs_total": num_total,
        "dataset": label,
    }


def eval_all_benchmarks(embeddings, word2idx, data_dir=None, verbose=True):
    results = {}
    for dataset in ("wordsim353", "simlex999"):
        try:
            results[dataset] = evaluate_similarity(
                embeddings, word2idx, dataset=dataset, data_dir=data_dir, verbose=verbose
            )
        except Exception as exc:
            results[dataset] = {"error": str(exc)}

    return results
