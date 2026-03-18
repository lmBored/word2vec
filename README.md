# Word2Vec

An implementation of Word2Vec (Skip-gram and CBOW) [1] in pure NumPy

## Table of Contents

- [Dataset](#dataset)
- [Implemented Methodds](#implemented-methods)
- [Usage](#usage)
<!-- - [Experiments](#experiments) -->
- [References](#references)


## Dataset

### Text8 Dataset

**Source**: http://mattmahoney.net/dc/text8.zip

Text8 is a standard dataset derived from the first 100MB of cleaned Wikipedia text (Mahoney, 2006) [2]. It was used in the original word2vec experiments [1].

| Property | Value |
|----------|-------|
| Tokens | 17 million |
| Vocabulary | 60,000 |
| Size | 95.4 MB |
| Content | English Wikipedia |

### Evaluation Datasets

#### WordSim-353: Word relatedness [3]
- 353 word pairs with human similarity ratings (0-10 scale)
- Measures semantic relatedness (similarity + association)
- Classic benchmark

#### SimLex-999: Word similarity [4]
- 999 word pairs rated for genuine similarity
- Distinguishes similarity from association ("coffee" and "cup" are related but not similar)
- More challenging and linguistically principled

#### Google Analogies: Word analogy accuracy [1]
- 19,544 analogy questions across 14 categories
- 5 semantic categories (capital-country, currency, etc.)
- 9 syntactic categories (verb tenses, plurals, etc.)

## Implemented Methods

+ Skip-gram with Negative Sampling (SGNS)
+ Continuous Bag of Words (CBOW) with Negative Sampling
+ JIT-compiled forward/backward and gradient update
+ Subsampling

## Usage

### Install requirements

```bash
# Install with uv
uv sync

# Or with pip
pip install -e .
```

### Training

```bash
# CBOW
uv run train_cbow.py

#SGNS
uv run train_sgns.py
```

### Evaluate

```bash
uv run python evaluate.py --model cbow --epochs 3

# To show all available flags
uv run python evaluate.py --help
```

**Success metrics**
1. **Word Relatedness**: Spearman rho > 0.5 (WordSim-353)
2. **Word Similarity**: Spearman rho > 0.3 (SimLex-999)
3. **Word Analogy**: Accuracy > 40% total (Text8 baseline)
    + Accuracy > 50% on semantic
    + Accuracy > 35% on syntactic


<!-- ## Experiments

### Transfer Learning

### Continual Learning -->

## References

[1] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient Estimation of Word Representations in Vector Space. 1st International Conference on Learning Representations (ICLR).

[2] Mahoney, M. (2006). About the Test Data / Rationale for a Large Text Compression Benchmark. *http://mattmahoney.net/dc/textdata.html*.

[3] Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z., Wolfman, G., & Ruppin, E. (2002). Placing Search in Context: The Concept Revisited. ACM Transactions on Information Systems, 20(1), 116-131.

[4] Hill, F., Reichart, R., & Korhonen, A. (2015). SimLex-999: Evaluating Semantic Models With (Genuine) Similarity Estimation. Computational Linguistics, 41(4), 665-695.
